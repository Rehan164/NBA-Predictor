"""
NBA Model Hyperparameter Tuner — Optuna-based search.

ROOT CAUSE of the train≈60% / test≈85-90% gap
----------------------------------------------
The sample weight formula  1 + (season - 2010) * 0.08  makes the training
loss dominated by the most recent seasons (which are also the test set).
The model optimises for those patterns → high test accuracy, poor fit on
the old training games → train << test.

This tuner explores the full hyperparameter space and finds the combination
where  train_acc ≈ test_acc  (both high, gap < target).  It penalises
underfitting (test >> train) and overfitting (train >> test) separately so
you can tune the penalty balance.

Usage examples
--------------
# Fastest useful run  (arch + reg + split, 1 model, 15 epochs)
python -m nba_ml.tune --trials 30 --ensemble-size 1 --max-epochs 15

# Balanced run  (all fast params, 2 models, 20 epochs)
python -m nba_ml.tune --trials 60 --ensemble-size 2 --max-epochs 20

# Deep run  (include tracker/ELO params — re-runs Phase 1 every trial)
python -m nba_ml.tune --trials 40 --tune-features --max-epochs 20

# CMA-ES sampler for fine-tuning continuous params
python -m nba_ml.tune --trials 80 --sampler cmaes --no-architecture

# Persistent study — run in parallel sessions or resume later
python -m nba_ml.tune --trials 100 --study-name nba_v1 --storage sqlite:///tuning.db

# Focus only on the gap problem (keep current architecture)
python -m nba_ml.tune --trials 50 --no-architecture --tune-split \\
       --underfit-penalty 4.0 --overfit-penalty 2.0

Flags overview
--------------
Search budget   --trials --timeout --n-startup
Objective       --gap-penalty --target-gap --underfit-penalty --overfit-penalty
Search space    --no-architecture --no-regularization --no-optimization
                --no-loss --tune-features --tune-split
Trial speed     --ensemble-size --max-epochs --patience --no-augmentation
Optuna          --sampler (tpe/random/cmaes/qmc) --pruner (median/halving/hyperband/none)
Persistence     --study-name --storage
Output          --save-best --quiet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, QMCSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner, NopPruner
except ImportError:
    print("ERROR: optuna not installed.  Run:  pip install optuna")
    sys.exit(1)

from nba_ml.advanced_model import (
    _load_all_data, build_player_game_index, _collect_features, _augment_data,
    WarmupCosineScheduler, COVID_SEASONS,
    PLAYER_INPUT_DIM, PLAYERS_PER_TEAM, TEAM_FEATURE_DIM, CONTEXT_DIM,
    ADVANCED_MODEL_DIR,
)

# ── Fixed (data-shape) dimensions ────────────────────────────────────────────
COMBINED_CTX_DIM = CONTEXT_DIM + TEAM_FEATURE_DIM * 2  # 32

# ── Default hyperparameters (current production values) ──────────────────────
DEFAULTS = {
    # architecture
    "player_embed_dim": 40,
    "team_repr_dim":    80,
    "matchup_dim":      80,
    "hidden_dim":       320,
    "num_trunk_layers": 5,
    "num_attn_heads":   4,
    # regularisation
    "dropout":          0.20,
    "weight_decay":     1e-4,
    "label_smoothing":  0.05,
    # optimiser
    "learning_rate":    8e-4,
    "batch_size":       128,
    "warmup_steps":     200,
    # loss weights
    "loss_w_win":       1.0,
    "loss_w_margin":    0.4,
    "loss_w_total":     0.2,
    # feature-collection (Phase 1)
    "player_alpha_short":   0.25,
    "player_alpha_long":    0.06,
    "team_alpha_short":     0.15,
    "team_alpha_long":      0.04,
    "elo_k":                20.0,
    "elo_home_adv":        100.0,
    "sample_weight_year":  2010,
    "sample_weight_slope":  0.08,
    # split
    "train_split": 0.85,
}

BEST_PARAMS_PATH = ADVANCED_MODEL_DIR / "best_params.json"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURABLE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.net(self.norm(x))


class TunableNBAModel(nn.Module):
    """Drop-in replacement for SequentialNBAModel with all dims configurable."""

    def __init__(self, cfg: dict):
        super().__init__()
        ped    = cfg["player_embed_dim"]
        trd    = cfg["team_repr_dim"]
        mxd    = cfg.get("matchup_dim", trd)
        hd     = cfg["hidden_dim"]
        heads  = cfg["num_attn_heads"]
        layers = cfg["num_trunk_layers"]
        drop   = cfg["dropout"]

        # num_attn_heads must divide player_embed_dim
        if ped % heads != 0:
            raise ValueError(f"player_embed_dim ({ped}) not divisible by num_attn_heads ({heads})")

        # ── Player encoder ────────────────────────────────────────────────────
        self.player_encoder = nn.Sequential(
            nn.Linear(PLAYER_INPUT_DIM, ped), nn.GELU(), nn.LayerNorm(ped),
            nn.Linear(ped, ped),              nn.GELU(), nn.LayerNorm(ped),
            nn.Linear(ped, ped),
        )

        # ── Roster aggregator (2-layer self-attention) ────────────────────────
        self.attn1 = nn.MultiheadAttention(ped, heads, dropout=drop * 0.5, batch_first=True)
        self.norm1 = nn.LayerNorm(ped)
        self.attn2 = nn.MultiheadAttention(ped, heads, dropout=drop * 0.5, batch_first=True)
        self.norm2 = nn.LayerNorm(ped)
        self.proj  = nn.Sequential(nn.Linear(ped, trd), nn.GELU(), nn.LayerNorm(trd))

        # ── Matchup interaction ───────────────────────────────────────────────
        self.matchup = nn.Sequential(
            nn.Linear(trd * 3, mxd * 2), nn.GELU(),
            nn.LayerNorm(mxd * 2),        nn.Dropout(drop),
            nn.Linear(mxd * 2, mxd),      nn.GELU(),
        )

        # ── Trunk ─────────────────────────────────────────────────────────────
        input_dim = trd * 2 + mxd + COMBINED_CTX_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hd), nn.GELU(), nn.LayerNorm(hd)
        )
        self.trunk = nn.Sequential(*[_ResBlock(hd, drop) for _ in range(layers)])

        # ── Prediction heads ──────────────────────────────────────────────────
        hi = max(32, hd // 4)
        def _head():
            return nn.Sequential(
                nn.LayerNorm(hd), nn.Linear(hd, hi), nn.GELU(),
                nn.Dropout(drop * 0.5), nn.Linear(hi, 1),
            )
        self.win_head    = _head()
        self.margin_head = _head()
        self.total_head  = _head()

    def _encode(self, roster, mask):
        emb = self.player_encoder(roster)
        pad = (mask == 0)
        x   = self.norm1(emb);  a1, _ = self.attn1(x, x, x, key_padding_mask=pad);  x = emb + a1
        x2  = self.norm2(x);    a2, _ = self.attn2(x2, x2, x2, key_padding_mask=pad); x = x + a2
        w   = mask.unsqueeze(-1) + 1e-8
        return self.proj((x * w).sum(1) / w.sum(1))

    def forward(self, hr, hm, ar, am, htf, atf, ctx):
        h  = self._encode(hr, hm)
        a  = self._encode(ar, am)
        mx = self.matchup(torch.cat([h, a, h * a], dim=1))
        x  = self.trunk(self.input_proj(torch.cat([h, a, mx, htf, atf, ctx], dim=1)))
        return self.win_head(x).squeeze(-1), self.margin_head(x).squeeze(-1), self.total_head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-MODEL TRAINING (trial-mode, supports Optuna pruning)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_trial(dataset: dict, split: int, device, cfg: dict,
                 seed: int, trial=None) -> tuple[float, float]:
    """
    Train one TunableNBAModel.  Returns (train_acc, test_acc).

    Reports intermediate test_acc to Optuna after every epoch so the
    MedianPruner can kill bad trials early.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    n   = len(dataset["win"])
    bs  = cfg["batch_size"]

    def _t(k): return torch.FloatTensor(dataset[k])
    hr, hm, ar, am     = _t("hr"), _t("hm"), _t("ar"), _t("am")
    htf, atf, ctx       = _t("htf"), _t("atf"), _t("ctx")
    tgt_w, tgt_m, tgt_t = _t("win"), _t("margin"), _t("total")
    weights              = _t("weights")

    tr_idx = torch.arange(split)
    te_idx = torch.arange(split, n)

    model     = TunableNBAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    sched = WarmupCosineScheduler(
        optimizer,
        warmup=cfg.get("warmup_steps", 200),
        total=(split // bs) * cfg["max_epochs"],
    )
    huber = nn.SmoothL1Loss(reduction="none")
    ls    = cfg["label_smoothing"]
    lw_w  = cfg["loss_w_win"]
    lw_m  = cfg["loss_w_margin"]
    lw_t  = cfg["loss_w_total"]

    best_test  = 0.0
    best_train = 0.0
    patience   = 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(perm), bs):
            bi = perm[i : i + bs]
            if len(bi) < 8:
                continue
            b = {k: v[bi].to(device) for k, v in [
                ("hr", hr), ("hm", hm), ("ar", ar), ("am", am),
                ("htf", htf), ("atf", atf), ("ctx", ctx),
                ("w", tgt_w), ("m", tgt_m), ("t", tgt_t), ("wt", weights),
            ]}
            b["wt"] = b["wt"] / b["wt"].mean()
            optimizer.zero_grad()
            wl, mp, tp = model(b["hr"], b["hm"], b["ar"], b["am"], b["htf"], b["atf"], b["ctx"])
            smooth = b["w"] * (1 - ls) + 0.5 * ls
            loss = (
                lw_w * F.binary_cross_entropy_with_logits(wl, smooth, weight=b["wt"])
                + lw_m * (huber(mp, b["m"]) * b["wt"]).mean()
                + lw_t * (huber(tp, b["t"]) * b["wt"]).mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step()

        # ── Evaluate ─────────────────────────────────────────────────────────
        model.eval()
        orig_tr = torch.arange(split // 2)   # original (non-augmented) train games
        tr_ok = te_ok = 0
        with torch.no_grad():
            for i in range(0, len(orig_tr), bs * 2):
                bi = orig_tr[i : i + bs * 2]
                wl, _, _ = model(hr[bi].to(device), hm[bi].to(device),
                                 ar[bi].to(device), am[bi].to(device),
                                 htf[bi].to(device), atf[bi].to(device), ctx[bi].to(device))
                tr_ok += ((torch.sigmoid(wl) > 0.5).float() == tgt_w[bi].to(device)).sum().item()
            for i in range(0, len(te_idx), bs * 2):
                bi = te_idx[i : i + bs * 2]
                wl, _, _ = model(hr[bi].to(device), hm[bi].to(device),
                                 ar[bi].to(device), am[bi].to(device),
                                 htf[bi].to(device), atf[bi].to(device), ctx[bi].to(device))
                te_ok += ((torch.sigmoid(wl) > 0.5).float() == tgt_w[bi].to(device)).sum().item()

        test_acc  = te_ok / (n - split)
        train_acc = tr_ok / (split // 2)

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping on test accuracy
        if test_acc > best_test:
            best_test  = test_acc
            best_train = train_acc
            patience   = 0
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break

    return best_train, best_test


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SUGGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def _suggest_cfg(trial, args) -> dict:
    """Map an Optuna trial → full config dict."""
    cfg = {}

    # ── Architecture ─────────────────────────────────────────────────────────
    if not args.no_architecture:
        heads = trial.suggest_categorical("num_attn_heads", [2, 4, 8])
        # player_embed_dim must be divisible by heads
        embed_mult = trial.suggest_int("player_embed_mult", 3, 10)
        cfg["player_embed_dim"]  = heads * embed_mult   # 6–80
        cfg["num_attn_heads"]    = heads
        cfg["team_repr_dim"]     = trial.suggest_int("team_repr_dim",  32, 192, step=16)
        cfg["matchup_dim"]       = trial.suggest_int("matchup_dim",    32, 192, step=16)
        cfg["hidden_dim"]        = trial.suggest_int("hidden_dim",    128, 512, step=32)
        cfg["num_trunk_layers"]  = trial.suggest_int("num_trunk_layers", 2, 8)
    else:
        for k in ("player_embed_dim", "num_attn_heads", "team_repr_dim",
                  "matchup_dim", "hidden_dim", "num_trunk_layers"):
            cfg[k] = DEFAULTS[k]

    # ── Regularisation ────────────────────────────────────────────────────────
    if not args.no_regularization:
        cfg["dropout"]         = trial.suggest_float("dropout",         0.05, 0.55)
        cfg["weight_decay"]    = trial.suggest_float("weight_decay",    1e-5, 5e-2, log=True)
        cfg["label_smoothing"] = trial.suggest_float("label_smoothing", 0.00, 0.20)
    else:
        for k in ("dropout", "weight_decay", "label_smoothing"):
            cfg[k] = DEFAULTS[k]

    # ── Optimiser ─────────────────────────────────────────────────────────────
    if not args.no_optimization:
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        cfg["batch_size"]    = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        cfg["warmup_steps"]  = trial.suggest_int("warmup_steps", 50, 600, step=50)
    else:
        for k in ("learning_rate", "batch_size", "warmup_steps"):
            cfg[k] = DEFAULTS[k]

    # ── Loss weights ──────────────────────────────────────────────────────────
    if not args.no_loss:
        cfg["loss_w_win"]    = trial.suggest_float("loss_w_win",    0.3, 3.0)
        cfg["loss_w_margin"] = trial.suggest_float("loss_w_margin", 0.05, 1.5)
        cfg["loss_w_total"]  = trial.suggest_float("loss_w_total",  0.05, 0.8)
    else:
        for k in ("loss_w_win", "loss_w_margin", "loss_w_total"):
            cfg[k] = DEFAULTS[k]

    # ── Feature params (tracker / ELO / sample weights — slow) ───────────────
    if args.tune_features:
        cfg["player_alpha_short"]  = trial.suggest_float("player_alpha_short",  0.05, 0.50)
        cfg["player_alpha_long"]   = trial.suggest_float("player_alpha_long",   0.01, 0.20)
        cfg["team_alpha_short"]    = trial.suggest_float("team_alpha_short",    0.05, 0.40)
        cfg["team_alpha_long"]     = trial.suggest_float("team_alpha_long",     0.01, 0.12)
        cfg["elo_k"]               = trial.suggest_float("elo_k",              5.0, 60.0)
        cfg["elo_home_adv"]        = trial.suggest_float("elo_home_adv",       0.0, 250.0)
        cfg["sample_weight_year"]  = trial.suggest_int("sample_weight_year",   2000, 2020)
        cfg["sample_weight_slope"] = trial.suggest_float("sample_weight_slope", 0.0, 0.25)
    else:
        for k in ("player_alpha_short", "player_alpha_long",
                  "team_alpha_short", "team_alpha_long",
                  "elo_k", "elo_home_adv",
                  "sample_weight_year", "sample_weight_slope"):
            cfg[k] = DEFAULTS[k]

    # ── Train/test split ratio ────────────────────────────────────────────────
    if args.tune_split:
        cfg["train_split"] = trial.suggest_float("train_split", 0.70, 0.92)
    else:
        cfg["train_split"] = DEFAULTS["train_split"]

    # ── Fixed trial budget ────────────────────────────────────────────────────
    cfg["max_epochs"]         = args.max_epochs
    cfg["early_stop_patience"] = args.patience

    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def _make_objective(base_dataset: Optional[dict], base_split: Optional[int],
                    games_ref, game_index_ref, device, args):
    """
    Return an objective function for optuna.create_study().maximize().

    Score = avg(train, test)
            - underfit_penalty * max(0, test - train - target_gap)   [test >> train]
            - overfit_penalty  * max(0, train - test - target_gap)   [train >> test]
            - gap_penalty      * max(0, |test-train| - target_gap)   [general gap]

    A trial using --tune-features re-runs Phase 1 (slow).
    Without --tune-features, base_dataset is reused (fast).
    """

    def objective(trial):
        cfg = _suggest_cfg(trial, args)

        # ── Build dataset (Phase 1 if feature params are tuned) ───────────────
        if args.tune_features:
            # Re-collect features with trial-specific tracker / ELO params
            def _noop(*a, **k): pass
            feat, _, _, _, _, _ = _collect_features(
                games_ref, game_index_ref, _noop, None, cfg=cfg
            )
            n_raw     = len(feat["win"])
            split_raw = int(n_raw * cfg["train_split"])
            train_sl  = {k: v[:split_raw] for k, v in feat.items()}
            test_sl   = {k: v[split_raw:]  for k, v in feat.items()}
            aug       = _augment_data(train_sl) if not args.no_augmentation else train_sl
            dataset   = {k: np.concatenate([aug[k], test_sl[k]], axis=0) for k in feat}
            split     = len(aug["win"])
        elif args.tune_split and base_dataset is not None:
            # Re-slice the cached raw dataset with a different split ratio
            n_raw     = len(base_dataset["win"])
            split_raw = int(n_raw * cfg["train_split"])
            train_sl  = {k: v[:split_raw] for k, v in base_dataset.items()}
            test_sl   = {k: v[split_raw:]  for k, v in base_dataset.items()}
            aug       = _augment_data(train_sl) if not args.no_augmentation else train_sl
            dataset   = {k: np.concatenate([aug[k], test_sl[k]], axis=0) for k in base_dataset}
            split     = len(aug["win"])
        else:
            dataset = base_dataset
            split   = base_split

        if dataset is None:
            raise RuntimeError("Dataset not built — run without --tune-features first or pass base data.")

        # ── Train ensemble models ─────────────────────────────────────────────
        train_accs, test_accs = [], []
        for i in range(args.ensemble_size):
            try:
                tr, te = _train_trial(
                    dataset, split, device, cfg,
                    seed=42 + i,
                    trial=(trial if i == 0 else None),   # pruning only on first model
                )
                train_accs.append(tr)
                test_accs.append(te)
            except optuna.TrialPruned:
                raise

        avg_train = float(np.mean(train_accs))
        avg_test  = float(np.mean(test_accs))
        avg_acc   = (avg_train + avg_test) / 2.0
        raw_gap   = avg_test - avg_train          # positive → test > train (underfitting)

        # ── Objective score ───────────────────────────────────────────────────
        underfit_pen = max(0.0,  raw_gap - args.target_gap) * args.underfit_penalty
        overfit_pen  = max(0.0, -raw_gap - args.target_gap) * args.overfit_penalty
        gap_pen      = max(0.0, abs(raw_gap) - args.target_gap) * args.gap_penalty
        score = avg_acc - underfit_pen - overfit_pen - gap_pen

        # Store for easy inspection
        trial.set_user_attr("train_acc", round(avg_train, 4))
        trial.set_user_attr("test_acc",  round(avg_test,  4))
        trial.set_user_attr("gap",       round(raw_gap,   4))
        trial.set_user_attr("avg_acc",   round(avg_acc,   4))
        trial.set_user_attr("score",     round(score,     4))

        return score

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TUNING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(args):
    print("=" * 70)
    print("  NBA Model Hyperparameter Tuner")
    print("=" * 70)
    print(f"  Trials:        {args.trials}")
    print(f"  Ensemble size: {args.ensemble_size} model(s) per trial")
    print(f"  Max epochs:    {args.max_epochs}   Patience: {args.patience}")
    print(f"  Sampler:       {args.sampler}   Pruner: {args.pruner}")
    print(f"  Augmentation:  {'off' if args.no_augmentation else 'on'}")
    print(f"  Search space:  arch={not args.no_architecture}  reg={not args.no_regularization}"
          f"  opt={not args.no_optimization}  loss={not args.no_loss}"
          f"  features={args.tune_features}  split={args.tune_split}")
    print(f"  Penalties:     gap={args.gap_penalty}  underfit={args.underfit_penalty}"
          f"  overfit={args.overfit_penalty}  target_gap={args.target_gap:.0%}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data (always needed for Phase 1) ─────────────────────────────────
    print("\nLoading historical data…")
    games, logs = _load_all_data()
    games = games[~games["season"].isin(COVID_SEASONS)].reset_index(drop=True)
    print(f"  {len(games):,} games after COVID filter")

    print("Building player-game index…")
    game_index = build_player_game_index(logs)

    # ── Phase 1: collect features (cache unless we re-run per trial) ──────────
    base_dataset = base_split = None
    if not args.tune_features:
        print("\nPhase 1: collecting features (cached for all trials)…")
        t0 = time.time()

        def _noop(*a, **k): pass
        raw_dataset, _, _, _, _, _ = _collect_features(games, game_index, _noop, None)

        n_raw     = len(raw_dataset["win"])
        split_raw = int(n_raw * DEFAULTS["train_split"])
        train_sl  = {k: v[:split_raw] for k, v in raw_dataset.items()}
        test_sl   = {k: v[split_raw:]  for k, v in raw_dataset.items()}
        aug       = _augment_data(train_sl) if not args.no_augmentation else train_sl
        base_dataset = {k: np.concatenate([aug[k], test_sl[k]], axis=0) for k in raw_dataset}
        base_split   = len(aug["win"])

        print(f"  Features ready in {time.time()-t0:.0f}s  "
              f"| Train: {base_split:,}  Test: {len(base_dataset['win'])-base_split:,}")

    # ── Build Optuna sampler ──────────────────────────────────────────────────
    n_startup = args.n_startup
    sampler_name = args.sampler.lower()
    if sampler_name == "tpe":
        sampler = TPESampler(n_startup_trials=n_startup, seed=args.seed)
    elif sampler_name == "random":
        sampler = RandomSampler(seed=args.seed)
    elif sampler_name == "cmaes":
        sampler = CmaEsSampler(n_startup_trials=n_startup, seed=args.seed, warn_independent_sampling=False)
    elif sampler_name == "qmc":
        sampler = QMCSampler(seed=args.seed)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}  (choices: tpe, random, cmaes, qmc)")

    # ── Build Optuna pruner ───────────────────────────────────────────────────
    pruner_name = args.pruner.lower()
    if pruner_name == "median":
        pruner = MedianPruner(n_startup_trials=max(5, n_startup), n_warmup_steps=3)
    elif pruner_name == "halving":
        pruner = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
    elif pruner_name == "hyperband":
        pruner = HyperbandPruner(min_resource=2, max_resource=args.max_epochs, reduction_factor=3)
    elif pruner_name in ("none", "nop"):
        pruner = NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {args.pruner}  (choices: median, halving, hyperband, none)")

    # ── Create or load study ──────────────────────────────────────────────────
    study_kwargs = dict(
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    if not args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence internal noise

    # ── Objective ─────────────────────────────────────────────────────────────
    objective = _make_objective(
        base_dataset, base_split,
        games, game_index,
        device, args,
    )

    # ── Run optimisation ──────────────────────────────────────────────────────
    print(f"\nStarting {args.trials} trials…  (Ctrl-C to stop early, best params still saved)\n")
    t_start = time.time()

    def _callback(study, trial):
        if args.quiet:
            return
        ta  = trial.user_attrs.get("train_acc", float("nan"))
        te  = trial.user_attrs.get("test_acc",  float("nan"))
        gap = trial.user_attrs.get("gap",        float("nan"))
        sc  = trial.user_attrs.get("score",      float("nan"))
        bt  = study.best_trial
        bsc = bt.value if bt else float("nan")
        bta = bt.user_attrs.get("train_acc", float("nan")) if bt else float("nan")
        bte = bt.user_attrs.get("test_acc",  float("nan")) if bt else float("nan")
        state_sym = {"COMPLETE": "✓", "PRUNED": "✗", "FAIL": "!"}.get(trial.state.name, "?")
        print(
            f"  [{trial.number:3d}] {state_sym} "
            f"train={ta:.1%}  test={te:.1%}  gap={gap:+.1%}  score={sc:.4f} "
            f"│ best: train={bta:.1%} test={bte:.1%} score={bsc:.4f}"
        )

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            timeout=args.timeout,
            callbacks=[_callback],
            show_progress_bar=args.quiet,   # progress bar when quiet, callback when verbose
        )
    except KeyboardInterrupt:
        print("\n  Interrupted — using best trial found so far.")

    # ── Results ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    best    = study.best_trial
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]

    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"  Trials: {len(study.trials)} total  |  {len(completed)} complete  |  {len(pruned)} pruned")
    print(f"  Time:   {elapsed/60:.1f} min")
    print()
    print(f"  Best trial #{best.number}")
    print(f"    Train accuracy : {best.user_attrs.get('train_acc', '?'):.1%}")
    print(f"    Test  accuracy : {best.user_attrs.get('test_acc',  '?'):.1%}")
    print(f"    Gap (test-train): {best.user_attrs.get('gap', float('nan')):+.1%}")
    print(f"    Score          : {best.value:.4f}")
    print()
    print("  Best parameters:")
    for k, v in sorted(best.params.items()):
        default_v = DEFAULTS.get(k)
        marker = " ◄" if default_v is not None and abs(float(v) - float(default_v)) > 1e-6 else ""
        print(f"    {k:<28s} = {v}{marker}")

    # ── Save best params ───────────────────────────────────────────────────────
    save_path = Path(args.save_best)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Reconstruct full cfg (includes non-tuned defaults) from best trial
    class _FakeTrial:
        """Replay best trial params without calling suggest_*."""
        def __init__(self, params):
            self._params = params
        def suggest_categorical(self, name, choices):
            return self._params.get(name, choices[0])
        def suggest_int(self, name, lo, hi, step=1):
            return self._params.get(name, lo)
        def suggest_float(self, name, lo, hi, log=False):
            return self._params.get(name, lo)
        def report(self, *a): pass
        def should_prune(self): return False

    best_full_cfg = _suggest_cfg(_FakeTrial(best.params), args)
    output = {
        "trial_number":  best.number,
        "score":         round(best.value, 6),
        "train_acc":     best.user_attrs.get("train_acc"),
        "test_acc":      best.user_attrs.get("test_acc"),
        "gap":           best.user_attrs.get("gap"),
        "tuned_params":  best.params,
        "full_config":   best_full_cfg,
        "search_args": {
            "trials": args.trials, "ensemble_size": args.ensemble_size,
            "max_epochs": args.max_epochs, "sampler": args.sampler,
            "tune_features": args.tune_features, "tune_split": args.tune_split,
        },
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Best params saved → {save_path}")
    print(f"\n  To apply: edit nba_ml/advanced_model.py constants with the values above,")
    print(f"  then run:  python -m web.app  (Model page → Train)")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="NBA Model Hyperparameter Tuner (Optuna)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Search budget ─────────────────────────────────────────────────────────
    bud = p.add_argument_group("Search budget")
    bud.add_argument("--trials",    type=int,   default=50,
                     help="Number of Optuna trials (default: 50)")
    bud.add_argument("--timeout",   type=float, default=None,
                     help="Max wall-clock seconds (default: no limit)")
    bud.add_argument("--n-startup", type=int,   default=10,
                     help="Random startup trials before TPE/CMA-ES (default: 10)")
    bud.add_argument("--seed",      type=int,   default=42,
                     help="Random seed (default: 42)")

    # ── Objective tuning ──────────────────────────────────────────────────────
    obj = p.add_argument_group("Objective penalties  (higher = stricter)")
    obj.add_argument("--gap-penalty",      type=float, default=1.5,
                     help="General |train-test| gap penalty weight (default: 1.5)")
    obj.add_argument("--underfit-penalty", type=float, default=3.0,
                     help="Penalty when test >> train  (current problem, default: 3.0)")
    obj.add_argument("--overfit-penalty",  type=float, default=2.0,
                     help="Penalty when train >> test  (default: 2.0)")
    obj.add_argument("--target-gap",       type=float, default=0.05,
                     help="Acceptable |train-test| fraction before penalties apply (default: 0.05)")

    # ── Search space ──────────────────────────────────────────────────────────
    sp = p.add_argument_group("Search space (disable to keep production defaults)")
    sp.add_argument("--no-architecture",   action="store_true",
                    help="Skip tuning hidden dims / attention heads / trunk depth")
    sp.add_argument("--no-regularization", action="store_true",
                    help="Skip tuning dropout / weight_decay / label_smoothing")
    sp.add_argument("--no-optimization",   action="store_true",
                    help="Skip tuning learning_rate / batch_size / warmup_steps")
    sp.add_argument("--no-loss",           action="store_true",
                    help="Skip tuning win/margin/total loss weights")
    sp.add_argument("--tune-features",     action="store_true",
                    help="Tune tracker alphas + ELO + sample weights  "
                         "(WARNING: re-runs Phase 1 per trial — very slow)")
    sp.add_argument("--tune-split",        action="store_true",
                    help="Tune train/test split ratio (fast — no Phase 1 re-run)")

    # ── Trial speed ───────────────────────────────────────────────────────────
    sp2 = p.add_argument_group("Trial speed")
    sp2.add_argument("--ensemble-size",  type=int, default=1,
                     help="Models trained per trial for averaging (default: 1, use 2+ for stability)")
    sp2.add_argument("--max-epochs",     type=int, default=20,
                     help="Max training epochs per trial (default: 20)")
    sp2.add_argument("--patience",       type=int, default=5,
                     help="Early-stopping patience per trial (default: 5)")
    sp2.add_argument("--no-augmentation", action="store_true",
                     help="Disable home/away data augmentation (cuts dataset in half, faster)")

    # ── Optuna settings ───────────────────────────────────────────────────────
    opt = p.add_argument_group("Optuna settings")
    opt.add_argument("--sampler",    default="tpe",
                     choices=["tpe", "random", "cmaes", "qmc"],
                     help="Optuna sampler (default: tpe)")
    opt.add_argument("--pruner",     default="median",
                     choices=["median", "halving", "hyperband", "none"],
                     help="Trial pruner — kills bad trials early (default: median)")
    opt.add_argument("--study-name", default="nba_tune",
                     help="Study name (used with --storage, default: nba_tune)")
    opt.add_argument("--storage",    default=None,
                     help="Optuna storage URL e.g. sqlite:///tuning.db  (enables persistence / resuming)")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--save-best", default=str(BEST_PARAMS_PATH),
                     help=f"Path to save best params JSON (default: {BEST_PARAMS_PATH})")
    out.add_argument("--quiet", action="store_true",
                     help="Suppress per-trial output (show Optuna progress bar instead)")

    args = p.parse_args()
    tune_hyperparameters(args)


if __name__ == "__main__":
    main()
