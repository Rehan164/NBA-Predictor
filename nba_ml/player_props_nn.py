"""
Player Props Neural Network — Ensemble model for predicting PTS, REB, AST.

Architecture:
  - Deep residual network with multi-head output
  - Ensemble of 5 models with different seeds
  - Teammate availability context (injury-aware)
  - Opponent defensive profile features
  - Test-time augmentation (home/away perspective swap)

  At prediction time: average all 5 models for robust predictions.

Usage:
    # Train
    python -m nba_ml.player_props_nn

    # From web
    from nba_ml.player_props_nn import train_model, predict_today, get_model_status
"""

import json
import math
import pickle
import time
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    PLAYER_GAME_LOGS_CSV,
    PLAYER_FEATURES_CSV,
    HISTORICAL_GAMES_CSV,
    MODELS_DIR,
    RANDOM_STATE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PLAYER_PROPS_DIR = MODELS_DIR / "player_props_nn"
PLAYER_PROPS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PT_PATH = PLAYER_PROPS_DIR / "model.pt"
MODEL_META_PATH = PLAYER_PROPS_DIR / "meta.pkl"
STATUS_PATH = PLAYER_PROPS_DIR / "status.json"

# Training
TRAIN_CUTOFF = "2025-07-01"
TEST_CUTOFF = "2025-10-01"
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTHING = 0.0
N_ENSEMBLE = 5
ENSEMBLE_SEEDS = [42, 137, 256, 789, 1337]

# Architecture
HIDDEN_DIM = 256
NUM_LAYERS = 5
DROPOUT = 0.15
NUM_HEADS = 4  # for self-attention on feature groups

# Sample weighting: recent player games matter more
SAMPLE_WEIGHT_DECAY = 0.9975

# ESPN normalization
ESPN_TO_CSV = {
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "WSH": "WAS", "PHO": "PHX", "UTAH": "UTA", "BK": "BKN",
    "OC": "OKC",
}

EXCLUDE_COLS = [
    "player_id", "player_name", "team", "opponent",
    "game_id", "game_date",
    "target_pts", "target_reb", "target_ast", "actual_min",
]

TARGET_COLS = ["target_pts", "target_reb", "target_ast"]
TARGET_NAMES = ["Points", "Rebounds", "Assists"]
TARGET_SHORT = ["pts", "reb", "ast"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_features() -> pd.DataFrame:
    """Load player features CSV."""
    if not PLAYER_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Player features not found: {PLAYER_FEATURES_CSV}\n"
            "Run 'python -m nba_ml.build_player_features' first."
        )
    df = pd.read_csv(PLAYER_FEATURES_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns (everything except metadata and targets)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS and not df[c].isna().all()]


def prepare_data(df: pd.DataFrame, feature_cols: List[str]):
    """Prepare X, Y arrays with proper handling."""
    valid = df.dropna(subset=TARGET_COLS)
    X = valid[feature_cols].fillna(0).values.astype(np.float32)
    Y = valid[TARGET_COLS].values.astype(np.float32)
    dates = valid["game_date"].values
    return X, Y, dates


def compute_sample_weights(n_samples: int) -> np.ndarray:
    """Exponential decay: recent games matter more."""
    weights = np.array([
        SAMPLE_WEIGHT_DECAY ** (n_samples - 1 - i) for i in range(n_samples)
    ], dtype=np.float32)
    weights = np.clip(weights, 1e-6, None)
    weights /= weights.mean()
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Pre-norm residual block with GELU activation."""

    def __init__(self, dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return residual + x


class PlayerPropsNet(nn.Module):
    """
    Deep residual network for player prop prediction.

    Input: normalized feature vector
    Output: (PTS, REB, AST) predictions
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Residual trunk
        self.trunk = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Separate prediction heads for each stat
        self.pts_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.reb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.ast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.trunk(x)
        x = self.final_norm(x)

        pts = self.pts_head(x).squeeze(-1)
        reb = self.reb_head(x).squeeze(-1)
        ast = self.ast_head(x).squeeze(-1)

        return pts, reb, ast


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_model(
    X_train: np.ndarray, Y_train: np.ndarray, W_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    norm_mean: np.ndarray, norm_std: np.ndarray,
    y_mean: np.ndarray, y_std: np.ndarray,
    seed: int, model_idx: int,
    progress_callback=None,
) -> Tuple[dict, dict]:
    """Train a single ensemble member."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    model = PlayerPropsNet(input_dim).to(device)

    # Normalize inputs
    X_train_n = (X_train - norm_mean) / norm_std
    X_val_n = (X_val - norm_mean) / norm_std

    # Normalize targets for training stability
    Y_train_n = (Y_train - y_mean) / y_std
    Y_val_n = (Y_val - y_mean) / y_std

    # Convert to tensors
    X_tr = torch.tensor(X_train_n, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_train_n, dtype=torch.float32, device=device)
    W_tr = torch.tensor(W_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val_n, dtype=torch.float32, device=device)
    Y_v = torch.tensor(Y_val_n, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # Cosine annealing with warmup
    warmup_steps = 200
    total_steps = MAX_EPOCHS * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    step = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Shuffle
        perm = torch.randperm(len(X_tr), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_tr[idx]
            yb = Y_tr[idx]
            wb = W_tr[idx]

            pts_pred, reb_pred, ast_pred = model(xb)

            # Huber loss (robust to outliers) weighted by sample importance
            loss_pts = F.huber_loss(pts_pred, yb[:, 0], reduction="none", delta=2.0)
            loss_reb = F.huber_loss(reb_pred, yb[:, 1], reduction="none", delta=1.5)
            loss_ast = F.huber_loss(ast_pred, yb[:, 2], reduction="none", delta=1.5)

            # Weight: PTS matters most, then REB, then AST
            loss = (loss_pts * 1.0 + loss_reb * 0.8 + loss_ast * 0.8) * wb
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            vp, vr, va = model(X_v)
            val_loss_pts = F.huber_loss(vp, Y_v[:, 0], delta=2.0).item()
            val_loss_reb = F.huber_loss(vr, Y_v[:, 1], delta=1.5).item()
            val_loss_ast = F.huber_loss(va, Y_v[:, 2], delta=1.5).item()
            val_loss = val_loss_pts + val_loss_reb * 0.8 + val_loss_ast * 0.8

            # Compute MAE in original scale
            pts_mae = float(torch.abs(vp * y_std[0] + y_mean[0] - (Y_v[:, 0] * y_std[0] + y_mean[0])).mean())
            reb_mae = float(torch.abs(vr * y_std[1] + y_mean[1] - (Y_v[:, 1] * y_std[1] + y_mean[1])).mean())
            ast_mae = float(torch.abs(va * y_std[2] + y_mean[2] - (Y_v[:, 2] * y_std[2] + y_mean[2])).mean())

        if progress_callback:
            progress_callback(
                f"  Model {model_idx+1}/{N_ENSEMBLE} | Epoch {epoch+1}/{MAX_EPOCHS} | "
                f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                f"MAE — PTS: {pts_mae:.2f}, REB: {reb_mae:.2f}, AST: {ast_mae:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                if progress_callback:
                    progress_callback(f"  Model {model_idx+1}: Early stop at epoch {epoch+1}")
                break

    metrics = {"pts_mae": pts_mae, "reb_mae": reb_mae, "ast_mae": ast_mae}
    return best_state, metrics


def train_model(progress_callback=None, progress_state=None):
    """
    Train the full ensemble of player prop models.

    Returns: (list of state_dicts, metrics_dict, norm_params)
    """
    cb = progress_callback or print

    cb("Loading player features...")
    df = load_features()
    feature_cols = get_feature_cols(df)
    cb(f"  {len(df):,} rows, {len(feature_cols)} features")

    # Train/test split by date
    train_df = df[df["game_date"] < TRAIN_CUTOFF].copy()
    test_df = df[df["game_date"] >= TEST_CUTOFF].copy()
    cb(f"  Train: {len(train_df):,} (before {TRAIN_CUTOFF})")
    cb(f"  Test:  {len(test_df):,} (after {TEST_CUTOFF})")

    X_train, Y_train, train_dates = prepare_data(train_df, feature_cols)
    X_test, Y_test, test_dates = prepare_data(test_df, feature_cols)

    # Normalization params
    norm_mean = X_train.mean(axis=0)
    norm_std = X_train.std(axis=0)
    norm_std[norm_std < 1e-8] = 1.0

    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0

    # Sample weights
    W_train = compute_sample_weights(len(X_train))

    # Validation split (last 15% of training data)
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    Y_tr, Y_val = Y_train[:split_idx], Y_train[split_idx:]
    W_tr = W_train[:split_idx]

    cb(f"\n  Training {N_ENSEMBLE} ensemble models...")
    cb(f"  Architecture: {HIDDEN_DIM}-dim, {NUM_LAYERS} layers, {DROPOUT} dropout")

    if progress_state is not None:
        progress_state["total_games"] = N_ENSEMBLE * MAX_EPOCHS
        progress_state["game_idx"] = 0

    all_states = []
    all_metrics = []

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        cb(f"\n  === Ensemble member {i+1}/{N_ENSEMBLE} (seed={seed}) ===")
        state, metrics = train_single_model(
            X_tr, Y_tr, W_tr, X_val, Y_val,
            norm_mean, norm_std, y_mean, y_std,
            seed=seed, model_idx=i,
            progress_callback=cb,
        )
        all_states.append(state)
        all_metrics.append(metrics)

        if progress_state is not None:
            progress_state["game_idx"] = (i + 1) * MAX_EPOCHS
            progress_state["pct"] = int(((i + 1) / N_ENSEMBLE) * 100)

    # Evaluate ensemble on test set
    cb("\n  Evaluating ensemble on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    X_test_n = (X_test - norm_mean) / norm_std
    X_t = torch.tensor(X_test_n, dtype=torch.float32, device=device)

    all_preds = []
    for state in all_states:
        model = PlayerPropsNet(input_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pts, reb, ast = model(X_t)
            preds = torch.stack([
                pts * y_std[0] + y_mean[0],
                reb * y_std[1] + y_mean[1],
                ast * y_std[2] + y_mean[2],
            ], dim=1)
            all_preds.append(preds.cpu().numpy())

    ensemble_preds = np.mean(all_preds, axis=0)

    # Compute final metrics
    final_metrics = {}
    for i, (name, short) in enumerate(zip(TARGET_NAMES, TARGET_SHORT)):
        pred = ensemble_preds[:, i]
        actual = Y_test[:, i]
        mae = float(np.abs(pred - actual).mean())
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

        within_2 = float((np.abs(pred - actual) <= 2).mean())
        within_3 = float((np.abs(pred - actual) <= 3).mean())
        within_5 = float((np.abs(pred - actual) <= 5).mean())

        final_metrics[short] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "within_2": round(within_2 * 100, 1),
            "within_3": round(within_3 * 100, 1),
            "within_5": round(within_5 * 100, 1),
        }
        cb(f"\n  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        cb(f"    Within 2: {within_2:.1%}, Within 3: {within_3:.1%}, Within 5: {within_5:.1%}")

    # Save model
    cb("\n  Saving model...")
    torch.save(all_states, MODEL_PT_PATH)

    meta = {
        "feature_cols": feature_cols,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": input_dim,
        "metrics": final_metrics,
    }
    with open(MODEL_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    status = {
        "trained": True,
        "trained_at": datetime.now().isoformat(),
        "metrics": final_metrics,
        "ensemble_size": N_ENSEMBLE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_cols),
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

    cb("\nTRAINING COMPLETE")
    return all_states, final_metrics, meta


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_name(name: str) -> str:
    """Lowercase + strip diacritics."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii").lower()


def _load_model():
    """Load trained model ensemble and metadata."""
    if not MODEL_PT_PATH.exists() or not MODEL_META_PATH.exists():
        raise FileNotFoundError("Player props model not trained. Train it first.")

    states = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=True)
    with open(MODEL_META_PATH, "rb") as f:
        meta = pickle.load(f)

    return states, meta


def _build_player_feature_vector(
    player_name: str,
    team: str,
    opponent: str,
    is_home: int,
    feature_cols: List[str],
    player_logs: pd.DataFrame,
    games_df: pd.DataFrame,
    teammate_info: dict = None,
) -> Optional[np.ndarray]:
    """
    Build a feature vector for a single player prediction.

    Uses the player's recent game logs to compute rolling stats,
    and the opponent's defensive profile.
    """
    # Get player's recent games
    p_logs = player_logs[player_logs["PLAYER_NAME"] == player_name].sort_values(
        "GAME_DATE", ascending=False
    )
    if len(p_logs) < 5:
        return None

    # Rolling stats from recent games
    feat = {}
    feat["is_home"] = is_home

    # Rest days
    if len(p_logs) >= 2:
        last_date = pd.to_datetime(p_logs.iloc[0]["GAME_DATE"])
        today = pd.Timestamp(date.today())
        rest = (today - last_date).days
        feat["rest_days"] = min(rest, 7)
    else:
        feat["rest_days"] = 2

    # Rolling averages
    stat_cols = ["PTS", "REB", "AST", "MIN", "FGM", "FGA", "FG3M", "FG3A",
                 "FTM", "FTA", "TOV", "STL", "BLK", "PLUS_MINUS"]

    for stat in stat_cols:
        if stat not in p_logs.columns:
            continue
        vals = pd.to_numeric(p_logs[stat], errors="coerce").fillna(0).values

        for w in [3, 5, 10, 20]:
            window = vals[:w]
            if len(window) > 0:
                feat[f"{stat}_avg_l{w}"] = float(np.mean(window))
                if stat in ["PTS", "REB", "AST"]:
                    feat[f"{stat}_std_l{w}"] = float(np.std(window)) if len(window) > 1 else 0.0
                    feat[f"{stat}_max_l{w}"] = float(np.max(window))
                    feat[f"{stat}_min_l{w}"] = float(np.min(window))

    # Season averages (use all games this season)
    season_logs = p_logs.head(82)
    for stat in ["PTS", "REB", "AST", "MIN"]:
        if stat in season_logs.columns:
            feat[f"{stat}_season_avg"] = float(
                pd.to_numeric(season_logs[stat], errors="coerce").mean()
            )

    # Momentum
    for stat in ["PTS", "REB", "AST"]:
        if stat in p_logs.columns:
            vals = pd.to_numeric(p_logs[stat], errors="coerce").fillna(0).values
            avg_3 = float(np.mean(vals[:3])) if len(vals) >= 3 else 0
            avg_10 = float(np.mean(vals[:10])) if len(vals) >= 3 else 0
            feat[f"{stat}_momentum"] = avg_3 - avg_10

    # Starter proxy
    avg_min_5 = feat.get("MIN_avg_l5", 0)
    feat["is_starter"] = 1 if avg_min_5 >= 24 else 0

    # Usage proxy
    fga_avg = feat.get("FGA_avg_l10", 0)
    min_avg = feat.get("MIN_avg_l10", 1)
    feat["usage_proxy"] = (fga_avg / max(min_avg, 1)) * 48

    # Opponent defensive profile
    opp_games = games_df[
        (games_df["home_team"] == opponent) | (games_df["away_team"] == opponent)
    ].sort_values("date", ascending=False).head(20)

    if len(opp_games) >= 5:
        opp_pts_allowed = []
        opp_reb_allowed = []
        opp_ast_allowed = []
        for _, g in opp_games.iterrows():
            if g["home_team"] == opponent:
                opp_pts_allowed.append(g.get("away_score", 0))
                opp_reb_allowed.append(g.get("away_reb", 0))
                opp_ast_allowed.append(g.get("away_ast", 0))
            else:
                opp_pts_allowed.append(g.get("home_score", 0))
                opp_reb_allowed.append(g.get("home_reb", 0))
                opp_ast_allowed.append(g.get("home_ast", 0))

        for w in [10, 20]:
            feat[f"opp_def_pts_allowed_l{w}"] = float(np.mean(opp_pts_allowed[:w]))
            reb_vals = [v for v in opp_reb_allowed[:w] if v and not np.isnan(v)]
            ast_vals = [v for v in opp_ast_allowed[:w] if v and not np.isnan(v)]
            if reb_vals:
                feat[f"opp_def_reb_allowed_l{w}"] = float(np.mean(reb_vals))
            if ast_vals:
                feat[f"opp_def_ast_allowed_l{w}"] = float(np.mean(ast_vals))

    # Teammate context
    if teammate_info:
        for k, v in teammate_info.items():
            feat[k] = v

    # Build feature vector aligned to expected columns
    vector = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in feat:
            vector[i] = feat[col]

    return vector


def predict_today() -> List[dict]:
    """
    Predict player props for all today's games.

    Returns list of game dicts, each with player predictions.
    """
    import requests

    states, meta = _load_model()
    feature_cols = meta["feature_cols"]
    norm_mean = meta["norm_mean"]
    norm_std = meta["norm_std"]
    y_mean = meta["y_mean"]
    y_std = meta["y_std"]
    input_dim = meta["input_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ensemble
    models = []
    for state in states:
        model = PlayerPropsNet(input_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    # Load data
    player_logs = pd.read_csv(PLAYER_GAME_LOGS_CSV, low_memory=False)
    player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])

    games_df = pd.read_csv(HISTORICAL_GAMES_CSV, low_memory=False)
    games_df["date"] = pd.to_datetime(games_df["date"])

    # Fetch today's games from ESPN
    from datetime import timezone
    ET = timezone(timedelta(hours=-5))
    today_str = datetime.now(ET).strftime("%Y%m%d")

    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today_str}",
            timeout=10,
        )
        r.raise_for_status()
        espn_data = r.json()
    except Exception as e:
        return [{"error": f"Could not fetch games: {e}"}]

    # Fetch injuries
    injuries = {}
    try:
        r2 = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=10,
        )
        r2.raise_for_status()
        for team_obj in r2.json().get("injuries", []):
            for inj in team_obj.get("injuries", []):
                athlete = inj.get("athlete", {})
                name = athlete.get("displayName", "")
                status = inj.get("status", "").upper()
                team_info = athlete.get("team", {})
                team_abbr = team_info.get("abbreviation", "")
                if "OUT" in status:
                    if team_abbr not in injuries:
                        injuries[team_abbr] = set()
                    injuries[team_abbr].add(name)
    except Exception:
        pass

    results = []

    for event in espn_data.get("events", []):
        comp = event["competitions"][0]
        home = away = None
        for c in comp["competitors"]:
            if c["homeAway"] == "home":
                home = c
            else:
                away = c
        if not home or not away:
            continue

        home_abbr = ESPN_TO_CSV.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
        away_abbr = ESPN_TO_CSV.get(away["team"]["abbreviation"], away["team"]["abbreviation"])
        home_name = home["team"]["displayName"]
        away_name = away["team"]["displayName"]

        game_result = {
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_name": home_name,
            "away_name": away_name,
            "home_logo": home["team"].get("logo", ""),
            "away_logo": away["team"].get("logo", ""),
            "game_time": event["date"],
            "players": [],
        }

        # Get players for each team
        for team_abbr, is_home, opp_abbr in [
            (home_abbr, 1, away_abbr),
            (away_abbr, 0, home_abbr),
        ]:
            team_out = injuries.get(team_abbr, set())

            # Get recent players for this team
            team_logs = player_logs[
                player_logs["TEAM_ABBREVIATION"] == team_abbr
            ].sort_values("GAME_DATE", ascending=False)

            if team_logs.empty:
                continue

            # Get players who played recently (last 10 team games)
            recent_game_ids = team_logs["GAME_ID"].unique()[:10]
            recent = team_logs[team_logs["GAME_ID"].isin(recent_game_ids)]

            agg = (
                recent.groupby("PLAYER_NAME")
                .agg(
                    avg_min=("MIN", "mean"),
                    games=("GAME_ID", "nunique"),
                )
                .reset_index()
            )
            # Rotation players (15+ min avg)
            rotation = agg[agg["avg_min"] >= 15].sort_values("avg_min", ascending=False)

            # Compute teammate context for this game
            missing_pts = 0.0
            missing_reb = 0.0
            missing_ast = 0.0
            n_missing = 0
            total_pts = 0.0
            total_reb = 0.0
            total_ast = 0.0

            for _, row in rotation.iterrows():
                pname = row["PLAYER_NAME"]
                p_recent = recent[recent["PLAYER_NAME"] == pname]
                avg_pts = float(p_recent["PTS"].mean()) if not p_recent.empty else 0
                avg_reb = float(p_recent["REB"].mean()) if not p_recent.empty else 0
                avg_ast = float(p_recent["AST"].mean()) if not p_recent.empty else 0
                total_pts += avg_pts
                total_reb += avg_reb
                total_ast += avg_ast

                if pname in team_out:
                    n_missing += 1
                    missing_pts += avg_pts
                    missing_reb += avg_reb
                    missing_ast += avg_ast

            teammate_info = {
                "n_teammates_missing": n_missing,
                "n_rotation_players": len(rotation),
                "missing_pts_total": missing_pts,
                "missing_reb_total": missing_reb,
                "missing_ast_total": missing_ast,
                "missing_min_total": n_missing * 25,  # estimate
                "missing_pts_share": missing_pts / max(total_pts, 1),
                "missing_reb_share": missing_reb / max(total_reb, 1),
                "missing_ast_share": missing_ast / max(total_ast, 1),
                "team_shorthanded": 1 if n_missing >= 2 else 0,
            }

            # Predict for each rotation player who isn't out
            for _, row in rotation.iterrows():
                pname = row["PLAYER_NAME"]
                is_out = pname in team_out

                if is_out:
                    game_result["players"].append({
                        "name": pname,
                        "team": team_abbr,
                        "is_home": is_home,
                        "is_out": True,
                        "pred_pts": 0, "pred_reb": 0, "pred_ast": 0,
                        "avg_min": round(float(row["avg_min"]), 1),
                    })
                    continue

                fv = _build_player_feature_vector(
                    pname, team_abbr, opp_abbr, is_home,
                    feature_cols, player_logs, games_df,
                    teammate_info=teammate_info,
                )
                if fv is None:
                    continue

                # Normalize and predict with ensemble
                fv_n = (fv - norm_mean) / norm_std
                fv_t = torch.tensor(fv_n, dtype=torch.float32, device=device).unsqueeze(0)

                preds_all = []
                with torch.no_grad():
                    for m in models:
                        pts, reb, ast = m(fv_t)
                        preds_all.append([
                            float(pts[0]) * y_std[0] + y_mean[0],
                            float(reb[0]) * y_std[1] + y_mean[1],
                            float(ast[0]) * y_std[2] + y_mean[2],
                        ])

                avg_pred = np.mean(preds_all, axis=0)

                # Clamp to reasonable ranges
                pred_pts = max(0, round(float(avg_pred[0]), 1))
                pred_reb = max(0, round(float(avg_pred[1]), 1))
                pred_ast = max(0, round(float(avg_pred[2]), 1))

                # Get recent averages for comparison
                p_recent = recent[recent["PLAYER_NAME"] == pname]
                avg_pts_actual = round(float(p_recent["PTS"].mean()), 1) if not p_recent.empty else 0
                avg_reb_actual = round(float(p_recent["REB"].mean()), 1) if not p_recent.empty else 0
                avg_ast_actual = round(float(p_recent["AST"].mean()), 1) if not p_recent.empty else 0

                game_result["players"].append({
                    "name": pname,
                    "team": team_abbr,
                    "is_home": bool(is_home),
                    "is_out": False,
                    "pred_pts": pred_pts,
                    "pred_reb": pred_reb,
                    "pred_ast": pred_ast,
                    "avg_pts": avg_pts_actual,
                    "avg_reb": avg_reb_actual,
                    "avg_ast": avg_ast_actual,
                    "avg_min": round(float(row["avg_min"]), 1),
                })

        # Sort players by predicted points (descending)
        game_result["players"].sort(key=lambda p: p.get("pred_pts", 0), reverse=True)
        results.append(game_result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_status() -> dict:
    """Get current model status and metrics."""
    if not STATUS_PATH.exists():
        return {"trained": False}

    with open(STATUS_PATH) as f:
        status = json.load(f)

    return status


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Player Props Neural Network Training")
    print("=" * 60)

    train_model(progress_callback=print)


if __name__ == "__main__":
    main()
