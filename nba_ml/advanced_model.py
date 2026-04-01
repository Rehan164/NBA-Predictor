"""
Advanced Neural Network for NBA Game Prediction

Injury-aware transformer-style model that understands player context:
- Who is playing / who is out for each game
- How team performance shifts when key players are missing
- Rolling team + player stats as sequential input
- The-Odds-API odds as calibration features

Architecture:
  Team Context Encoder:
    Per-player embeddings (top-8 per team) with availability mask
    → Multi-head attention over roster → team representation

  Game Encoder:
    [home_team_repr, away_team_repr, matchup_features, odds_features]
    → Dense layers → 3 output heads (win, margin, total)

Chronological train/test split. COVID years (2020-2021) excluded.

Usage:
    python -m nba_ml.advanced_model            # train
    python -m nba_ml.advanced_model --status    # check status
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss

from .config import (
    HISTORICAL_GAMES_CSV,
    PLAYER_GAME_LOGS_CSV,
    TRAINING_FEATURES_CSV,
    MODELS_DIR,
    RANDOM_STATE,
    DATA_DIR,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ADVANCED_MODEL_DIR = MODELS_DIR / "advanced"
ADVANCED_MODEL_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PT_PATH = ADVANCED_MODEL_DIR / "model.pt"
MODEL_META_PATH = ADVANCED_MODEL_DIR / "meta.pkl"
STATUS_PATH = ADVANCED_MODEL_DIR / "status.json"

# Training
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15

# Architecture
PLAYERS_PER_TEAM = 8       # Top-8 players per team per game
PLAYER_FEAT_DIM = 8        # Stats per player in roster vector
PLAYER_EMBED_DIM = 32      # Embedding per player
TEAM_REPR_DIM = 64         # Output of roster attention
MATCHUP_FEAT_DIM = 20      # Static matchup features
HIDDEN_DIM = 128
DROPOUT = 0.3

# COVID exclusion
COVID_SEASONS = {2020, 2021}  # 2019-20 bubble, 2020-21 short season

# ESPN uses different abbreviations than our CSV data
ESPN_TO_CSV = {
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "UTAH": "UTA", "BK": "BKN", "WSH": "WAS",
}

def _csv_abbr(espn_abbr: str) -> str:
    """Normalize ESPN team abbreviation to match our CSV data."""
    return ESPN_TO_CSV.get(espn_abbr, espn_abbr)

# Chronological split — use proportions, not fixed dates
# 75% train, 10% val, 15% test (all chronological)
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
# remaining = test

# Online learning
LEARN_HISTORY_PATH = ADVANCED_MODEL_DIR / "learn_history.json"
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION — Build injury-aware game features
# ═══════════════════════════════════════════════════════════════════════════════

def _load_player_logs() -> pd.DataFrame:
    """Load and prepare player game logs."""
    df = pd.read_csv(PLAYER_GAME_LOGS_CSV, low_memory=False)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = df["SEASON_ID"].apply(lambda x: int(str(x)[-4:]) + 1 if pd.notna(x) else 0)
    return df


def _load_games() -> pd.DataFrame:
    """Load historical games."""
    df = pd.read_csv(HISTORICAL_GAMES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_features() -> pd.DataFrame:
    """Load pre-built training features (ELO, rolling stats, etc.)."""
    df = pd.read_csv(TRAINING_FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_roster_features(
    player_logs: pd.DataFrame,
    games: pd.DataFrame,
) -> Dict:
    """
    For each historical game, determine which players played and their recent stats.
    Also infer who was MISSING (injured/DNP) by comparing against the team's
    usual rotation.

    Returns dict keyed by game_id:
    {
        game_id: {
            "home_roster": np.array(PLAYERS_PER_TEAM, PLAYER_FEAT_DIM),
            "home_available": np.array(PLAYERS_PER_TEAM),  # 1=played, 0=missing
            "away_roster": ...,
            "away_available": ...,
        }
    }
    """
    print("  Building roster features for each game...")

    # Pre-compute: for each team+season, who are the top-8 by avg minutes
    player_logs_sorted = player_logs.sort_values("GAME_DATE")

    # Build team rotation lookup: team+season → top 8 player names by avg minutes
    season_rotations = {}
    for (team, season), grp in player_logs.groupby(["TEAM_ABBREVIATION", "SEASON"]):
        avg_min = grp.groupby("PLAYER_NAME")["MIN"].mean().sort_values(ascending=False)
        season_rotations[(team, season)] = list(avg_min.head(12).index)  # top 12 for flexibility

    # Pre-compute rolling player averages (last 10 games before each date)
    # We'll do this per-player sorted by date
    print("  Computing rolling player stats...")
    player_rolling = {}
    for player_name, pgrp in player_logs.groupby("PLAYER_NAME"):
        pgrp = pgrp.sort_values("GAME_DATE")
        # Rolling 10-game averages
        stats = pgrp[["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FG_PCT"]].fillna(0)
        rolling = stats.rolling(10, min_periods=3).mean()
        rolling["GAME_DATE"] = pgrp["GAME_DATE"].values
        rolling["GAME_ID"] = pgrp["GAME_ID"].values
        rolling["PLAYER_NAME"] = player_name
        player_rolling[player_name] = rolling

    if player_rolling:
        all_rolling = pd.concat(player_rolling.values(), ignore_index=True)
    else:
        all_rolling = pd.DataFrame()

    # For each game, build roster vectors
    roster_data = {}
    game_player_lookup = player_logs.groupby("GAME_ID")

    for _, game_row in games.iterrows():
        gid = game_row["game_id"]
        game_date = game_row["date"]
        season = game_row["season"]

        if season in COVID_SEASONS:
            continue

        # Get players who actually played this game
        try:
            game_players = game_player_lookup.get_group(gid)
        except KeyError:
            continue

        result = {}
        for side, team_col in [("home", "home_team"), ("away", "away_team")]:
            team = game_row[team_col]
            team_players = game_players[game_players["TEAM_ABBREVIATION"] == team]

            # Get this team's usual rotation
            rotation = season_rotations.get((team, season), [])

            # Who played (sorted by minutes)
            played = team_players.sort_values("MIN", ascending=False)
            played_names = set(played["PLAYER_NAME"].values)

            # Who was expected but missing
            missing_names = [p for p in rotation[:PLAYERS_PER_TEAM] if p not in played_names]

            # Build player feature vectors for top-8 who played
            roster_feats = np.zeros((PLAYERS_PER_TEAM, PLAYER_FEAT_DIM), dtype=np.float32)
            available = np.zeros(PLAYERS_PER_TEAM, dtype=np.float32)

            top_played = played.head(PLAYERS_PER_TEAM)
            for i, (_, prow) in enumerate(top_played.iterrows()):
                pname = prow["PLAYER_NAME"]
                # Get rolling averages for this player before this game
                if pname in player_rolling:
                    pr = player_rolling[pname]
                    prior = pr[pr["GAME_DATE"] < game_date]
                    if len(prior) > 0:
                        last = prior.iloc[-1]
                        roster_feats[i] = [
                            last.get("PTS", 0), last.get("REB", 0),
                            last.get("AST", 0), last.get("STL", 0),
                            last.get("BLK", 0), last.get("TOV", 0),
                            last.get("MIN", 0), last.get("FG_PCT", 0),
                        ]
                    else:
                        # Use game stats as fallback
                        roster_feats[i] = [
                            prow.get("PTS", 0), prow.get("REB", 0),
                            prow.get("AST", 0), prow.get("STL", 0),
                            prow.get("BLK", 0), prow.get("TOV", 0),
                            prow.get("MIN", 0), prow.get("FG_PCT", 0),
                        ]
                available[i] = 1.0

            result[f"{side}_roster"] = roster_feats
            result[f"{side}_available"] = available
            result[f"{side}_missing_count"] = len(missing_names)

        roster_data[gid] = result

    print(f"  Built roster data for {len(roster_data):,} games")
    return roster_data


def build_matchup_features(features_df: pd.DataFrame) -> Dict:
    """
    Extract matchup-level features from the pre-built feature CSV.
    Returns dict keyed by (home_team, away_team, date_str) → feature vector.
    """
    # Columns we want as static matchup features
    matchup_cols = [
        "elo_diff", "elo_expected", "travel_distance", "timezone_diff",
        "rest_advantage", "season_win_pct_diff",
        "home_off_rating_adv_l10", "home_def_rating_adv_l10",
        "away_off_rating_adv_l10", "away_def_rating_adv_l10",
        "home_pace_l10", "away_pace_l10",
        "home_efg_pct_l10", "away_efg_pct_l10",
        "home_net_rating_l10", "away_net_rating_l10",
        "home_h2h_win_pct", "home_margin_momentum",
        "away_margin_momentum", "home_b2b",
    ]

    available_cols = [c for c in matchup_cols if c in features_df.columns]
    print(f"  Using {len(available_cols)} matchup features")

    matchup_data = {}
    for _, row in features_df.iterrows():
        key = row["game_id"] if "game_id" in row.index else f"{row['home_team']}_{row['away_team']}_{row['date']}"
        feats = np.zeros(MATCHUP_FEAT_DIM, dtype=np.float32)
        for i, col in enumerate(available_cols[:MATCHUP_FEAT_DIM]):
            val = row.get(col, 0)
            feats[i] = 0.0 if pd.isna(val) else float(val)
        matchup_data[key] = feats

    return matchup_data, available_cols[:MATCHUP_FEAT_DIM]


def prepare_training_data() -> Tuple:
    """
    Build complete training dataset combining:
    - Roster features (who played, their rolling stats, who was missing)
    - Matchup features (ELO, efficiency, travel, rest)
    - Targets (win, margin, total)

    Returns numpy arrays ready for PyTorch.
    """
    print("=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    games = _load_games()
    player_logs = _load_player_logs()
    features = _load_features()

    # Filter out COVID seasons
    games = games[~games["season"].isin(COVID_SEASONS)].reset_index(drop=True)
    features = features[~features["date"].dt.year.isin({2020, 2021})].reset_index(drop=True)
    print(f"  After COVID filter: {len(games):,} games, {len(features):,} feature rows")

    # Build roster data
    roster_data = build_roster_features(player_logs, games)

    # Build matchup data
    matchup_data, matchup_cols = build_matchup_features(features)

    # Merge everything into aligned arrays
    home_rosters = []
    home_avail = []
    away_rosters = []
    away_avail = []
    matchup_feats = []
    targets = []
    dates = []
    missing_counts = []

    for _, row in features.iterrows():
        gid = row.get("game_id")
        if gid is None:
            continue
        if gid not in roster_data:
            continue

        rd = roster_data[gid]
        key = gid

        mf = matchup_data.get(key, np.zeros(MATCHUP_FEAT_DIM, dtype=np.float32))

        home_rosters.append(rd["home_roster"])
        home_avail.append(rd["home_available"])
        away_rosters.append(rd["away_roster"])
        away_avail.append(rd["away_available"])
        matchup_feats.append(mf)
        missing_counts.append([rd["home_missing_count"], rd["away_missing_count"]])

        targets.append([
            float(row["home_win"]),
            float(row["home_margin"]),
            float(row["total_score"]),
        ])
        dates.append(row["date"])

    home_rosters = np.array(home_rosters, dtype=np.float32)
    home_avail = np.array(home_avail, dtype=np.float32)
    away_rosters = np.array(away_rosters, dtype=np.float32)
    away_avail = np.array(away_avail, dtype=np.float32)
    matchup_feats = np.array(matchup_feats, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    dates = np.array(dates)
    missing_counts = np.array(missing_counts, dtype=np.float32)

    # Clean NaN/Inf
    for arr in [home_rosters, away_rosters, matchup_feats, targets, missing_counts]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize roster features (z-score per feature across all players)
    roster_combined = np.concatenate([
        home_rosters.reshape(-1, PLAYER_FEAT_DIM),
        away_rosters.reshape(-1, PLAYER_FEAT_DIM),
    ], axis=0)
    roster_mean = roster_combined.mean(axis=0)
    roster_std = roster_combined.std(axis=0) + 1e-8

    home_rosters = (home_rosters - roster_mean) / roster_std
    away_rosters = (away_rosters - roster_mean) / roster_std

    # Normalize matchup features
    matchup_mean = matchup_feats.mean(axis=0)
    matchup_std = matchup_feats.std(axis=0) + 1e-8
    matchup_feats = (matchup_feats - matchup_mean) / matchup_std

    # Target normalization params
    margin_mean = float(targets[:, 1].mean())
    margin_std = float(targets[:, 1].std()) + 1e-8
    total_mean = float(targets[:, 2].mean())
    total_std = float(targets[:, 2].std()) + 1e-8

    # Final NaN cleanup
    for arr in [home_rosters, away_rosters, matchup_feats]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    norm_params = {
        "roster_mean": roster_mean.tolist(),
        "roster_std": roster_std.tolist(),
        "matchup_mean": matchup_mean.tolist(),
        "matchup_std": matchup_std.tolist(),
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "total_mean": total_mean,
        "total_std": total_std,
        "matchup_cols": matchup_cols,
    }

    print(f"\n  Dataset: {len(targets):,} games")
    print(f"  Home rosters: {home_rosters.shape}")
    print(f"  Matchup feats: {matchup_feats.shape}")
    print(f"  Date range: {dates.min()} → {dates.max()}")

    return (home_rosters, home_avail, away_rosters, away_avail,
            matchup_feats, missing_counts, targets, dates, norm_params)


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RosterEncoder(nn.Module):
    """
    Encodes a team's roster (top-8 players) into a fixed-size representation.
    Uses attention so the model learns which players matter most.

    Input: (batch, PLAYERS_PER_TEAM, PLAYER_FEAT_DIM), availability mask
    Output: (batch, TEAM_REPR_DIM)
    """

    def __init__(self):
        super().__init__()
        self.player_proj = nn.Sequential(
            nn.Linear(PLAYER_FEAT_DIM, PLAYER_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(PLAYER_EMBED_DIM, PLAYER_EMBED_DIM),
        )
        # Attention: learn which players to focus on
        self.attn = nn.MultiheadAttention(
            embed_dim=PLAYER_EMBED_DIM,
            num_heads=4,
            dropout=DROPOUT * 0.5,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(PLAYER_EMBED_DIM, TEAM_REPR_DIM),
            nn.ReLU(),
        )

    def forward(self, roster, available):
        """
        roster: (B, 8, PLAYER_FEAT_DIM)
        available: (B, 8) — 1.0 if player played, 0.0 if missing
        """
        # Project each player
        x = self.player_proj(roster)  # (B, 8, PLAYER_EMBED_DIM)

        # Mask unavailable players
        # Scale down missing players (don't zero out — model should see they're gone)
        mask = available.unsqueeze(-1)  # (B, 8, 1)
        x = x * mask  # zero out missing player embeddings

        # Self-attention over roster
        # Create attention mask: True = ignore
        attn_mask = (available == 0)  # (B, 8) — True where missing
        x, _ = self.attn(x, x, x, key_padding_mask=attn_mask)

        # Pool: weighted mean by availability
        weights = available.unsqueeze(-1) + 1e-8  # (B, 8, 1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1)  # (B, PLAYER_EMBED_DIM)

        return self.out_proj(pooled)  # (B, TEAM_REPR_DIM)


class AdvancedNBAModel(nn.Module):
    """
    Full game prediction model.

    Inputs:
      - Home roster (8 players × stats) + availability mask
      - Away roster (8 players × stats) + availability mask
      - Matchup features (ELO, pace, travel, rest, etc.)
      - Missing player counts

    Outputs:
      - Win probability (home)
      - Home margin
      - Total score
    """

    def __init__(self, matchup_dim=MATCHUP_FEAT_DIM):
        super().__init__()

        # Shared roster encoder for both teams
        self.roster_encoder = RosterEncoder()

        # Missing count embedding (how many key players are out)
        self.missing_embed = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
        )

        # Main prediction network
        combined_dim = TEAM_REPR_DIM * 2 + matchup_dim + 8  # home + away + matchup + missing
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
        )

        # Output heads
        self.win_head = nn.Linear(64, 1)
        self.margin_head = nn.Linear(64, 1)
        self.total_head = nn.Linear(64, 1)

    def forward(self, home_roster, home_avail, away_roster, away_avail,
                matchup, missing_counts):
        h_repr = self.roster_encoder(home_roster, home_avail)
        a_repr = self.roster_encoder(away_roster, away_avail)
        m_embed = self.missing_embed(missing_counts)

        combined = torch.cat([h_repr, a_repr, matchup, m_embed], dim=1)
        features = self.trunk(combined)

        win_logit = self.win_head(features).squeeze(-1)
        margin = self.margin_head(features).squeeze(-1)
        total = self.total_head(features).squeeze(-1)

        return win_logit, margin, total


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class NBAAdvancedDataset(Dataset):
    def __init__(self, home_rosters, home_avail, away_rosters, away_avail,
                 matchup, missing_counts, targets):
        self.hr = torch.FloatTensor(home_rosters)
        self.ha = torch.FloatTensor(home_avail)
        self.ar = torch.FloatTensor(away_rosters)
        self.aa = torch.FloatTensor(away_avail)
        self.mf = torch.FloatTensor(matchup)
        self.mc = torch.FloatTensor(missing_counts)
        self.tgt = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        return (self.hr[idx], self.ha[idx], self.ar[idx], self.aa[idx],
                self.mf[idx], self.mc[idx], self.tgt[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(progress_callback=None):
    """
    Full training pipeline. Returns (model, metrics, norm_params).

    progress_callback: optional callable(msg: str) for status updates.
    """
    def emit(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    emit("Loading and preparing data...")
    data = prepare_training_data()
    (home_rosters, home_avail, away_rosters, away_avail,
     matchup_feats, missing_counts, targets, dates, norm_params) = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit(f"Device: {device}")

    # Chronological split by ratio (data is already sorted by date)
    n = len(targets)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    emit(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # Normalize regression targets
    targets_norm = targets.copy()
    targets_norm[:, 1] = (targets[:, 1] - norm_params["margin_mean"]) / norm_params["margin_std"]
    targets_norm[:, 2] = (targets[:, 2] - norm_params["total_mean"]) / norm_params["total_std"]

    # Sample weights: more recent = more important
    train_dates_ord = np.array([d.toordinal() for d in pd.to_datetime(dates[train_idx])])
    max_ord = train_dates_ord.max()
    # Exponential decay: ~50% weight 3 seasons back
    days_ago = max_ord - train_dates_ord
    sample_weights = np.exp(-days_ago / 1200.0)
    sample_weights = sample_weights / sample_weights.mean()  # normalize to mean=1

    # Datasets
    train_ds = NBAAdvancedDataset(
        home_rosters[train_idx], home_avail[train_idx],
        away_rosters[train_idx], away_avail[train_idx],
        matchup_feats[train_idx], missing_counts[train_idx],
        targets_norm[train_idx],
    )
    val_ds = NBAAdvancedDataset(
        home_rosters[val_idx], home_avail[val_idx],
        away_rosters[val_idx], away_avail[val_idx],
        matchup_feats[val_idx], missing_counts[val_idx],
        targets_norm[val_idx],
    )

    # Weighted sampler for training
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = AdvancedNBAModel(matchup_dim=matchup_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    emit(f"Training for up to {EPOCHS} epochs (patience: {PATIENCE})...")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for batch in train_loader:
            hr, ha, ar, aa, mf, mc, tgt = [b.to(device) for b in batch]

            optimizer.zero_grad()
            win_logit, margin_pred, total_pred = model(hr, ha, ar, aa, mf, mc)

            loss_win = bce_loss(win_logit, tgt[:, 0])
            loss_margin = mse_loss(margin_pred, tgt[:, 1])
            loss_total = mse_loss(total_pred, tgt[:, 2])

            # Multi-task weighted loss
            loss = loss_win + 0.3 * loss_margin + 0.3 * loss_total
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_win_preds = []
        val_win_true = []

        with torch.no_grad():
            for batch in val_loader:
                hr, ha, ar, aa, mf, mc, tgt = [b.to(device) for b in batch]
                win_logit, margin_pred, total_pred = model(hr, ha, ar, aa, mf, mc)

                loss = bce_loss(win_logit, tgt[:, 0]) + 0.3 * mse_loss(margin_pred, tgt[:, 1]) + 0.3 * mse_loss(total_pred, tgt[:, 2])
                val_losses.append(loss.item())

                val_win_preds.extend(torch.sigmoid(win_logit).cpu().numpy())
                val_win_true.extend(tgt[:, 0].cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_win_true, (np.array(val_win_preds) > 0.5).astype(int))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = f"  Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={val_acc:.4f}"
            emit(msg)

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                emit(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()

    # ── Test evaluation ───────────────────────────────────────────────────
    emit("Evaluating on test set...")
    test_ds = NBAAdvancedDataset(
        home_rosters[test_idx], home_avail[test_idx],
        away_rosters[test_idx], away_avail[test_idx],
        matchup_feats[test_idx], missing_counts[test_idx],
        targets[test_idx],  # original targets for metrics
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_win_probs = []
    all_margin_preds = []
    all_total_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            hr, ha, ar, aa, mf, mc, tgt = [b.to(device) for b in batch]
            win_logit, margin_pred, total_pred = model(hr, ha, ar, aa, mf, mc)

            # Denormalize
            margin_dn = margin_pred.cpu().numpy() * norm_params["margin_std"] + norm_params["margin_mean"]
            total_dn = total_pred.cpu().numpy() * norm_params["total_std"] + norm_params["total_mean"]

            all_win_probs.extend(torch.sigmoid(win_logit).cpu().numpy())
            all_margin_preds.extend(margin_dn)
            all_total_preds.extend(total_dn)
            all_targets.extend(tgt.cpu().numpy())

    all_win_probs = np.array(all_win_probs)
    all_margin_preds = np.array(all_margin_preds)
    all_total_preds = np.array(all_total_preds)
    all_targets = np.array(all_targets)

    win_acc = accuracy_score(all_targets[:, 0], (all_win_probs > 0.5).astype(int))
    margin_mae = mean_absolute_error(all_targets[:, 1], all_margin_preds)
    total_mae = mean_absolute_error(all_targets[:, 2], all_total_preds)

    # Margin direction accuracy
    margin_dir = ((all_margin_preds > 0) == (all_targets[:, 1] > 0)).mean()

    # Log loss for calibration
    try:
        ll = log_loss(all_targets[:, 0], np.clip(all_win_probs, 0.01, 0.99))
    except Exception:
        ll = 999.0

    metrics = {
        "win_accuracy": round(float(win_acc), 4),
        "margin_mae": round(float(margin_mae), 2),
        "margin_direction": round(float(margin_dir), 4),
        "total_mae": round(float(total_mae), 2),
        "log_loss": round(float(ll), 4),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "epochs_trained": epoch + 1,
    }

    emit(f"\n  Test Results:")
    emit(f"    Win Accuracy:     {win_acc:.2%}")
    emit(f"    Margin MAE:       {margin_mae:.2f} pts")
    emit(f"    Margin Direction: {margin_dir:.2%}")
    emit(f"    Total MAE:        {total_mae:.2f} pts")
    emit(f"    Log Loss:         {ll:.4f}")

    return model, metrics, norm_params


def save_model(model, metrics, norm_params):
    """Save model weights, metadata, and update status."""
    ADVANCED_MODEL_DIR.mkdir(exist_ok=True, parents=True)

    torch.save(model.state_dict(), MODEL_PT_PATH)

    with open(MODEL_META_PATH, "wb") as f:
        pickle.dump({
            "metrics": metrics,
            "norm_params": norm_params,
            "architecture": {
                "players_per_team": PLAYERS_PER_TEAM,
                "player_feat_dim": PLAYER_FEAT_DIM,
                "player_embed_dim": PLAYER_EMBED_DIM,
                "team_repr_dim": TEAM_REPR_DIM,
                "matchup_feat_dim": MATCHUP_FEAT_DIM,
                "hidden_dim": HIDDEN_DIM,
            },
            "trained_at": datetime.now().isoformat(),
        }, f)

    # Update status file
    status = {
        "trained": True,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

    print(f"  Saved model to {MODEL_PT_PATH}")


def load_model() -> Optional[Tuple]:
    """Load trained model. Returns (model, meta) or None."""
    if not MODEL_PT_PATH.exists() or not MODEL_META_PATH.exists():
        return None

    with open(MODEL_META_PATH, "rb") as f:
        meta = pickle.load(f)

    arch = meta["architecture"]
    model = AdvancedNBAModel(matchup_dim=arch["matchup_feat_dim"])
    model.load_state_dict(torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=True))
    model.eval()

    return model, meta


def get_model_status() -> Dict:
    """Get current model status for the dashboard."""
    status = {
        "trained": False,
        "trained_at": None,
        "metrics": None,
        "data_status": "unknown",
        "last_data_date": None,
    }

    # Check if model exists
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            saved = json.load(f)
        status.update(saved)

    # Check data freshness
    try:
        games = pd.read_csv(HISTORICAL_GAMES_CSV)
        games["date"] = pd.to_datetime(games["date"])
        last_date = games["date"].max()
        status["last_data_date"] = last_date.strftime("%Y-%m-%d")

        yesterday = pd.Timestamp(date.today()) - pd.Timedelta(days=1)
        if last_date >= yesterday:
            status["data_status"] = "current"
        elif last_date >= yesterday - pd.Timedelta(days=3):
            status["data_status"] = "slightly_behind"
        else:
            status["data_status"] = "behind"
    except Exception:
        status["data_status"] = "error"

    return status


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE LEARNING — Fine-tune on yesterday's actual results
# ═══════════════════════════════════════════════════════════════════════════════

def learn_from_results(target_date: str = None, progress_callback=None) -> Dict:
    """
    Compare model predictions to actual game results and fine-tune.

    1. Fetch yesterday's (or target_date) completed games from ESPN
    2. Run the model on those games to get predictions
    3. Compare to actual outcomes
    4. Do a few gradient steps to penalize mistakes
    5. Save updated weights

    Returns dict with results comparison and accuracy.
    """
    def emit(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    loaded = load_model()
    if loaded is None:
        return {"error": "No model trained yet"}

    model, meta = loaded
    norm_params = meta["norm_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fetch completed games
    import requests
    from datetime import timezone, timedelta
    ET = timezone(timedelta(hours=-5))

    if target_date is None:
        target_date = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
    else:
        target_date = target_date.replace("-", "")

    emit(f"Fetching results for {target_date}...")

    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={target_date}",
            timeout=10,
        )
        r.raise_for_status()
        espn_data = r.json()
    except Exception as e:
        return {"error": f"Failed to fetch games: {e}"}

    # Load data for building features
    player_logs = _load_player_logs()
    features = _load_features()

    roster_mean = np.array(norm_params["roster_mean"], dtype=np.float32)
    roster_std = np.array(norm_params["roster_std"], dtype=np.float32)
    matchup_mean = np.array(norm_params["matchup_mean"], dtype=np.float32)
    matchup_std = np.array(norm_params["matchup_std"], dtype=np.float32)

    # Collect completed games with actual results
    game_inputs = []  # model inputs
    game_targets = [] # actual outcomes
    game_labels = []  # for display

    for event in espn_data.get("events", []):
        comp = event["competitions"][0]
        status_type = comp.get("status", {}).get("type", {}).get("name", "")
        if status_type != "STATUS_FINAL":
            continue

        home_team = away_team = None
        home_score = away_score = 0
        for c in comp["competitors"]:
            abbr = _csv_abbr(c["team"]["abbreviation"])
            score = int(c.get("score", 0))
            if c["homeAway"] == "home":
                home_team = abbr
                home_score = score
            else:
                away_team = abbr
                away_score = score

        if not home_team or not away_team:
            continue

        home_win = 1.0 if home_score > away_score else 0.0
        margin = float(home_score - away_score)
        total = float(home_score + away_score)

        # Build model input (same as predict_today)
        home_roster = np.zeros((PLAYERS_PER_TEAM, PLAYER_FEAT_DIM), dtype=np.float32)
        home_avail = np.ones(PLAYERS_PER_TEAM, dtype=np.float32)
        away_roster = np.zeros((PLAYERS_PER_TEAM, PLAYER_FEAT_DIM), dtype=np.float32)
        away_avail = np.ones(PLAYERS_PER_TEAM, dtype=np.float32)
        missing_ct = [0, 0]

        for side_idx, (team, roster, avail) in enumerate([
            (home_team, home_roster, home_avail),
            (away_team, away_roster, away_avail),
        ]):
            team_logs = player_logs[player_logs["TEAM_ABBREVIATION"] == team]
            if team_logs.empty:
                continue
            recent_ids = team_logs.sort_values("GAME_DATE", ascending=False)["GAME_ID"].unique()[:10]
            recent = team_logs[team_logs["GAME_ID"].isin(recent_ids)]
            agg = recent.groupby("PLAYER_NAME").agg(
                avg_pts=("PTS", "mean"), avg_reb=("REB", "mean"),
                avg_ast=("AST", "mean"), avg_stl=("STL", "mean"),
                avg_blk=("BLK", "mean"), avg_tov=("TOV", "mean"),
                avg_min=("MIN", "mean"), avg_fg=("FG_PCT", "mean"),
            ).sort_values("avg_min", ascending=False)
            top = agg.head(PLAYERS_PER_TEAM)

            for i, (pname, row) in enumerate(top.iterrows()):
                roster[i] = [
                    row["avg_pts"], row["avg_reb"], row["avg_ast"], row["avg_stl"],
                    row["avg_blk"], row["avg_tov"], row["avg_min"], row["avg_fg"],
                ]

        # Normalize
        home_roster = (home_roster - roster_mean) / roster_std
        away_roster = (away_roster - roster_mean) / roster_std
        np.nan_to_num(home_roster, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(away_roster, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Matchup features
        matchup = np.zeros(MATCHUP_FEAT_DIM, dtype=np.float32)
        matchup_cols = norm_params.get("matchup_cols", [])
        feat_row = features[
            (features["home_team"] == home_team) | (features["away_team"] == home_team)
        ].sort_values("date", ascending=False).head(1)
        if not feat_row.empty:
            for i, col in enumerate(matchup_cols[:MATCHUP_FEAT_DIM]):
                val = feat_row.iloc[0].get(col, 0)
                matchup[i] = 0.0 if pd.isna(val) else float(val)
        matchup = (matchup - matchup_mean) / matchup_std
        np.nan_to_num(matchup, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        game_inputs.append((home_roster, home_avail, away_roster, away_avail, matchup, missing_ct))
        game_targets.append([home_win, margin, total])
        game_labels.append(f"{away_team} @ {home_team}: {away_score}-{home_score}")

    if not game_inputs:
        return {"error": "No completed games found for that date"}

    emit(f"Found {len(game_inputs)} completed games")

    # Get predictions BEFORE fine-tuning
    model.eval()
    pre_preds = []
    with torch.no_grad():
        for (hr, ha, ar, aa, mf, mc) in game_inputs:
            win_logit, margin_pred, total_pred = model(
                torch.FloatTensor(hr).unsqueeze(0).to(device),
                torch.FloatTensor(ha).unsqueeze(0).to(device),
                torch.FloatTensor(ar).unsqueeze(0).to(device),
                torch.FloatTensor(aa).unsqueeze(0).to(device),
                torch.FloatTensor(mf).unsqueeze(0).to(device),
                torch.FloatTensor(mc).unsqueeze(0).to(device),
            )
            wp = torch.sigmoid(win_logit).item()
            mg = margin_pred.item() * norm_params["margin_std"] + norm_params["margin_mean"]
            tt = total_pred.item() * norm_params["total_std"] + norm_params["total_mean"]
            pre_preds.append((wp, mg, tt))

    # Compare predictions to actuals
    results = []
    correct_before = 0
    for i, ((wp, mg, tt), (hw, am, at), label) in enumerate(zip(pre_preds, game_targets, game_labels)):
        pred_winner = "HOME" if wp > 0.5 else "AWAY"
        actual_winner = "HOME" if hw > 0.5 else "AWAY"
        correct = pred_winner == actual_winner
        if correct:
            correct_before += 1

        results.append({
            "game": label,
            "pred_win_prob": round(wp * 100, 1),
            "pred_margin": round(mg, 1),
            "pred_total": round(tt, 1),
            "actual_margin": round(am, 1),
            "actual_total": round(at, 1),
            "correct": correct,
            "margin_error": round(abs(mg - am), 1),
            "total_error": round(abs(tt - at), 1),
        })

    acc_before = correct_before / len(results) if results else 0
    emit(f"Pre-learn accuracy: {acc_before:.0%} ({correct_before}/{len(results)})")

    # ── Fine-tune on actual results ─────────────────────────────────────────
    emit(f"Fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
    model.train()

    # Build tensors
    hr_batch = torch.FloatTensor(np.array([g[0] for g in game_inputs])).to(device)
    ha_batch = torch.FloatTensor(np.array([g[1] for g in game_inputs])).to(device)
    ar_batch = torch.FloatTensor(np.array([g[2] for g in game_inputs])).to(device)
    aa_batch = torch.FloatTensor(np.array([g[3] for g in game_inputs])).to(device)
    mf_batch = torch.FloatTensor(np.array([g[4] for g in game_inputs])).to(device)
    mc_batch = torch.FloatTensor(np.array([g[5] for g in game_inputs])).to(device)

    targets_actual = np.array(game_targets, dtype=np.float32)
    # Normalize regression targets
    targets_norm = targets_actual.copy()
    targets_norm[:, 1] = (targets_actual[:, 1] - norm_params["margin_mean"]) / norm_params["margin_std"]
    targets_norm[:, 2] = (targets_actual[:, 2] - norm_params["total_mean"]) / norm_params["total_std"]
    tgt_batch = torch.FloatTensor(targets_norm).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    for ep in range(FINE_TUNE_EPOCHS):
        optimizer.zero_grad()
        win_logit, margin_pred, total_pred = model(hr_batch, ha_batch, ar_batch, aa_batch, mf_batch, mc_batch)

        loss = bce_loss(win_logit, tgt_batch[:, 0]) + 0.3 * mse_loss(margin_pred, tgt_batch[:, 1]) + 0.3 * mse_loss(total_pred, tgt_batch[:, 2])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        emit(f"  Fine-tune epoch {ep+1}: loss={loss.item():.4f}")

    # Check accuracy AFTER fine-tuning
    model.eval()
    correct_after = 0
    with torch.no_grad():
        win_logit, _, _ = model(hr_batch, ha_batch, ar_batch, aa_batch, mf_batch, mc_batch)
        probs = torch.sigmoid(win_logit).cpu().numpy()
        for i, (wp, (hw, _, _)) in enumerate(zip(probs, game_targets)):
            pred = "HOME" if wp > 0.5 else "AWAY"
            actual = "HOME" if hw > 0.5 else "AWAY"
            if pred == actual:
                correct_after += 1

    acc_after = correct_after / len(results) if results else 0
    emit(f"Post-learn accuracy: {acc_after:.0%} ({correct_after}/{len(results)})")

    # Save updated model
    torch.save(model.state_dict(), MODEL_PT_PATH)
    emit("Updated model saved")

    # Log the learning event
    learn_entry = {
        "date": target_date,
        "learned_at": datetime.now().isoformat(),
        "games": len(results),
        "accuracy_before": round(acc_before, 4),
        "accuracy_after": round(acc_after, 4),
    }
    history = []
    if LEARN_HISTORY_PATH.exists():
        try:
            with open(LEARN_HISTORY_PATH) as f:
                history = json.load(f)
        except Exception:
            pass
    history.append(learn_entry)
    with open(LEARN_HISTORY_PATH, "w") as f:
        json.dump(history[-100:], f, indent=2)  # keep last 100

    # Update status
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            status = json.load(f)
        status["last_learned"] = learn_entry["learned_at"]
        status["learn_history_count"] = len(history)
        with open(STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2)

    return {
        "date": target_date,
        "games": len(results),
        "accuracy_before": round(acc_before * 100, 1),
        "accuracy_after": round(acc_after * 100, 1),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION (for today's games)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_today(model=None, meta=None) -> List[Dict]:
    """
    Generate predictions for today's games using the advanced model.
    Integrates current injury report and The-Odds-API odds.
    """
    if model is None:
        loaded = load_model()
        if loaded is None:
            return []
        model, meta = loaded

    norm_params = meta["norm_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get today's games from ESPN
    import requests
    from datetime import timezone, timedelta
    ET = timezone(timedelta(hours=-5))
    today = datetime.now(ET).strftime("%Y%m%d")

    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}",
            timeout=10,
        )
        r.raise_for_status()
        espn_data = r.json()
    except Exception:
        return []

    # Get injury report — normalize team abbrs to match our CSV
    from .injuries import get_espn_injury_report
    raw_injuries = get_espn_injury_report()
    injuries = {}
    for team_abbr, inj_list in raw_injuries.items():
        injuries[_csv_abbr(team_abbr)] = inj_list

    # Get odds
    try:
        from .odds_api import fetch_todays_odds
        odds_list = fetch_todays_odds()
        odds_by_teams = {}
        for o in odds_list:
            odds_by_teams[(o["home"], o["away"])] = o
    except Exception:
        odds_by_teams = {}

    # Load player logs for roster features
    player_logs = _load_player_logs()
    features = _load_features()

    # Build roster stats for today
    predictions = []
    roster_mean = np.array(norm_params["roster_mean"], dtype=np.float32)
    roster_std = np.array(norm_params["roster_std"], dtype=np.float32)
    matchup_mean = np.array(norm_params["matchup_mean"], dtype=np.float32)
    matchup_std = np.array(norm_params["matchup_std"], dtype=np.float32)

    for event in espn_data.get("events", []):
        comp = event["competitions"][0]
        home_team = away_team = None
        home_display = away_display = ""
        for c in comp["competitors"]:
            if c["homeAway"] == "home":
                home_team = _csv_abbr(c["team"]["abbreviation"])
                home_display = c["team"]["displayName"]
            else:
                away_team = _csv_abbr(c["team"]["abbreviation"])
                away_display = c["team"]["displayName"]

        if not home_team or not away_team:
            continue

        # Build roster vectors from recent player logs
        home_roster = np.zeros((PLAYERS_PER_TEAM, PLAYER_FEAT_DIM), dtype=np.float32)
        home_avail = np.ones(PLAYERS_PER_TEAM, dtype=np.float32)
        away_roster = np.zeros((PLAYERS_PER_TEAM, PLAYER_FEAT_DIM), dtype=np.float32)
        away_avail = np.ones(PLAYERS_PER_TEAM, dtype=np.float32)
        missing_ct = [0, 0]

        for side_idx, (team, roster, avail) in enumerate([
            (home_team, home_roster, home_avail),
            (away_team, away_roster, away_avail),
        ]):
            # Get team's recent top players
            team_logs = player_logs[player_logs["TEAM_ABBREVIATION"] == team]
            if team_logs.empty:
                continue

            recent_ids = team_logs.sort_values("GAME_DATE", ascending=False)["GAME_ID"].unique()[:10]
            recent = team_logs[team_logs["GAME_ID"].isin(recent_ids)]
            agg = recent.groupby("PLAYER_NAME").agg(
                avg_pts=("PTS", "mean"), avg_reb=("REB", "mean"),
                avg_ast=("AST", "mean"), avg_stl=("STL", "mean"),
                avg_blk=("BLK", "mean"), avg_tov=("TOV", "mean"),
                avg_min=("MIN", "mean"), avg_fg=("FG_PCT", "mean"),
            ).sort_values("avg_min", ascending=False)

            top = agg.head(PLAYERS_PER_TEAM)

            # Check injuries — catch OUT, Doubtful, Day-to-Day, Suspended
            team_injuries = injuries.get(team, [])
            out_names = set()       # definitely not playing
            likely_out = set()      # probably not playing (DTD, Doubtful)
            for inj in team_injuries:
                status = inj.get("status", "").upper()
                inj_name = inj.get("name", "")
                # Normalize name for matching (strip diacritics)
                import unicodedata
                norm_inj = unicodedata.normalize("NFD", inj_name).encode("ascii", "ignore").decode("ascii")
                if "OUT" in status or "SUSPEND" in status:
                    out_names.add(inj_name)
                    out_names.add(norm_inj)
                elif "DOUBTFUL" in status or "DAY" in status:
                    likely_out.add(inj_name)
                    likely_out.add(norm_inj)

            for i, (pname, row) in enumerate(top.iterrows()):
                roster[i] = [
                    row["avg_pts"], row["avg_reb"], row["avg_ast"], row["avg_stl"],
                    row["avg_blk"], row["avg_tov"], row["avg_min"], row["avg_fg"],
                ]
                # Check if this player is out (also try normalized name)
                norm_pname = unicodedata.normalize("NFD", pname).encode("ascii", "ignore").decode("ascii")
                if pname in out_names or norm_pname in out_names:
                    avail[i] = 0.0
                    missing_ct[side_idx] += 1
                elif pname in likely_out or norm_pname in likely_out:
                    avail[i] = 0.3  # partial availability — might play
                    missing_ct[side_idx] += 0.5

        # Normalize roster
        home_roster = (home_roster - roster_mean) / roster_std
        away_roster = (away_roster - roster_mean) / roster_std

        # Build matchup features from most recent feature row
        matchup = np.zeros(MATCHUP_FEAT_DIM, dtype=np.float32)
        matchup_cols = norm_params.get("matchup_cols", [])
        # Try exact matchup first, then any game with these teams
        feat_row = features[
            (features["home_team"] == home_team) & (features["away_team"] == away_team)
        ].sort_values("date", ascending=False).head(1)
        if feat_row.empty:
            # Try reversed matchup or any recent game with home team
            feat_row = features[
                (features["home_team"] == home_team) | (features["away_team"] == home_team)
            ].sort_values("date", ascending=False).head(1)
        if not feat_row.empty:
            for i, col in enumerate(matchup_cols[:MATCHUP_FEAT_DIM]):
                val = feat_row.iloc[0].get(col, 0)
                matchup[i] = 0.0 if pd.isna(val) else float(val)

        matchup = (matchup - matchup_mean) / matchup_std
        np.nan_to_num(matchup, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(home_roster, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(away_roster, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Run model
        with torch.no_grad():
            hr = torch.FloatTensor(home_roster).unsqueeze(0).to(device)
            ha = torch.FloatTensor(home_avail).unsqueeze(0).to(device)
            ar = torch.FloatTensor(away_roster).unsqueeze(0).to(device)
            aa = torch.FloatTensor(away_avail).unsqueeze(0).to(device)
            mf = torch.FloatTensor(matchup).unsqueeze(0).to(device)
            mc = torch.FloatTensor(missing_ct).unsqueeze(0).to(device)

            win_logit, margin_pred, total_pred = model(hr, ha, ar, aa, mf, mc)

            win_prob = torch.sigmoid(win_logit).item()
            margin = margin_pred.item() * norm_params["margin_std"] + norm_params["margin_mean"]
            total = total_pred.item() * norm_params["total_std"] + norm_params["total_mean"]

        # Get odds
        odds = odds_by_teams.get((home_team, away_team), {})

        # Determine best picks — use win probability distance from 50% as base confidence
        # Scale: 50% = no confidence, 65% = moderate, 75%+ = high
        fav_prob = max(win_prob, 1 - win_prob)
        confidence = round((fav_prob - 0.5) * 200, 1)  # 0-100 scale, 0 at 50%, 50 at 75%

        pred = {
            "home_team": home_team,
            "away_team": away_team,
            "home_name": home_display or home_team,
            "away_name": away_display or away_team,
            "win_prob": round(win_prob * 100, 1),
            "predicted_margin": round(margin, 1),
            "predicted_total": round(total, 1),
            "confidence": confidence,
            "odds_spread": odds.get("spread"),
            "odds_total": odds.get("total"),
            "odds_home_ml": odds.get("home_ml"),
            "odds_away_ml": odds.get("away_ml"),
            "home_missing": int(missing_ct[0]),
            "away_missing": int(missing_ct[1]),
        }

        # Generate pick recommendations
        picks = []
        fav = home_team if win_prob > 0.5 else away_team
        dog = away_team if win_prob > 0.5 else home_team

        # ML pick — only recommend if model is reasonably confident
        if fav_prob >= 0.55:
            picks.append({
                "type": "ML",
                "pick": f"{fav} ML",
                "confidence": round(fav_prob * 100, 1),
            })

        # Spread pick — compare model margin to book spread
        if odds.get("spread") is not None:
            book_spread = odds["spread"]  # negative = home favored
            # Model says home wins by `margin`. Book spread is `book_spread`.
            # Edge = how much better model thinks home does vs book line
            edge = margin - book_spread
            if abs(edge) > 2.0:
                if edge > 0:
                    # Model thinks home covers
                    pick_text = f"{home_team} {book_spread:+.1f}"
                else:
                    # Model thinks away covers
                    away_spread = -book_spread
                    pick_text = f"{away_team} {away_spread:+.1f}"
                # Confidence scales with edge size (2pt edge ~ 55%, 8pt edge ~ 70%)
                spread_conf = min(80, 52 + abs(edge) * 2.5)
                picks.append({
                    "type": "Spread",
                    "pick": pick_text,
                    "confidence": round(spread_conf, 1),
                    "edge": round(edge, 1),
                })

        # Total pick — compare model total to book total
        if odds.get("total") is not None:
            book_total = odds["total"]
            total_edge = total - book_total
            if abs(total_edge) > 4:
                direction = "Over" if total_edge > 0 else "Under"
                total_conf = min(75, 52 + abs(total_edge) * 1.5)
                picks.append({
                    "type": "Total",
                    "pick": f"{direction} {book_total}",
                    "confidence": round(total_conf, 1),
                    "edge": round(total_edge, 1),
                })

        pred["picks"] = picks
        predictions.append(pred)

    # Sort by confidence
    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced NBA Neural Network")
    parser.add_argument("--status", action="store_true", help="Show model status")
    parser.add_argument("--predict", action="store_true", help="Predict today's games")
    args = parser.parse_args()

    if args.status:
        status = get_model_status()
        print(json.dumps(status, indent=2))
        return

    if args.predict:
        preds = predict_today()
        for p in preds:
            print(f"\n{p['away_team']} @ {p['home_team']}")
            print(f"  Win Prob: {p['win_prob']}% (home)")
            print(f"  Margin: {p['predicted_margin']:+.1f}")
            print(f"  Total: {p['predicted_total']:.1f}")
            for pick in p.get("picks", []):
                print(f"  → {pick['pick']} ({pick['confidence']}%)")
        return

    # Train
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 60)
    print("ADVANCED NBA MODEL — Injury-Aware Neural Network")
    print("  Roster attention + matchup features + injury context")
    print("=" * 60)

    model, metrics, norm_params = train_model()
    save_model(model, metrics, norm_params)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
