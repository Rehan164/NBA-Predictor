"""
LSTM Neural Network for NBA Game Prediction

Captures sequential patterns in team performance that tree-based
models miss. Each team's last N games form a sequence, and the
LSTM learns temporal patterns like momentum, slumps, and fatigue.

Architecture:
  Home Team (seq of 10 games) -> LSTM(64) -> h_home
  Away Team (seq of 10 games) -> LSTM(64) -> h_away  (shared weights)
  [h_home, h_away, static_features] -> Dense(128) -> Dense(64) -> Outputs

Three output heads:
  1. Win probability (binary cross-entropy)
  2. Home margin (MSE)
  3. Total score (MSE)

Usage:
    python -m nba_ml.neural_model
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error

from .config import (
    TRAINING_FEATURES_CSV,
    HISTORICAL_GAMES_CSV,
    MODELS_DIR,
    TRAIN_END_DATE,
    TEST_START_DATE,
    RANDOM_STATE,
)


# ===================================================================
# CONFIGURATION
# ===================================================================

SEQ_LEN = 10          # Number of past games per team
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
DROPOUT = 0.3
PATIENCE = 12         # Early stopping patience

# Features per game in the sequence
SEQ_FEATURES = [
    "pts", "opp_pts", "margin", "fg_pct", "fg3_pct", "ft_pct",
    "oreb", "ast", "tov", "won", "is_home", "rest_days",
    "efg_pct", "tov_pct", "oreb_pct", "pace", "off_rating", "def_rating",
]

# Static features (per-game context)
STATIC_FEATURES = [
    "elo_diff", "elo_expected", "travel_distance", "timezone_diff",
    "rest_advantage", "season_win_pct_diff",
]


# ===================================================================
# DATA PREPARATION
# ===================================================================

def build_game_sequences(games_csv: Path, features_csv: Path):
    """
    Build sequential training data for the LSTM.

    For each game, extract the last SEQ_LEN games for both home and
    away teams as sequences, plus static matchup features.

    Returns:
        home_seqs: (N, SEQ_LEN, n_seq_features)
        away_seqs: (N, SEQ_LEN, n_seq_features)
        static: (N, n_static_features)
        targets: (N, 3) - [win, margin, total]
        dates: (N,) - for train/test split
    """
    print("Loading data for LSTM...")

    # Load raw games for box score data
    raw = pd.read_csv(games_csv)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    # Load features for ELO and static features
    feat_df = pd.read_csv(features_csv)
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    print(f"  {len(raw):,} games loaded")

    # Build per-team game history with sequence features
    team_histories = {}
    all_teams = set(raw["home_team"].unique()) | set(raw["away_team"].unique())

    for team in all_teams:
        # Home games
        home = raw[raw["home_team"] == team].copy()
        home["_team"] = team
        home["pts"] = home["home_score"]
        home["opp_pts"] = home["away_score"]
        home["margin"] = home["home_margin"]
        home["fg_pct"] = home["home_fg_pct"]
        home["fg3_pct"] = home["home_fg3_pct"]
        home["ft_pct"] = home["home_ft_pct"]
        home["oreb"] = home["home_oreb"]
        home["ast"] = home["home_ast"]
        home["tov"] = home["home_tov"]
        home["won"] = home["home_win"]
        home["is_home"] = 1

        # Away games
        away = raw[raw["away_team"] == team].copy()
        away["_team"] = team
        away["pts"] = away["away_score"]
        away["opp_pts"] = away["home_score"]
        away["margin"] = -away["home_margin"]
        away["fg_pct"] = away["away_fg_pct"]
        away["fg3_pct"] = away["away_fg3_pct"]
        away["ft_pct"] = away["away_ft_pct"]
        away["oreb"] = away["away_oreb"]
        away["ast"] = away["away_ast"]
        away["tov"] = away["away_tov"]
        away["won"] = 1 - away["home_win"]
        away["is_home"] = 0

        combined = pd.concat([home, away]).sort_values("date").reset_index(drop=True)

        # Compute rest days
        combined["rest_days"] = combined["date"].diff().dt.days.fillna(3).clip(0, 7)

        # Compute Four Factors from raw data using vectorized operations
        is_home = combined["is_home"] == 1
        fga = np.where(is_home, combined.get("home_fg_att", 0), combined.get("away_fg_att", 0))
        fg3m = np.where(is_home, combined.get("home_fg3_made", 0), combined.get("away_fg3_made", 0))
        fgm = np.where(is_home, combined.get("home_fg_made", 0), combined.get("away_fg_made", 0))
        fta = np.where(is_home, combined.get("home_ft_att", 0), combined.get("away_ft_att", 0))

        # Replace NaN with 0 for missing box score data
        fga = np.nan_to_num(fga, nan=0.0).astype(float)
        fg3m = np.nan_to_num(fg3m, nan=0.0).astype(float)
        fgm = np.nan_to_num(fgm, nan=0.0).astype(float)
        fta = np.nan_to_num(fta, nan=0.0).astype(float)

        combined["efg_pct"] = np.where(fga > 0, (fgm + 0.5 * fg3m) / fga, 0.0)
        possessions = fga + 0.44 * fta - combined["oreb"].fillna(0) + combined["tov"].fillna(0)
        combined["tov_pct"] = np.where(possessions > 0, combined["tov"].fillna(0) / possessions, 0.0)
        combined["oreb_pct"] = 0.25  # Approximate (need opponent dreb which is complex here)
        combined["pace"] = possessions
        combined["off_rating"] = np.where(possessions > 0, combined["pts"].fillna(0) / possessions * 100, 100)
        combined["def_rating"] = np.where(possessions > 0, combined["opp_pts"].fillna(0) / possessions * 100, 100)

        team_histories[team] = combined

    # Build sequences for each game in features_df
    home_sequences = []
    away_sequences = []
    static_features = []
    targets = []
    dates = []

    n_seq_feats = len(SEQ_FEATURES)

    for _, row in feat_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        game_date = row["date"]

        if home_team not in team_histories or away_team not in team_histories:
            continue

        # Get last SEQ_LEN games for home team before this game
        h_hist = team_histories[home_team]
        h_prior = h_hist[h_hist["date"] < game_date].tail(SEQ_LEN)

        a_hist = team_histories[away_team]
        a_prior = a_hist[a_hist["date"] < game_date].tail(SEQ_LEN)

        if len(h_prior) < SEQ_LEN or len(a_prior) < SEQ_LEN:
            continue

        # Extract sequence features
        h_seq = np.zeros((SEQ_LEN, n_seq_feats))
        a_seq = np.zeros((SEQ_LEN, n_seq_feats))

        for i, feat_name in enumerate(SEQ_FEATURES):
            if feat_name in h_prior.columns:
                h_seq[:, i] = h_prior[feat_name].values
            if feat_name in a_prior.columns:
                a_seq[:, i] = a_prior[feat_name].values

        # Static features
        static = []
        for sf in STATIC_FEATURES:
            val = row.get(sf, 0)
            static.append(0.0 if pd.isna(val) else float(val))

        home_sequences.append(h_seq)
        away_sequences.append(a_seq)
        static_features.append(static)
        targets.append([
            float(row["home_win"]),
            float(row["home_margin"]),
            float(row["total_score"]),
        ])
        dates.append(game_date)

    home_sequences = np.array(home_sequences, dtype=np.float32)
    away_sequences = np.array(away_sequences, dtype=np.float32)
    static_features = np.array(static_features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    dates = np.array(dates)

    print(f"  Built {len(targets):,} game sequences")
    print(f"  Sequence shape: ({SEQ_LEN}, {n_seq_feats})")
    print(f"  Static features: {len(STATIC_FEATURES)}")

    # Replace any NaN/Inf BEFORE normalization
    home_sequences = np.nan_to_num(home_sequences, nan=0.0, posinf=0.0, neginf=0.0)
    away_sequences = np.nan_to_num(away_sequences, nan=0.0, posinf=0.0, neginf=0.0)
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize sequences (z-score per feature)
    for i in range(n_seq_feats):
        h_vals = home_sequences[:, :, i].flatten()
        a_vals = away_sequences[:, :, i].flatten()
        all_vals = np.concatenate([h_vals, a_vals])
        mean_v = np.nanmean(all_vals)
        std_v = np.nanstd(all_vals) + 1e-8
        home_sequences[:, :, i] = (home_sequences[:, :, i] - mean_v) / std_v
        away_sequences[:, :, i] = (away_sequences[:, :, i] - mean_v) / std_v

    # Normalize static features
    for i in range(static_features.shape[1]):
        mean_v = np.nanmean(static_features[:, i])
        std_v = np.nanstd(static_features[:, i]) + 1e-8
        static_features[:, i] = (static_features[:, i] - mean_v) / std_v

    # Final NaN cleanup after normalization
    home_sequences = np.nan_to_num(home_sequences, nan=0.0, posinf=0.0, neginf=0.0)
    away_sequences = np.nan_to_num(away_sequences, nan=0.0, posinf=0.0, neginf=0.0)
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Clean targets
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    # Report data health
    n_total = home_sequences.size + away_sequences.size + static_features.size + targets.size
    print(f"  Data health: {n_total:,} total values, all NaN/Inf cleaned")

    # Normalize regression targets for better training
    margin_mean = targets[:, 1].mean()
    margin_std = targets[:, 1].std() + 1e-8
    total_mean = targets[:, 2].mean()
    total_std = targets[:, 2].std() + 1e-8

    norm_params = {
        "margin_mean": float(margin_mean), "margin_std": float(margin_std),
        "total_mean": float(total_mean), "total_std": float(total_std),
        "seq_features": SEQ_FEATURES,
        "static_features": STATIC_FEATURES,
    }

    return home_sequences, away_sequences, static_features, targets, dates, norm_params


# ===================================================================
# PYTORCH DATASET & MODEL
# ===================================================================

class NBAGameDataset(Dataset):
    def __init__(self, home_seqs, away_seqs, static, targets):
        self.home_seqs = torch.FloatTensor(home_seqs)
        self.away_seqs = torch.FloatTensor(away_seqs)
        self.static = torch.FloatTensor(static)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.home_seqs[idx], self.away_seqs[idx],
                self.static[idx], self.targets[idx])


class NBALSTMModel(nn.Module):
    """
    Dual-LSTM model with shared weights for home/away team sequences.

    Architecture:
      Each team's sequence -> LSTM -> last hidden state
      Concatenate [h_home, h_away, static] -> MLP -> outputs
    """

    def __init__(self, seq_features, static_features, hidden_size=64, dropout=0.3):
        super().__init__()

        # Shared LSTM for both teams
        self.lstm = nn.LSTM(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # MLP head
        concat_size = hidden_size * 2 + static_features  # home + away + static
        self.mlp = nn.Sequential(
            nn.Linear(concat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.win_head = nn.Linear(64, 1)      # Win probability
        self.margin_head = nn.Linear(64, 1)   # Home margin
        self.total_head = nn.Linear(64, 1)    # Total score

    def forward(self, home_seq, away_seq, static):
        # Process both teams through shared LSTM
        _, (h_home, _) = self.lstm(home_seq)
        _, (h_away, _) = self.lstm(away_seq)

        # Take last layer's hidden state
        h_home = h_home[-1]  # (batch, hidden)
        h_away = h_away[-1]  # (batch, hidden)

        # Concatenate and pass through MLP
        combined = torch.cat([h_home, h_away, static], dim=1)
        features = self.mlp(combined)

        # Output predictions
        win_logit = self.win_head(features).squeeze(-1)
        margin = self.margin_head(features).squeeze(-1)
        total = self.total_head(features).squeeze(-1)

        return win_logit, margin, total


# ===================================================================
# TRAINING
# ===================================================================

def train_lstm(home_seqs, away_seqs, static, targets, dates, norm_params):
    """Train the LSTM model with early stopping."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Split chronologically
    train_end = pd.to_datetime(TRAIN_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)

    train_mask = dates < train_end
    test_mask = dates >= test_start

    # Further split training into train/val (last 15% of train for validation)
    train_indices = np.where(train_mask)[0]
    split_idx = int(len(train_indices) * 0.85)
    fit_indices = train_indices[:split_idx]
    val_indices = train_indices[split_idx:]
    test_indices = np.where(test_mask)[0]

    print(f"  Train: {len(fit_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")

    # Normalize regression targets
    targets_norm = targets.copy()
    targets_norm[:, 1] = (targets[:, 1] - norm_params["margin_mean"]) / norm_params["margin_std"]
    targets_norm[:, 2] = (targets[:, 2] - norm_params["total_mean"]) / norm_params["total_std"]

    # Create datasets
    train_ds = NBAGameDataset(
        home_seqs[fit_indices], away_seqs[fit_indices],
        static[fit_indices], targets_norm[fit_indices]
    )
    val_ds = NBAGameDataset(
        home_seqs[val_indices], away_seqs[val_indices],
        static[val_indices], targets_norm[val_indices]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    n_seq_feats = home_seqs.shape[2]
    n_static_feats = static.shape[1]
    model = NBALSTMModel(n_seq_feats, n_static_feats, HIDDEN_SIZE, DROPOUT).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n  Training for up to {EPOCHS} epochs (patience: {PATIENCE})...")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_losses = []

        for h_seq, a_seq, stat, tgt in train_loader:
            h_seq, a_seq, stat, tgt = h_seq.to(device), a_seq.to(device), stat.to(device), tgt.to(device)

            optimizer.zero_grad()
            win_logit, margin_pred, total_pred = model(h_seq, a_seq, stat)

            loss_win = bce_loss(win_logit, tgt[:, 0])
            loss_margin = mse_loss(margin_pred, tgt[:, 1])
            loss_total = mse_loss(total_pred, tgt[:, 2])

            # Multi-task loss (weighted sum)
            loss = loss_win + 0.5 * loss_margin + 0.5 * loss_total
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_preds_win = []
        val_true_win = []

        with torch.no_grad():
            for h_seq, a_seq, stat, tgt in val_loader:
                h_seq, a_seq, stat, tgt = h_seq.to(device), a_seq.to(device), stat.to(device), tgt.to(device)
                win_logit, margin_pred, total_pred = model(h_seq, a_seq, stat)

                loss_win = bce_loss(win_logit, tgt[:, 0])
                loss_margin = mse_loss(margin_pred, tgt[:, 1])
                loss_total = mse_loss(total_pred, tgt[:, 2])
                loss = loss_win + 0.5 * loss_margin + 0.5 * loss_total
                val_losses.append(loss.item())

                val_preds_win.extend((torch.sigmoid(win_logit) > 0.5).cpu().numpy())
                val_true_win.extend(tgt[:, 0].cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_true_win, val_preds_win)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Early stopping (treat NaN as non-improvement)
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best model (fallback to final state if no improvement was recorded)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("    WARNING: No best model state saved, using final epoch weights")
    model.eval()

    # Evaluate on test set
    print(f"\n  Evaluating on test set...")
    test_ds = NBAGameDataset(
        home_seqs[test_indices], away_seqs[test_indices],
        static[test_indices], targets[test_indices]  # Use original targets for eval
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_win_preds = []
    all_margin_preds = []
    all_total_preds = []
    all_targets = []

    with torch.no_grad():
        for h_seq, a_seq, stat, tgt in test_loader:
            h_seq, a_seq, stat = h_seq.to(device), a_seq.to(device), stat.to(device)
            win_logit, margin_pred, total_pred = model(h_seq, a_seq, stat)

            # Denormalize predictions
            margin_denorm = margin_pred.cpu().numpy() * norm_params["margin_std"] + norm_params["margin_mean"]
            total_denorm = total_pred.cpu().numpy() * norm_params["total_std"] + norm_params["total_mean"]

            all_win_preds.extend(torch.sigmoid(win_logit).cpu().numpy())
            all_margin_preds.extend(margin_denorm)
            all_total_preds.extend(total_denorm)
            all_targets.extend(tgt.numpy())

    all_win_preds = np.array(all_win_preds)
    all_margin_preds = np.array(all_margin_preds)
    all_total_preds = np.array(all_total_preds)
    all_targets = np.array(all_targets)

    # Metrics
    win_acc = accuracy_score(all_targets[:, 0], (all_win_preds > 0.5).astype(int))
    margin_mae = mean_absolute_error(all_targets[:, 1], all_margin_preds)
    total_mae = mean_absolute_error(all_targets[:, 2], all_total_preds)
    margin_dir = ((all_margin_preds > 0) == (all_targets[:, 1] > 0)).mean()

    print(f"\n  LSTM Test Results:")
    print(f"    Win Accuracy:     {win_acc:.4f}")
    print(f"    Margin MAE:       {margin_mae:.2f} pts")
    print(f"    Margin Direction: {margin_dir:.2%}")
    print(f"    Total MAE:        {total_mae:.2f} pts")

    metrics = {
        "win_accuracy": float(win_acc),
        "margin_mae": float(margin_mae),
        "margin_direction": float(margin_dir),
        "total_mae": float(total_mae),
    }

    return model, metrics, norm_params


# ===================================================================
# SAVE / LOAD
# ===================================================================

def save_lstm_model(model, metrics, norm_params):
    """Save LSTM model as both PyTorch state dict and pickle metadata."""
    model_dir = MODELS_DIR
    model_dir.mkdir(exist_ok=True)

    # Save PyTorch model
    torch_path = model_dir / "lstm_model.pt"
    torch.save(model.state_dict(), torch_path)

    # Save metadata
    meta_path = model_dir / "lstm_model.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "model_type": "lstm",
            "metrics": metrics,
            "norm_params": norm_params,
            "architecture": {
                "seq_features": len(SEQ_FEATURES),
                "static_features": len(STATIC_FEATURES),
                "hidden_size": HIDDEN_SIZE,
                "dropout": DROPOUT,
                "seq_len": SEQ_LEN,
            },
            "trained_at": datetime.now().isoformat(),
        }, f)

    print(f"\n  Saved LSTM model to: {torch_path}")
    print(f"  Saved metadata to: {meta_path}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("NBA LSTM Neural Network Training")
    print("  Dual-LSTM with shared weights for team sequences")
    print("  Multi-task: Win prob + Margin + Total")
    print("=" * 60)

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    home_seqs, away_seqs, static, targets, dates, norm_params = build_game_sequences(
        HISTORICAL_GAMES_CSV, TRAINING_FEATURES_CSV
    )

    model, metrics, norm_params = train_lstm(
        home_seqs, away_seqs, static, targets, dates, norm_params
    )

    save_lstm_model(model, metrics, norm_params)

    print(f"\n{'='*60}")
    print("LSTM TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Win Accuracy:     {metrics['win_accuracy']:.2%}")
    print(f"  Margin MAE:       {metrics['margin_mae']:.2f} pts")
    print(f"  Margin Direction: {metrics['margin_direction']:.2%}")
    print(f"  Total MAE:        {metrics['total_mae']:.2f} pts")
    print(f"\nNext: python -m nba_ml.train_models (LSTM output feeds into ensemble)")


if __name__ == "__main__":
    main()
