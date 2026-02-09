"""
Player Prop Feature Engineering

Builds per-player training data for predicting PTS, REB, AST.

Features per player-game:
  - Rolling averages (last 3, 5, 10, 20 games)
  - Scoring variance and momentum
  - Home/away, starter status, rest days
  - Opponent defensive profile (pts/reb/ast allowed)
  - Usage rate proxy
  - Minutes trend

Usage:
    python -m nba_ml.build_player_features
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import (
    PLAYER_GAME_LOGS_CSV,
    HISTORICAL_GAMES_CSV,
    PLAYER_FEATURES_CSV,
)

ROLLING_WINDOWS = [3, 5, 10, 20]
MIN_PLAYER_GAMES = 10
STAT_COLS = ["PTS", "REB", "AST", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "TOV", "STL", "BLK", "PLUS_MINUS"]


def load_data():
    """Load player game logs and historical game records."""
    print("Loading player game logs...")
    if not PLAYER_GAME_LOGS_CSV.exists():
        raise FileNotFoundError(
            f"Player game logs not found: {PLAYER_GAME_LOGS_CSV}\n"
            "Run 'python -m nba_ml.collect_players' first."
        )

    plr = pd.read_csv(PLAYER_GAME_LOGS_CSV)
    plr["GAME_DATE"] = pd.to_datetime(plr["GAME_DATE"])

    for col in STAT_COLS:
        if col in plr.columns:
            plr[col] = pd.to_numeric(plr[col], errors="coerce").fillna(0)

    print(f"  {len(plr):,} player-game rows, {plr['PLAYER_ID'].nunique():,} players")

    print("Loading game records...")
    if not HISTORICAL_GAMES_CSV.exists():
        raise FileNotFoundError(
            f"Historical games not found: {HISTORICAL_GAMES_CSV}\n"
            "Run 'python -m nba_ml.collect_data' first."
        )

    games = pd.read_csv(HISTORICAL_GAMES_CSV)
    games["date"] = pd.to_datetime(games["date"])
    print(f"  {len(games):,} games")

    return plr, games


def build_game_lookup(games: pd.DataFrame) -> dict:
    """Build game_id → game info lookup."""
    lookup = {}
    for _, row in games.iterrows():
        lookup[row["game_id"]] = {
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
        }
    return lookup


def build_team_defense_profiles(games: pd.DataFrame) -> dict:
    """
    For each team, build rolling averages of what opponents score against them.

    Returns dict: (team_abbr, game_id) → {def_pts_allowed_l10, def_reb_allowed_l10, ...}
    """
    print("Building team defensive profiles...")

    # Build a record for each team-game: what the opponent scored
    records = []
    for _, row in games.iterrows():
        records.append({
            "team": row["home_team"],
            "game_id": row["game_id"],
            "date": row["date"],
            "pts_allowed": row["away_score"],
            "reb_allowed": row.get("away_reb", np.nan),
            "ast_allowed": row.get("away_ast", np.nan),
        })
        records.append({
            "team": row["away_team"],
            "game_id": row["game_id"],
            "date": row["date"],
            "pts_allowed": row["home_score"],
            "reb_allowed": row.get("home_reb", np.nan),
            "ast_allowed": row.get("home_ast", np.nan),
        })

    df = pd.DataFrame(records).sort_values(["team", "date"])

    # Rolling averages per team (shifted so we only use PRIOR games)
    profiles = {}

    for team, group in tqdm(df.groupby("team"), desc="Team defense profiles"):
        group = group.sort_values("date").reset_index(drop=True)

        for stat in ["pts_allowed", "reb_allowed", "ast_allowed"]:
            for w in [10, 20]:
                group[f"{stat}_l{w}"] = group[stat].rolling(w, min_periods=5).mean().shift(1)

        for _, row in group.iterrows():
            profiles[(team, row["game_id"])] = {
                "opp_def_pts_allowed_l10": row.get("pts_allowed_l10"),
                "opp_def_pts_allowed_l20": row.get("pts_allowed_l20"),
                "opp_def_reb_allowed_l10": row.get("reb_allowed_l10"),
                "opp_def_reb_allowed_l20": row.get("reb_allowed_l20"),
                "opp_def_ast_allowed_l10": row.get("ast_allowed_l10"),
                "opp_def_ast_allowed_l20": row.get("ast_allowed_l20"),
            }

    print(f"  {len(profiles):,} team-game defensive profiles built")
    return profiles


def build_features(plr: pd.DataFrame, game_lookup: dict,
                   defense_profiles: dict) -> pd.DataFrame:
    """Build the full player-level feature matrix."""

    print(f"\nBuilding player features for {plr['PLAYER_ID'].nunique():,} players...")

    features_list = []
    skipped = 0

    for player_id, p_games in tqdm(plr.groupby("PLAYER_ID"), desc="Player features"):
        p_games = p_games.sort_values("GAME_DATE").reset_index(drop=True)

        if len(p_games) < MIN_PLAYER_GAMES:
            skipped += 1
            continue

        player_name = p_games["PLAYER_NAME"].iloc[0]

        # Pre-compute rolling stats for all windows (shifted by 1)
        rolling = {}
        for stat in STAT_COLS:
            if stat not in p_games.columns:
                continue
            for w in ROLLING_WINDOWS:
                rolling[f"{stat}_avg_l{w}"] = p_games[stat].rolling(w, min_periods=1).mean().shift(1)
                if stat in ["PTS", "REB", "AST"]:
                    rolling[f"{stat}_std_l{w}"] = p_games[stat].rolling(w, min_periods=2).std().shift(1)
                    rolling[f"{stat}_max_l{w}"] = p_games[stat].rolling(w, min_periods=1).max().shift(1)
                    rolling[f"{stat}_min_l{w}"] = p_games[stat].rolling(w, min_periods=1).min().shift(1)

        rolling_df = pd.DataFrame(rolling)

        # Expanding season averages (shifted)
        season_col = p_games["SEASON_ID"]
        for stat in ["PTS", "REB", "AST", "MIN"]:
            if stat in p_games.columns:
                rolling_df[f"{stat}_season_avg"] = (
                    p_games.groupby(season_col)[stat]
                    .expanding().mean()
                    .shift(1)
                    .reset_index(level=0, drop=True)
                )

        # Momentum: last 3 vs last 10
        for stat in ["PTS", "REB", "AST"]:
            avg_3 = p_games[stat].rolling(3, min_periods=1).mean().shift(1)
            avg_10 = p_games[stat].rolling(10, min_periods=3).mean().shift(1)
            rolling_df[f"{stat}_momentum"] = avg_3 - avg_10

        # Usage proxy: FGA per 48 minutes
        if "FGA" in p_games.columns and "MIN" in p_games.columns:
            min_safe = p_games["MIN"].replace(0, 1)
            rolling_df["usage_proxy"] = (
                (p_games["FGA"] / min_safe * 48)
                .rolling(10, min_periods=3).mean().shift(1)
            )

        # Rest days
        rest = p_games["GAME_DATE"].diff().dt.days.clip(upper=7)

        # Process each game starting from MIN_PLAYER_GAMES
        for idx in range(MIN_PLAYER_GAMES, len(p_games)):
            row = p_games.iloc[idx]
            game_id = row["GAME_ID"]
            team = row["TEAM_ABBREVIATION"]

            ginfo = game_lookup.get(game_id)
            if ginfo is None:
                continue

            # Determine home/away and opponent
            if ginfo["home_team"] == team:
                opponent = ginfo["away_team"]
                is_home = 1
            elif ginfo["away_team"] == team:
                opponent = ginfo["home_team"]
                is_home = 0
            else:
                continue

            # Start building feature dict
            feat = {
                "player_id": player_id,
                "player_name": player_name,
                "team": team,
                "opponent": opponent,
                "game_id": game_id,
                "game_date": row["GAME_DATE"],
                "is_home": is_home,
                "rest_days": rest.iloc[idx] if pd.notna(rest.iloc[idx]) else 2,

                # Targets
                "target_pts": row["PTS"],
                "target_reb": row["REB"],
                "target_ast": row["AST"],
                "actual_min": row["MIN"],
            }

            # Starter proxy: avg minutes >= 24 in last 5
            avg_min_5 = rolling_df.at[idx, "MIN_avg_l5"] if "MIN_avg_l5" in rolling_df.columns else 0
            feat["is_starter"] = 1 if (pd.notna(avg_min_5) and avg_min_5 >= 24) else 0

            # Add all rolling features
            for col in rolling_df.columns:
                val = rolling_df.at[idx, col]
                if pd.notna(val):
                    feat[col] = val

            # Opponent defensive profile (BEFORE this game)
            opp_def = defense_profiles.get((opponent, game_id), {})
            for k, v in opp_def.items():
                if pd.notna(v):
                    feat[k] = v

            features_list.append(feat)

    print(f"  Skipped {skipped} players with < {MIN_PLAYER_GAMES} games")
    df = pd.DataFrame(features_list)

    # Drop rows with missing essential rolling stats
    essential = ["PTS_avg_l5", "REB_avg_l5", "AST_avg_l5", "MIN_avg_l5"]
    existing = [c for c in essential if c in df.columns]
    if existing:
        before = len(df)
        df = df.dropna(subset=existing)
        print(f"  Dropped {before - len(df)} rows with missing rolling stats")

    return df


def main():
    print("=" * 60)
    print("Player Prop Feature Engineering")
    print("=" * 60)

    plr, games = load_data()

    game_lookup = build_game_lookup(games)
    defense_profiles = build_team_defense_profiles(games)

    df = build_features(plr, game_lookup, defense_profiles)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Player-game rows: {len(df):,}")
    print(f"  Unique players: {df['player_id'].nunique():,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    # Feature categories
    feature_cols = [c for c in df.columns if c not in [
        "player_id", "player_name", "team", "opponent", "game_id", "game_date",
        "target_pts", "target_reb", "target_ast", "actual_min",
    ]]
    print(f"  Feature count: {len(feature_cols)}")

    df.to_csv(PLAYER_FEATURES_CSV, index=False)
    print(f"\n  Saved to: {PLAYER_FEATURES_CSV}")
    print(f"  File size: {PLAYER_FEATURES_CSV.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
