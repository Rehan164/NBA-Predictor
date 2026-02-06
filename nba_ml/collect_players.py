"""
NBA Player Game Log Collection

Collects player-level game logs for all seasons to derive
star power, depth, and dependency features.

Uses LeagueGameLog in player mode - only 1 API call per season.

Usage:
    python -m nba_ml.collect_players
"""

import time
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import LeagueGameLog

from .config import (
    START_SEASON, END_SEASON,
    NBA_API_DELAY,
    PLAYER_GAME_LOGS_CSV,
)


def fetch_player_season(season_year: int) -> pd.DataFrame:
    """Fetch all player game logs for a season."""
    season_str = f"{season_year - 1}-{str(season_year)[2:]}"

    try:
        game_log = LeagueGameLog(
            season=season_str,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",  # Player-level logs
        )
        time.sleep(NBA_API_DELAY)

        df = game_log.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"  Error fetching player data for {season_str}: {e}")
        return pd.DataFrame()


def main():
    print("=" * 60)
    print("NBA Player Game Log Collection")
    print("=" * 60)
    print(f"\nCollecting seasons {START_SEASON} to {END_SEASON}")
    print(f"Output: {PLAYER_GAME_LOGS_CSV}\n")

    all_logs = []

    for year in tqdm(range(START_SEASON, END_SEASON + 1), desc="Collecting player logs"):
        print(f"\n  Fetching {year - 1}-{str(year)[2:]} season...")

        df = fetch_player_season(year)
        if df.empty:
            print(f"    No data")
            continue

        # Keep only essential columns to save space
        keep_cols = [
            "SEASON_ID", "PLAYER_ID", "PLAYER_NAME",
            "TEAM_ID", "TEAM_ABBREVIATION",
            "GAME_ID", "GAME_DATE",
            "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
            "FGM", "FGA", "FG_PCT",
            "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT",
            "PLUS_MINUS",
        ]

        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols].copy()

        print(f"    Found {len(df):,} player-game rows")
        all_logs.append(df)

        time.sleep(1)

    if not all_logs:
        print("No data collected!")
        return

    combined = pd.concat(all_logs, ignore_index=True)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
    combined = combined.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print("Collection Summary")
    print(f"{'=' * 60}")
    print(f"Total player-game rows: {len(combined):,}")
    print(f"Unique players: {combined['PLAYER_ID'].nunique():,}")
    print(f"Date range: {combined['GAME_DATE'].min().date()} to {combined['GAME_DATE'].max().date()}")

    combined.to_csv(PLAYER_GAME_LOGS_CSV, index=False)
    print(f"\nSaved to: {PLAYER_GAME_LOGS_CSV}")
    print(f"File size: {PLAYER_GAME_LOGS_CSV.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
