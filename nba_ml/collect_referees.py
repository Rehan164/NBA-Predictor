"""
NBA Referee Data Collection

Collects referee assignments for each game using BoxScoreSummaryV2.

WARNING: This requires ~31,000 API calls (one per game).
At 0.6s delay, this takes approximately 5-6 hours to complete.
Run this overnight or in the background.

It supports resuming from where it left off if interrupted.

Usage:
    python -m nba_ml.collect_referees
"""

import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from nba_api.stats.endpoints import BoxScoreSummaryV2

from .config import (
    NBA_API_DELAY,
    HISTORICAL_GAMES_CSV,
    REFEREE_DATA_CSV,
)


def fetch_game_officials(game_id: str) -> dict:
    """Fetch referee assignments for a single game."""
    try:
        summary = BoxScoreSummaryV2(game_id=game_id)
        time.sleep(NBA_API_DELAY)

        officials_df = summary.get_data_frames()[2]  # Officials table

        refs = {}
        for i, row in officials_df.iterrows():
            ref_num = i + 1
            refs[f"ref_{ref_num}_id"] = row.get("OFFICIAL_ID", "")
            refs[f"ref_{ref_num}_name"] = (
                f"{row.get('FIRST_NAME', '')} {row.get('LAST_NAME', '')}".strip()
            )

        return refs
    except Exception:
        return {}


def main():
    print("=" * 60)
    print("NBA Referee Data Collection")
    print("=" * 60)
    print("\nWARNING: This takes 5-6 hours. Run in background.")
    print("Supports resume - safe to interrupt and restart.\n")

    if not HISTORICAL_GAMES_CSV.exists():
        print("ERROR: Historical games CSV not found!")
        print("Run 'python -m nba_ml.collect_data' first.")
        return

    # Load game IDs
    games_df = pd.read_csv(HISTORICAL_GAMES_CSV, usecols=["game_id", "date"])
    all_game_ids = games_df["game_id"].unique().tolist()
    print(f"Total games: {len(all_game_ids):,}")

    # Check for existing progress (resume support)
    already_done = set()
    if REFEREE_DATA_CSV.exists():
        existing = pd.read_csv(REFEREE_DATA_CSV)
        already_done = set(existing["game_id"].unique())
        print(f"Already collected: {len(already_done):,} games")

    remaining = [gid for gid in all_game_ids if gid not in already_done]
    print(f"Remaining: {len(remaining):,} games\n")

    if not remaining:
        print("All games already collected!")
        return

    # Collect in batches and save periodically
    batch_size = 200
    results = []

    for i, game_id in enumerate(tqdm(remaining, desc="Collecting referees")):
        refs = fetch_game_officials(game_id)
        if refs:
            refs["game_id"] = game_id
            results.append(refs)

        # Save every batch_size games
        if len(results) >= batch_size:
            new_df = pd.DataFrame(results)

            if REFEREE_DATA_CSV.exists():
                existing = pd.read_csv(REFEREE_DATA_CSV)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            combined.to_csv(REFEREE_DATA_CSV, index=False)
            results = []

    # Save remaining
    if results:
        new_df = pd.DataFrame(results)
        if REFEREE_DATA_CSV.exists():
            existing = pd.read_csv(REFEREE_DATA_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(REFEREE_DATA_CSV, index=False)

    # Final summary
    final_df = pd.read_csv(REFEREE_DATA_CSV)
    print(f"\n{'=' * 60}")
    print("Collection Summary")
    print(f"{'=' * 60}")
    print(f"Total games with referee data: {len(final_df):,}")
    print(f"Unique referees: {final_df['ref_1_name'].nunique():,}")
    print(f"\nSaved to: {REFEREE_DATA_CSV}")
    print(f"File size: {REFEREE_DATA_CSV.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
