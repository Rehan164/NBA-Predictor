"""
NBA Historical Data Collection Script

Collects 25 years of NBA game data from the official NBA API
and saves it to a unified CSV file for ML training.

Usage:
    python -m nba_ml.collect_data
"""

import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from nba_api.stats.endpoints import (
    LeagueGameLog,
    BoxScoreAdvancedV2,
)
from nba_api.stats.static import teams

from .config import (
    START_SEASON, END_SEASON,
    NBA_API_DELAY,
    HISTORICAL_GAMES_CSV,
    RAW_DATA_DIR,
)


def get_all_teams():
    """Get list of all NBA teams with IDs."""
    return {t["abbreviation"]: t["id"] for t in teams.get_teams()}


def fetch_season_games(season_year: int) -> pd.DataFrame:
    """
    Fetch all games for a given season using LeagueGameLog.

    Args:
        season_year: The ending year of the season (e.g., 2024 for 2023-24)

    Returns:
        DataFrame with all games from that season
    """
    season_str = f"{season_year - 1}-{str(season_year)[2:]}"

    try:
        # Get all team game logs for the season
        game_log = LeagueGameLog(
            season=season_str,
            season_type_all_star="Regular Season",
        )
        time.sleep(NBA_API_DELAY)

        df = game_log.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"  Error fetching season {season_str}: {e}")
        return pd.DataFrame()


def fetch_advanced_box_score(game_id: str) -> dict:
    """Fetch advanced stats for a single game."""
    try:
        box = BoxScoreAdvancedV2(game_id=game_id)
        time.sleep(NBA_API_DELAY)

        team_stats = box.get_data_frames()[1]  # Team stats

        if len(team_stats) >= 2:
            # First row is away team, second is home team
            away = team_stats.iloc[0]
            home = team_stats.iloc[1]

            return {
                "home_off_rating": home.get("OFF_RATING"),
                "home_def_rating": home.get("DEF_RATING"),
                "home_net_rating": home.get("NET_RATING"),
                "home_pace": home.get("PACE"),
                "home_ts_pct": home.get("TS_PCT"),
                "home_efg_pct": home.get("EFG_PCT"),
                "away_off_rating": away.get("OFF_RATING"),
                "away_def_rating": away.get("DEF_RATING"),
                "away_net_rating": away.get("NET_RATING"),
                "away_pace": away.get("PACE"),
                "away_ts_pct": away.get("TS_PCT"),
                "away_efg_pct": away.get("EFG_PCT"),
            }
    except Exception as e:
        pass  # Silent fail, advanced stats are supplementary

    return {}


def process_game_logs_to_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level game logs to game-level rows.

    NBA API returns one row per team per game. We need to combine
    into one row per game with home/away prefixed columns.
    """
    if df.empty:
        return pd.DataFrame()

    # Group by GAME_ID
    games = []

    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue

        # Determine home vs away from MATCHUP column
        # Format: "PHI vs. BOS" (home) or "BOS @ PHI" (away)
        row1 = group.iloc[0]
        row2 = group.iloc[1]

        if " vs. " in str(row1["MATCHUP"]):
            home = row1
            away = row2
        else:
            home = row2
            away = row1

        game = {
            "game_id": game_id,
            "date": pd.to_datetime(home["GAME_DATE"]).strftime("%Y-%m-%d"),
            "season": int(str(home["SEASON_ID"])[1:]),  # e.g., "22023" -> 2023

            # Teams
            "home_team": home["TEAM_ABBREVIATION"],
            "away_team": away["TEAM_ABBREVIATION"],
            "home_team_id": home["TEAM_ID"],
            "away_team_id": away["TEAM_ID"],

            # Scores
            "home_score": int(home["PTS"]),
            "away_score": int(away["PTS"]),

            # Home box score
            "home_fg_made": int(home["FGM"]),
            "home_fg_att": int(home["FGA"]),
            "home_fg_pct": float(home["FG_PCT"]) if home["FG_PCT"] else 0,
            "home_fg3_made": int(home["FG3M"]),
            "home_fg3_att": int(home["FG3A"]),
            "home_fg3_pct": float(home["FG3_PCT"]) if home["FG3_PCT"] else 0,
            "home_ft_made": int(home["FTM"]),
            "home_ft_att": int(home["FTA"]),
            "home_ft_pct": float(home["FT_PCT"]) if home["FT_PCT"] else 0,
            "home_oreb": int(home["OREB"]),
            "home_dreb": int(home["DREB"]),
            "home_reb": int(home["REB"]),
            "home_ast": int(home["AST"]),
            "home_stl": int(home["STL"]),
            "home_blk": int(home["BLK"]),
            "home_tov": int(home["TOV"]),
            "home_pf": int(home["PF"]),

            # Away box score
            "away_fg_made": int(away["FGM"]),
            "away_fg_att": int(away["FGA"]),
            "away_fg_pct": float(away["FG_PCT"]) if away["FG_PCT"] else 0,
            "away_fg3_made": int(away["FG3M"]),
            "away_fg3_att": int(away["FG3A"]),
            "away_fg3_pct": float(away["FG3_PCT"]) if away["FG3_PCT"] else 0,
            "away_ft_made": int(away["FTM"]),
            "away_ft_att": int(away["FTA"]),
            "away_ft_pct": float(away["FT_PCT"]) if away["FT_PCT"] else 0,
            "away_oreb": int(away["OREB"]),
            "away_dreb": int(away["DREB"]),
            "away_reb": int(away["REB"]),
            "away_ast": int(away["AST"]),
            "away_stl": int(away["STL"]),
            "away_blk": int(away["BLK"]),
            "away_tov": int(away["TOV"]),
            "away_pf": int(away["PF"]),

            # Derived stats
            "total_score": int(home["PTS"]) + int(away["PTS"]),
            "home_margin": int(home["PTS"]) - int(away["PTS"]),

            # Outcome columns (will be filled when betting lines are added)
            "home_win": 1 if int(home["PTS"]) > int(away["PTS"]) else 0,
        }

        games.append(game)

    return pd.DataFrame(games)


def collect_all_seasons(start: int, end: int, fetch_advanced: bool = False) -> pd.DataFrame:
    """
    Collect game data for all seasons in range.

    Args:
        start: Starting season year (e.g., 2000 for 1999-2000)
        end: Ending season year (e.g., 2025 for 2024-2025)
        fetch_advanced: Whether to fetch advanced box scores (slower)

    Returns:
        DataFrame with all games
    """
    all_games = []

    for year in tqdm(range(start, end + 1), desc="Collecting seasons"):
        print(f"\n  Fetching {year - 1}-{str(year)[2:]} season...")

        season_df = fetch_season_games(year)
        if season_df.empty:
            print(f"    No data for season {year}")
            continue

        games_df = process_game_logs_to_games(season_df)
        print(f"    Found {len(games_df)} games")

        all_games.append(games_df)

        # Be nice to the API between seasons
        time.sleep(1)

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    return combined


def add_betting_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add betting outcome columns based on margin and total.

    Note: Actual spread/total lines would need to come from a betting
    data source. Here we just calculate the outcomes that would be
    used for training once lines are added.
    """
    # These will be populated when betting lines are merged
    df["spread_line"] = None  # Home team spread
    df["total_line"] = None   # Over/under line
    df["home_ml"] = None      # Home moneyline odds
    df["away_ml"] = None      # Away moneyline odds

    # Outcome placeholders (will be calculated when lines are added)
    df["home_cover"] = None   # 1 if home covers spread
    df["total_over"] = None   # 1 if total goes over

    return df


def main():
    """Main data collection routine."""
    print("=" * 60)
    print("NBA Historical Data Collection")
    print("=" * 60)
    print(f"\nCollecting seasons {START_SEASON} to {END_SEASON}")
    print(f"Output: {HISTORICAL_GAMES_CSV}\n")

    # Collect all game data
    df = collect_all_seasons(START_SEASON, END_SEASON)

    if df.empty:
        print("No data collected!")
        return

    # Add betting outcome columns
    df = add_betting_outcomes(df)

    # Summary stats
    print(f"\n{'=' * 60}")
    print("Collection Summary")
    print(f"{'=' * 60}")
    print(f"Total games: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Seasons: {df['season'].nunique()}")
    print(f"Teams: {df['home_team'].nunique()}")

    # Save to CSV
    df.to_csv(HISTORICAL_GAMES_CSV, index=False)
    print(f"\nSaved to: {HISTORICAL_GAMES_CSV}")
    print(f"File size: {HISTORICAL_GAMES_CSV.stat().st_size / 1024 / 1024:.1f} MB")

    # Show sample
    print(f"\nSample rows:")
    print(df[["date", "home_team", "away_team", "home_score",
              "away_score", "home_win"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
