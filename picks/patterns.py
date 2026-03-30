"""
Injury-Driven Performance Pattern Detection.

Analyzes historical player data to find patterns like:
  "When Jaylen Brown is OUT, Payton Pritchard averages +8.2 PTS"
  "When LeBron sits, Austin Reaves usage jumps 25%"

This is the core insight engine — it learns recurring patterns
from actual game logs so the model can exploit them.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from nba_ml.config import PLAYER_GAME_LOGS_CSV, HISTORICAL_GAMES_CSV, DATA_DIR


# Minimum sample size to trust a pattern
MIN_GAMES_WITH_STAR = 10
MIN_GAMES_WITHOUT_STAR = 5


def load_player_logs() -> pd.DataFrame:
    """Load player game logs with necessary columns."""
    if not PLAYER_GAME_LOGS_CSV.exists():
        print(f"  WARNING: Player logs not found at {PLAYER_GAME_LOGS_CSV}")
        print("  Run: python -m nba_ml.collect_players")
        return pd.DataFrame()

    df = pd.read_csv(PLAYER_GAME_LOGS_CSV)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def identify_star_absences(player_logs: pd.DataFrame, season_cutoff: str = "2023-01-01") -> pd.DataFrame:
    """
    Find games where known star players were absent from the lineup.

    A player is considered "absent" if they have no game log entry for a game
    that their team played (they didn't play at all).

    Returns DataFrame of (game_id, team, absent_star) tuples.
    """
    # Focus on recent seasons for relevance
    recent = player_logs[player_logs["GAME_DATE"] >= season_cutoff].copy()

    if recent.empty:
        return pd.DataFrame()

    # Get all unique games per team
    team_games = recent.groupby(["TEAM_ABBREVIATION", "GAME_ID"]).size().reset_index()[
        ["TEAM_ABBREVIATION", "GAME_ID"]
    ]

    # Identify high-usage players per team (top scorers)
    player_stats = (
        recent.groupby(["TEAM_ABBREVIATION", "PLAYER_NAME"])
        .agg(
            games_played=("GAME_ID", "nunique"),
            avg_pts=("PTS", "mean"),
            avg_min=("MIN", "mean"),
        )
        .reset_index()
    )

    # "Star" = plays 25+ min and scores 18+ ppg (or top 2 on team by points)
    stars_by_team = {}
    for team in player_stats["TEAM_ABBREVIATION"].unique():
        team_players = player_stats[player_stats["TEAM_ABBREVIATION"] == team]
        # Top 3 scorers who play significant minutes
        top_players = team_players[
            (team_players["avg_min"] >= 20) & (team_players["games_played"] >= 15)
        ].nlargest(3, "avg_pts")
        stars_by_team[team] = set(top_players["PLAYER_NAME"].tolist())

    # Find games where stars were absent
    absences = []
    for team, stars in stars_by_team.items():
        team_game_ids = set(team_games[team_games["TEAM_ABBREVIATION"] == team]["GAME_ID"])

        for star in stars:
            # Games this star actually played
            star_games = set(
                recent[
                    (recent["PLAYER_NAME"] == star) & (recent["TEAM_ABBREVIATION"] == team)
                ]["GAME_ID"]
            )

            # Games the star missed
            missed_games = team_game_ids - star_games
            for game_id in missed_games:
                absences.append({
                    "GAME_ID": game_id,
                    "TEAM_ABBREVIATION": team,
                    "absent_star": star,
                })

    return pd.DataFrame(absences)


def compute_with_without_stats(
    player_logs: pd.DataFrame,
    absences: pd.DataFrame,
) -> List[Dict]:
    """
    For each teammate, compute their stats WITH vs WITHOUT each star.

    Returns list of pattern dicts:
    {
        "team": "BOS",
        "absent_star": "Jaylen Brown",
        "beneficiary": "Payton Pritchard",
        "games_with": 45,
        "games_without": 12,
        "pts_with": 11.2,
        "pts_without": 19.4,
        "pts_boost": +8.2,
        "reb_boost": +1.1,
        "ast_boost": +2.3,
        "min_boost": +8.5,
        "confidence": 0.82,
    }
    """
    if absences.empty:
        return []

    patterns = []

    for (team, star), group in absences.groupby(["TEAM_ABBREVIATION", "absent_star"]):
        missed_game_ids = set(group["GAME_ID"])

        # Get all teammates on this team
        team_logs = player_logs[player_logs["TEAM_ABBREVIATION"] == team]
        teammates = set(team_logs["PLAYER_NAME"].unique()) - {star}

        for teammate in teammates:
            tm_logs = team_logs[team_logs["PLAYER_NAME"] == teammate]

            # Split into games WITH star vs WITHOUT star
            with_star = tm_logs[~tm_logs["GAME_ID"].isin(missed_game_ids)]
            without_star = tm_logs[tm_logs["GAME_ID"].isin(missed_game_ids)]

            if len(with_star) < MIN_GAMES_WITH_STAR or len(without_star) < MIN_GAMES_WITHOUT_STAR:
                continue

            # Compute averages
            stats_with = {
                "pts": with_star["PTS"].mean(),
                "reb": with_star["REB"].mean(),
                "ast": with_star["AST"].mean(),
                "min": with_star["MIN"].mean(),
            }
            stats_without = {
                "pts": without_star["PTS"].mean(),
                "reb": without_star["REB"].mean(),
                "ast": without_star["AST"].mean(),
                "min": without_star["MIN"].mean(),
            }

            pts_boost = stats_without["pts"] - stats_with["pts"]
            reb_boost = stats_without["reb"] - stats_with["reb"]
            ast_boost = stats_without["ast"] - stats_with["ast"]
            min_boost = stats_without["min"] - stats_with["min"]

            # Only keep meaningful boosts (at least 2+ pts or 1.5+ reb/ast)
            if abs(pts_boost) < 2.0 and abs(reb_boost) < 1.5 and abs(ast_boost) < 1.5:
                continue

            # Confidence based on sample size and consistency
            n_without = len(without_star)
            sample_confidence = min(1.0, n_without / 20)  # Max confidence at 20+ games

            # Check consistency (low std = more reliable pattern)
            pts_std_without = without_star["PTS"].std()
            consistency = max(0.3, 1.0 - (pts_std_without / (stats_without["pts"] + 0.1)))

            confidence = round(sample_confidence * consistency, 2)

            patterns.append({
                "team": team,
                "absent_star": star,
                "beneficiary": teammate,
                "games_with": len(with_star),
                "games_without": n_without,
                "pts_with": round(stats_with["pts"], 1),
                "pts_without": round(stats_without["pts"], 1),
                "pts_boost": round(pts_boost, 1),
                "reb_with": round(stats_with["reb"], 1),
                "reb_without": round(stats_without["reb"], 1),
                "reb_boost": round(reb_boost, 1),
                "ast_with": round(stats_with["ast"], 1),
                "ast_without": round(stats_without["ast"], 1),
                "ast_boost": round(ast_boost, 1),
                "min_with": round(stats_with["min"], 1),
                "min_without": round(stats_without["min"], 1),
                "min_boost": round(min_boost, 1),
                "confidence": confidence,
            })

    # Sort by absolute points boost (most impactful first)
    patterns.sort(key=lambda x: abs(x["pts_boost"]), reverse=True)
    return patterns


def find_active_patterns(
    patterns: List[Dict],
    out_players: Dict[str, List[str]],
) -> List[Dict]:
    """
    Given today's injury report, find which patterns are active.

    Args:
        patterns: All historical patterns from compute_with_without_stats
        out_players: Dict of team -> list of OUT player names (from injury report)

    Returns:
        List of active patterns for today's games.
    """
    active = []

    for pattern in patterns:
        team = pattern["team"]
        star = pattern["absent_star"]

        if team in out_players and star in out_players[team]:
            active.append(pattern)

    return active


def format_patterns(patterns: List[Dict], top_n: int = 20) -> str:
    """Format patterns into a readable summary for Claude analysis."""
    if not patterns:
        return "No injury-driven performance patterns detected for today's games."

    lines = ["INJURY-DRIVEN PERFORMANCE PATTERNS (data-backed):"]
    lines.append("=" * 60)

    shown = patterns[:top_n]
    for p in shown:
        lines.append(
            f"\n  When {p['absent_star']} ({p['team']}) is OUT:"
        )
        lines.append(
            f"    {p['beneficiary']} averages:"
        )
        lines.append(
            f"      PTS: {p['pts_with']} -> {p['pts_without']} ({p['pts_boost']:+.1f})"
        )
        lines.append(
            f"      REB: {p['reb_with']} -> {p['reb_without']} ({p['reb_boost']:+.1f})"
        )
        lines.append(
            f"      AST: {p['ast_with']} -> {p['ast_without']} ({p['ast_boost']:+.1f})"
        )
        lines.append(
            f"      MIN: {p['min_with']} -> {p['min_without']} ({p['min_boost']:+.1f})"
        )
        lines.append(
            f"      Sample: {p['games_without']} games without, "
            f"{p['games_with']} games with | Confidence: {p['confidence']:.0%}"
        )

    return "\n".join(lines)


def build_pattern_database() -> List[Dict]:
    """
    Full pipeline: load data, find absences, compute patterns.
    Cache results for quick lookup during prediction time.
    """
    print("  Loading player game logs for pattern detection...")
    player_logs = load_player_logs()

    if player_logs.empty:
        return []

    print("  Identifying star player absences...")
    absences = identify_star_absences(player_logs)
    print(f"  Found {len(absences)} star absence events")

    if absences.empty:
        return []

    print("  Computing with/without stat differentials...")
    patterns = compute_with_without_stats(player_logs, absences)
    print(f"  Discovered {len(patterns)} significant performance patterns")

    # Cache patterns to disk
    cache_path = DATA_DIR / "injury_patterns_cache.json"
    try:
        pd.DataFrame(patterns).to_json(cache_path, orient="records", indent=2)
        print(f"  Cached patterns to {cache_path}")
    except Exception as e:
        print(f"  Warning: Could not cache patterns: {e}")

    return patterns


def load_cached_patterns() -> List[Dict]:
    """Load cached patterns if available, otherwise build fresh."""
    cache_path = DATA_DIR / "injury_patterns_cache.json"

    if cache_path.exists():
        try:
            df = pd.read_json(cache_path, orient="records")
            patterns = df.to_dict("records")
            print(f"  Loaded {len(patterns)} cached patterns")
            return patterns
        except Exception:
            pass

    return build_pattern_database()
