"""
Feature Engineering Pipeline for NBA Betting ML

Takes the raw historical game data and creates features suitable
for machine learning prediction of spread, total, and moneyline outcomes.

Features include:
  - Rolling team stats (multiple windows)
  - Advanced efficiency metrics
  - Travel distance between teams
  - Player-level stats (star power, depth, dependency)
  - Referee tendencies (if data available)
  - Head-to-head history
  - Momentum and fatigue indicators

Usage:
    python -m nba_ml.build_features
"""

import math
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

from .config import (
    HISTORICAL_GAMES_CSV,
    PLAYER_GAME_LOGS_CSV,
    REFEREE_DATA_CSV,
    TRAINING_FEATURES_CSV,
    ROLLING_WINDOWS,
    MIN_GAMES_FOR_FEATURES,
    TEAM_COORDINATES,
)


# ═════════════════════════════════════════════════════════════════════════════
# TRAVEL DISTANCE
# ═════════════════════════════════════════════════════════════════════════════

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Calculate great-circle distance between two points in miles."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_travel_distance(team1: str, team2: str) -> float:
    """Get distance in miles between two team cities."""
    c1 = TEAM_COORDINATES.get(team1)
    c2 = TEAM_COORDINATES.get(team2)
    if c1 and c2:
        return haversine_miles(c1[0], c1[1], c2[0], c2[1])
    return 0.0


def get_timezone_diff(team1: str, team2: str) -> int:
    """Approximate timezone difference between two teams (hours)."""
    tz_map = {
        "ATL": -5, "BOS": -5, "BKN": -5, "NJN": -5, "CHA": -5, "CHH": -5,
        "CHI": -6, "CLE": -5, "DAL": -6, "DEN": -7, "DET": -5,
        "GSW": -8, "GS": -8, "HOU": -6, "IND": -5, "LAC": -8, "LAL": -8,
        "MEM": -6, "VAN": -8, "MIA": -5, "MIL": -6, "MIN": -6,
        "NOP": -6, "NO": -6, "NOH": -6, "NOK": -6, "NYK": -5, "NY": -5,
        "OKC": -6, "SEA": -8, "ORL": -5, "PHI": -5, "PHX": -7,
        "POR": -8, "SAC": -8, "SAS": -6, "SA": -6, "TOR": -5,
        "UTA": -7, "UTAH": -7, "WAS": -5, "WSH": -5,
    }
    tz1 = tz_map.get(team1, -6)
    tz2 = tz_map.get(team2, -6)
    return abs(tz1 - tz2)


# ═════════════════════════════════════════════════════════════════════════════
# PLAYER-LEVEL FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def precompute_player_game_metrics(player_csv_path) -> dict:
    """
    Pre-compute per-game team-level player metrics from player game logs.

    Returns dict keyed by game_id+team_abbr with metrics like:
      - top_scorer_pts, top_scorer_pct, top2_pct, bench_pts, team_depth
    """
    if not player_csv_path.exists():
        return {}

    print("Loading player game logs...")
    pdf = pd.read_csv(player_csv_path)
    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])

    # Ensure PTS is numeric
    pdf["PTS"] = pd.to_numeric(pdf["PTS"], errors="coerce").fillna(0)
    pdf["MIN"] = pd.to_numeric(pdf["MIN"], errors="coerce").fillna(0)
    pdf["PLUS_MINUS"] = pd.to_numeric(pdf["PLUS_MINUS"], errors="coerce").fillna(0)

    print(f"  {len(pdf):,} player-game rows loaded")

    metrics = {}

    for (game_id, team_abbr), group in tqdm(
        pdf.groupby(["GAME_ID", "TEAM_ABBREVIATION"]),
        desc="Computing player metrics"
    ):
        players_sorted = group.sort_values("PTS", ascending=False)
        team_pts = players_sorted["PTS"].sum()

        if team_pts == 0:
            continue

        # Top scorer
        top_pts = players_sorted["PTS"].iloc[0]
        top_pct = top_pts / team_pts

        # Top 2 scorers
        top2_pts = players_sorted["PTS"].iloc[:2].sum()
        top2_pct = top2_pts / team_pts

        # Top 3 scorers
        top3_pts = players_sorted["PTS"].iloc[:3].sum()
        top3_pct = top3_pts / team_pts

        # Starters proxy (top 5 by minutes)
        by_min = group.sort_values("MIN", ascending=False)
        starters = by_min.head(5)
        bench = by_min.iloc[5:]
        starters_pts = starters["PTS"].sum()
        bench_pts = bench["PTS"].sum()

        # Bench contribution ratio
        bench_pct = bench_pts / team_pts if team_pts > 0 else 0

        # Team depth: how many players scored 10+
        double_digit_scorers = (players_sorted["PTS"] >= 10).sum()

        # Star plus/minus
        star_pm = players_sorted["PLUS_MINUS"].iloc[0] if len(players_sorted) > 0 else 0

        # Scoring distribution (Herfindahl index - lower = more balanced)
        shares = (players_sorted["PTS"] / team_pts) ** 2
        hhi = shares.sum()

        key = f"{game_id}_{team_abbr}"
        metrics[key] = {
            "top_scorer_pts": top_pts,
            "top_scorer_pct": top_pct,
            "top2_scorer_pct": top2_pct,
            "top3_scorer_pct": top3_pct,
            "starters_pts": starters_pts,
            "bench_pts": bench_pts,
            "bench_pct": bench_pct,
            "double_digit_scorers": double_digit_scorers,
            "star_plus_minus": star_pm,
            "scoring_hhi": hhi,
        }

    print(f"  Computed metrics for {len(metrics):,} team-games")
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# REFEREE FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def load_referee_data(referee_csv_path) -> pd.DataFrame:
    """Load referee assignments."""
    if not referee_csv_path.exists():
        return pd.DataFrame()

    print("Loading referee data...")
    rdf = pd.read_csv(referee_csv_path)
    print(f"  {len(rdf):,} games with referee assignments")
    return rdf


def build_referee_histories(ref_df: pd.DataFrame, games_df: pd.DataFrame) -> dict:
    """
    Build rolling referee tendencies from historical games.

    Returns dict keyed by ref_name with their stats.
    """
    if ref_df.empty:
        return {}

    # Merge referee data with game outcomes
    merged = ref_df.merge(
        games_df[["game_id", "date", "home_win", "total_score", "home_margin"]],
        on="game_id",
        how="inner",
    )
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date")

    # Build per-referee rolling stats
    ref_stats = {}

    for ref_col in ["ref_1_name", "ref_2_name", "ref_3_name"]:
        if ref_col not in merged.columns:
            continue

        for ref_name, group in merged.groupby(ref_col):
            if pd.isna(ref_name) or ref_name == "":
                continue

            if ref_name not in ref_stats:
                ref_stats[ref_name] = []

            for _, row in group.iterrows():
                ref_stats[ref_name].append({
                    "date": row["date"],
                    "game_id": row["game_id"],
                    "home_win": row["home_win"],
                    "total": row["total_score"],
                    "margin": abs(row["home_margin"]),
                })

    # Sort each ref's history by date
    for ref_name in ref_stats:
        ref_stats[ref_name] = sorted(ref_stats[ref_name], key=lambda x: x["date"])

    return ref_stats


def get_referee_features_for_game(game_id, ref_df: pd.DataFrame,
                                   ref_histories: dict, game_date) -> dict:
    """Get referee tendency features for a specific game."""
    features = {}

    if ref_df.empty or not ref_histories:
        return features

    game_refs = ref_df[ref_df["game_id"] == game_id]
    if game_refs.empty:
        return features

    game_date = pd.to_datetime(game_date)
    ref_home_rates = []
    ref_totals = []
    ref_margins = []
    ref_experience = []

    for ref_col in ["ref_1_name", "ref_2_name", "ref_3_name"]:
        if ref_col not in game_refs.columns:
            continue

        ref_name = game_refs[ref_col].iloc[0]
        if pd.isna(ref_name) or ref_name == "" or ref_name not in ref_histories:
            continue

        # Get this ref's games BEFORE this date
        prior = [g for g in ref_histories[ref_name] if g["date"] < game_date]

        if len(prior) >= 20:
            recent = prior[-50:]  # Last 50 games
            home_rate = sum(g["home_win"] for g in recent) / len(recent)
            avg_total = sum(g["total"] for g in recent) / len(recent)
            avg_margin = sum(g["margin"] for g in recent) / len(recent)

            ref_home_rates.append(home_rate)
            ref_totals.append(avg_total)
            ref_margins.append(avg_margin)
            ref_experience.append(len(prior))

    if ref_home_rates:
        features["ref_home_win_rate"] = np.mean(ref_home_rates)
        features["ref_avg_total"] = np.mean(ref_totals)
        features["ref_avg_margin"] = np.mean(ref_margins)
        features["ref_experience"] = np.mean(ref_experience)
        features["ref_total_bias"] = np.mean(ref_totals) - 210  # Deviation from league avg

    return features


# ═════════════════════════════════════════════════════════════════════════════
# ROLLING TEAM STATS
# ═════════════════════════════════════════════════════════════════════════════

def calculate_rolling_stats(team_games: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Calculate rolling statistics for a team's games.

    Args:
        team_games: DataFrame of a team's games sorted by date
        windows: List of window sizes (e.g., [3, 5, 10, 15, 20])

    Returns:
        DataFrame with rolling stat columns
    """
    stats = {}

    for w in windows:
        suffix = f"_l{w}"

        # Points
        stats[f"ppg{suffix}"] = team_games["pts"].rolling(w, min_periods=1).mean()
        stats[f"opp_ppg{suffix}"] = team_games["opp_pts"].rolling(w, min_periods=1).mean()

        # Margin
        margin = team_games["pts"] - team_games["opp_pts"]
        stats[f"margin{suffix}"] = margin.rolling(w, min_periods=1).mean()

        # Shooting
        stats[f"fg_pct{suffix}"] = team_games["fg_pct"].rolling(w, min_periods=1).mean()
        stats[f"fg3_pct{suffix}"] = team_games["fg3_pct"].rolling(w, min_periods=1).mean()
        stats[f"ft_pct{suffix}"] = team_games["ft_pct"].rolling(w, min_periods=1).mean()

        # Rebounds
        stats[f"reb{suffix}"] = team_games["reb"].rolling(w, min_periods=1).mean()
        stats[f"oreb{suffix}"] = team_games["oreb"].rolling(w, min_periods=1).mean()

        # Other
        stats[f"ast{suffix}"] = team_games["ast"].rolling(w, min_periods=1).mean()
        stats[f"tov{suffix}"] = team_games["tov"].rolling(w, min_periods=1).mean()

        # Win percentage
        stats[f"win_pct{suffix}"] = team_games["won"].rolling(w, min_periods=1).mean()

        # Assist-to-turnover ratio
        ast_roll = team_games["ast"].rolling(w, min_periods=1).mean()
        tov_roll = team_games["tov"].rolling(w, min_periods=1).mean().replace(0, 1)
        stats[f"ast_tov_ratio{suffix}"] = ast_roll / tov_roll

        # Offensive efficiency proxy
        pts_roll = team_games["pts"].rolling(w, min_periods=1).mean()
        opp_roll = team_games["opp_pts"].rolling(w, min_periods=1).mean()
        total_roll = (pts_roll + opp_roll).replace(0, 1)
        stats[f"off_efficiency{suffix}"] = pts_roll / total_roll * 100
        stats[f"def_efficiency{suffix}"] = 100 - opp_roll / total_roll * 100

        # Scoring variance (consistency)
        stats[f"pts_std{suffix}"] = team_games["pts"].rolling(w, min_periods=2).std().fillna(0)

        # Margin variance
        stats[f"margin_std{suffix}"] = margin.rolling(w, min_periods=2).std().fillna(0)

        # Close games (decided by 5 or less)
        close_games = (abs(margin) <= 5).astype(int)
        stats[f"close_game_pct{suffix}"] = close_games.rolling(w, min_periods=1).mean()

        # Blowout wins (won by 15+)
        blowout_wins = (margin >= 15).astype(int)
        stats[f"blowout_win_pct{suffix}"] = blowout_wins.rolling(w, min_periods=1).mean()

        # Player metrics rolling (if available)
        if "top_scorer_pts" in team_games.columns:
            stats[f"top_scorer_pts{suffix}"] = team_games["top_scorer_pts"].rolling(w, min_periods=1).mean()
            stats[f"top_scorer_pct{suffix}"] = team_games["top_scorer_pct"].rolling(w, min_periods=1).mean()
            stats[f"bench_pct{suffix}"] = team_games["bench_pct"].rolling(w, min_periods=1).mean()
            stats[f"scoring_hhi{suffix}"] = team_games["scoring_hhi"].rolling(w, min_periods=1).mean()
            stats[f"double_digit_scorers{suffix}"] = team_games["double_digit_scorers"].rolling(w, min_periods=1).mean()

    return pd.DataFrame(stats)


# ═════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def get_head_to_head_features(team: str, opponent: str, game_date: str,
                               team_histories: dict) -> dict:
    """Get head-to-head history between two teams."""
    features = {}

    if team not in team_histories:
        return features

    team_df = team_histories[team]
    h2h_games = team_df[(team_df["opponent"] == opponent) & (team_df["date"] < game_date)]

    if len(h2h_games) >= 2:
        features["h2h_wins"] = int(h2h_games["won"].sum())
        features["h2h_losses"] = len(h2h_games) - int(h2h_games["won"].sum())
        features["h2h_win_pct"] = h2h_games["won"].mean()

        # Recent H2H (last 5 meetings)
        recent_h2h = h2h_games.tail(5)
        features["h2h_recent_win_pct"] = recent_h2h["won"].mean()
        features["h2h_avg_margin"] = (recent_h2h["pts"] - recent_h2h["opp_pts"]).mean()
        features["h2h_avg_total"] = (recent_h2h["pts"] + recent_h2h["opp_pts"]).mean()

        # Last meeting result
        last_game = h2h_games.iloc[-1]
        features["h2h_last_won"] = int(last_game["won"])
        features["h2h_last_margin"] = last_game["pts"] - last_game["opp_pts"]

        # H2H this season
        season_start = f"{int(game_date[:4])-1}-10-01"
        season_h2h = h2h_games[h2h_games["date"] >= season_start]
        if len(season_h2h) > 0:
            features["h2h_season_win_pct"] = season_h2h["won"].mean()

    return features


# ═════════════════════════════════════════════════════════════════════════════
# MOMENTUM & FATIGUE
# ═════════════════════════════════════════════════════════════════════════════

def get_momentum_features(prior_games: pd.DataFrame, prefix: str) -> dict:
    """Calculate momentum and trend features."""
    features = {}

    if len(prior_games) < 10:
        return features

    recent_5 = prior_games.tail(5)
    long_term = prior_games.tail(20)

    # Scoring momentum: recent vs long term
    features[f"{prefix}ppg_momentum"] = recent_5["pts"].mean() - long_term["pts"].mean()
    features[f"{prefix}win_momentum"] = recent_5["won"].mean() - long_term["won"].mean()

    recent_margin = (recent_5["pts"] - recent_5["opp_pts"]).mean()
    long_margin = (long_term["pts"] - long_term["opp_pts"]).mean()
    features[f"{prefix}margin_momentum"] = recent_margin - long_margin

    # Last 3 vs previous 3 (micro-trend)
    if len(prior_games) >= 6:
        last_3_margin = (prior_games.tail(3)["pts"] - prior_games.tail(3)["opp_pts"]).mean()
        prev_3_margin = (prior_games.tail(6).head(3)["pts"] - prior_games.tail(6).head(3)["opp_pts"]).mean()
        features[f"{prefix}trend_3g"] = last_3_margin - prev_3_margin

    # Defensive momentum
    features[f"{prefix}def_momentum"] = long_term["opp_pts"].mean() - recent_5["opp_pts"].mean()

    # Shooting momentum
    if "fg_pct" in recent_5.columns:
        features[f"{prefix}fg_pct_momentum"] = recent_5["fg_pct"].mean() - long_term["fg_pct"].mean()

    return features


def get_fatigue_features(prior_games: pd.DataFrame, game_date: str, prefix: str) -> dict:
    """Calculate fatigue and schedule-related features."""
    features = {}

    if len(prior_games) < 3:
        return features

    game_dt = pd.to_datetime(game_date)

    # Games in last 7 and 14 days
    week_ago = game_dt - timedelta(days=7)
    two_weeks_ago = game_dt - timedelta(days=14)
    dates = pd.to_datetime(prior_games["date"])

    features[f"{prefix}games_last_7d"] = (dates >= week_ago).sum()
    features[f"{prefix}games_last_14d"] = (dates >= two_weeks_ago).sum()

    # Average rest between games (last 10)
    recent_dates = dates.tail(10)
    if len(recent_dates) >= 2:
        rest_days = recent_dates.diff().dt.days.dropna()
        features[f"{prefix}avg_rest"] = rest_days.mean()

    # Road trip length (consecutive away games)
    recent_home = prior_games.tail(5)["is_home"].tolist()
    away_streak = 0
    for is_home in reversed(recent_home):
        if is_home == 0:
            away_streak += 1
        else:
            break
    features[f"{prefix}road_trip_length"] = away_streak

    # Performance on back-to-backs (how they handle fatigue)
    if len(prior_games) >= 20:
        recent_20 = prior_games.tail(20)
        recent_dates_20 = pd.to_datetime(recent_20["date"])
        diffs = recent_dates_20.diff().dt.days
        b2b_mask = (diffs <= 1)
        b2b_games = recent_20[b2b_mask.values]
        if len(b2b_games) >= 2:
            features[f"{prefix}b2b_win_rate"] = b2b_games["won"].mean()

    return features


# ═════════════════════════════════════════════════════════════════════════════
# TEAM GAME HISTORY
# ═════════════════════════════════════════════════════════════════════════════

def build_team_game_history(df: pd.DataFrame, player_metrics: dict = None) -> dict:
    """
    Build a dictionary of each team's game history with stats.

    Args:
        df: Game-level DataFrame
        player_metrics: Pre-computed player metrics dict (optional)

    Returns:
        {team_abbr: DataFrame with game-by-game stats}
    """
    team_histories = {}
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

    for team in tqdm(all_teams, desc="Building team histories"):
        # Home games
        home_games = df[df["home_team"] == team].copy()
        home_games["is_home"] = 1
        home_games["pts"] = home_games["home_score"]
        home_games["opp_pts"] = home_games["away_score"]
        home_games["fg_pct"] = home_games["home_fg_pct"]
        home_games["fg3_pct"] = home_games["home_fg3_pct"]
        home_games["ft_pct"] = home_games["home_ft_pct"]
        home_games["reb"] = home_games["home_reb"]
        home_games["oreb"] = home_games["home_oreb"]
        home_games["ast"] = home_games["home_ast"]
        home_games["tov"] = home_games["home_tov"]
        home_games["won"] = home_games["home_win"]
        home_games["opponent"] = home_games["away_team"]

        # Away games
        away_games = df[df["away_team"] == team].copy()
        away_games["is_home"] = 0
        away_games["pts"] = away_games["away_score"]
        away_games["opp_pts"] = away_games["home_score"]
        away_games["fg_pct"] = away_games["away_fg_pct"]
        away_games["fg3_pct"] = away_games["away_fg3_pct"]
        away_games["ft_pct"] = away_games["away_ft_pct"]
        away_games["reb"] = away_games["away_reb"]
        away_games["oreb"] = away_games["away_oreb"]
        away_games["ast"] = away_games["away_ast"]
        away_games["tov"] = away_games["away_tov"]
        away_games["won"] = 1 - away_games["home_win"]
        away_games["opponent"] = away_games["home_team"]

        # Merge player metrics if available
        if player_metrics:
            for subset in [home_games, away_games]:
                for idx in subset.index:
                    key = f"{subset.at[idx, 'game_id']}_{team}"
                    if key in player_metrics:
                        for metric, value in player_metrics[key].items():
                            subset.at[idx, metric] = value

        team_games = pd.concat([home_games, away_games])
        team_games = team_games.sort_values("date").reset_index(drop=True)
        team_histories[team] = team_games

    return team_histories


# ═════════════════════════════════════════════════════════════════════════════
# PER-GAME FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def get_team_features_before_game(team: str, game_date: str, team_histories: dict,
                                   is_home: bool, opponent: str = None) -> dict:
    """
    Get team's rolling features BEFORE a specific game (no data leakage).
    """
    if team not in team_histories:
        return {}

    team_df = team_histories[team]
    prior_games = team_df[team_df["date"] < game_date]

    if len(prior_games) < MIN_GAMES_FOR_FEATURES:
        return {}

    # Calculate rolling stats
    rolling = calculate_rolling_stats(prior_games, ROLLING_WINDOWS)
    if rolling.empty:
        return {}

    latest = rolling.iloc[-1].to_dict()

    # Add prefix based on home/away
    prefix = "home_" if is_home else "away_"
    features = {f"{prefix}{k}": v for k, v in latest.items()}

    # Season record
    season_games = prior_games[prior_games["date"] >= f"{int(game_date[:4])-1}-10-01"]
    if len(season_games) > 0:
        features[f"{prefix}season_wins"] = int(season_games["won"].sum())
        features[f"{prefix}season_losses"] = len(season_games) - int(season_games["won"].sum())
        features[f"{prefix}season_win_pct"] = season_games["won"].mean()
        features[f"{prefix}season_ppg"] = season_games["pts"].mean()
        features[f"{prefix}season_opp_ppg"] = season_games["opp_pts"].mean()
        features[f"{prefix}season_margin"] = (season_games["pts"] - season_games["opp_pts"]).mean()

    # Home/away splits
    home_only = prior_games[prior_games["is_home"] == 1].tail(20)
    away_only = prior_games[prior_games["is_home"] == 0].tail(20)

    if len(home_only) >= 5:
        features[f"{prefix}home_win_pct"] = home_only["won"].mean()
        features[f"{prefix}home_ppg"] = home_only["pts"].mean()
        features[f"{prefix}home_margin"] = (home_only["pts"] - home_only["opp_pts"]).mean()
    if len(away_only) >= 5:
        features[f"{prefix}away_win_pct"] = away_only["won"].mean()
        features[f"{prefix}away_ppg"] = away_only["pts"].mean()
        features[f"{prefix}away_margin"] = (away_only["pts"] - away_only["opp_pts"]).mean()

    # Streak
    streak = 0
    for won in reversed(prior_games["won"].tolist()[-10:]):
        if won == prior_games["won"].iloc[-1]:
            streak += 1
        else:
            break
    features[f"{prefix}streak"] = streak if prior_games["won"].iloc[-1] else -streak

    # Rest days & back-to-back
    last_game_date = pd.to_datetime(prior_games["date"].iloc[-1])
    current_date = pd.to_datetime(game_date)
    features[f"{prefix}rest_days"] = (current_date - last_game_date).days
    features[f"{prefix}b2b"] = 1 if features[f"{prefix}rest_days"] <= 1 else 0

    # Momentum
    features.update(get_momentum_features(prior_games, prefix))

    # Fatigue
    features.update(get_fatigue_features(prior_games, game_date, prefix))

    # Head-to-head
    if opponent:
        h2h = get_head_to_head_features(team, opponent, game_date, team_histories)
        features.update({f"{prefix}{k}": v for k, v in h2h.items()})

    return features


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_training_features(df: pd.DataFrame, player_metrics: dict = None,
                             ref_df: pd.DataFrame = None,
                             ref_histories: dict = None) -> pd.DataFrame:
    """
    Build feature matrix for all games.

    For each game, calculate rolling features for both teams
    using only data available BEFORE that game.
    """
    print("Building team game histories...")
    team_histories = build_team_game_history(df, player_metrics)

    print(f"\nGenerating features for {len(df):,} games...")
    features_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating features"):
        game_features = {
            "game_id": row["game_id"],
            "date": row["date"],
            "season": row["season"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],

            # Targets
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "home_win": row["home_win"],
            "total_score": row["total_score"],
            "home_margin": row["home_margin"],

            # Betting lines (if available)
            "spread_line": row.get("spread_line"),
            "total_line": row.get("total_line"),
        }

        # ── Home team features ──
        home_feats = get_team_features_before_game(
            row["home_team"], row["date"], team_histories,
            is_home=True, opponent=row["away_team"]
        )
        game_features.update(home_feats)

        # ── Away team features ──
        away_feats = get_team_features_before_game(
            row["away_team"], row["date"], team_histories,
            is_home=False, opponent=row["home_team"]
        )
        game_features.update(away_feats)

        # ── Travel distance ──
        game_features["travel_distance"] = get_travel_distance(
            row["away_team"], row["home_team"]
        )
        game_features["timezone_diff"] = get_timezone_diff(
            row["away_team"], row["home_team"]
        )

        # ── Referee features ──
        if ref_df is not None and ref_histories:
            ref_feats = get_referee_features_for_game(
                row["game_id"], ref_df, ref_histories, row["date"]
            )
            game_features.update(ref_feats)

        # ── Derived matchup features ──
        if "home_ppg_l10" in game_features and "away_ppg_l10" in game_features:
            # Projected total (average of both teams' offense + defense)
            game_features["projected_total"] = (
                game_features.get("home_ppg_l10", 100) +
                game_features.get("away_ppg_l10", 100) +
                game_features.get("home_opp_ppg_l10", 100) +
                game_features.get("away_opp_ppg_l10", 100)
            ) / 2

            # Power differentials
            game_features["margin_diff_l10"] = (
                game_features.get("home_margin_l10", 0) -
                game_features.get("away_margin_l10", 0)
            )
            game_features["margin_diff_l5"] = (
                game_features.get("home_margin_l5", 0) -
                game_features.get("away_margin_l5", 0)
            )
            game_features["win_pct_diff_l10"] = (
                game_features.get("home_win_pct_l10", 0.5) -
                game_features.get("away_win_pct_l10", 0.5)
            )
            game_features["win_pct_diff_l20"] = (
                game_features.get("home_win_pct_l20", 0.5) -
                game_features.get("away_win_pct_l20", 0.5)
            )

            # Efficiency differentials
            game_features["off_efficiency_diff"] = (
                game_features.get("home_off_efficiency_l10", 50) -
                game_features.get("away_off_efficiency_l10", 50)
            )
            game_features["def_efficiency_diff"] = (
                game_features.get("home_def_efficiency_l10", 50) -
                game_features.get("away_def_efficiency_l10", 50)
            )

            # Pace mismatch (scoring totals indicate pace)
            game_features["pace_mismatch"] = abs(
                (game_features.get("home_ppg_l10", 100) + game_features.get("home_opp_ppg_l10", 100)) -
                (game_features.get("away_ppg_l10", 100) + game_features.get("away_opp_ppg_l10", 100))
            )

            # Rest advantage
            home_rest = game_features.get("home_rest_days", 1)
            away_rest = game_features.get("away_rest_days", 1)
            game_features["rest_advantage"] = home_rest - away_rest

            # Streak differential
            game_features["streak_diff"] = (
                game_features.get("home_streak", 0) -
                game_features.get("away_streak", 0)
            )

            # Consistency differential (lower std = more consistent)
            game_features["consistency_diff"] = (
                game_features.get("away_pts_std_l10", 10) -
                game_features.get("home_pts_std_l10", 10)
            )

            # Season record differential
            game_features["season_win_pct_diff"] = (
                game_features.get("home_season_win_pct", 0.5) -
                game_features.get("away_season_win_pct", 0.5)
            )

            # Fatigue differential
            game_features["fatigue_diff"] = (
                game_features.get("away_games_last_7d", 3) -
                game_features.get("home_games_last_7d", 3)
            )

            # Travel fatigue for away team
            game_features["away_travel_fatigue"] = (
                game_features.get("travel_distance", 0) *
                game_features.get("away_b2b", 0)
            )

            # Momentum differentials
            game_features["momentum_diff"] = (
                game_features.get("home_margin_momentum", 0) -
                game_features.get("away_margin_momentum", 0)
            )

        # Player-level differentials (if available)
        if "home_top_scorer_pts_l10" in game_features and "away_top_scorer_pts_l10" in game_features:
            game_features["star_power_diff"] = (
                game_features.get("home_top_scorer_pts_l10", 20) -
                game_features.get("away_top_scorer_pts_l10", 20)
            )
            game_features["depth_diff"] = (
                game_features.get("home_bench_pct_l10", 0.3) -
                game_features.get("away_bench_pct_l10", 0.3)
            )
            game_features["balance_diff"] = (
                game_features.get("away_scoring_hhi_l10", 0.1) -
                game_features.get("home_scoring_hhi_l10", 0.1)
            )

        features_list.append(game_features)

    features_df = pd.DataFrame(features_list)

    # Drop rows with insufficient features
    required_cols = ["home_ppg_l10", "away_ppg_l10"]
    before = len(features_df)
    features_df = features_df.dropna(subset=required_cols)
    print(f"\nDropped {before - len(features_df)} games with insufficient history")

    return features_df


def main():
    """Main feature engineering routine."""
    print("=" * 60)
    print("NBA Feature Engineering Pipeline")
    print("=" * 60)

    # ── Load game data ──
    print(f"\nLoading data from {HISTORICAL_GAMES_CSV}...")
    if not HISTORICAL_GAMES_CSV.exists():
        print("ERROR: Historical games CSV not found!")
        print("Run 'python -m nba_ml.collect_data' first.")
        return

    df = pd.read_csv(HISTORICAL_GAMES_CSV)
    print(f"Loaded {len(df):,} games")

    # ── Load player data (optional) ──
    player_metrics = None
    if PLAYER_GAME_LOGS_CSV.exists():
        player_metrics = precompute_player_game_metrics(PLAYER_GAME_LOGS_CSV)
    else:
        print(f"\nPlayer data not found at {PLAYER_GAME_LOGS_CSV}")
        print("Run 'python -m nba_ml.collect_players' for player features.")
        print("Continuing without player features...\n")

    # ── Load referee data (optional) ──
    ref_df = pd.DataFrame()
    ref_histories = {}
    if REFEREE_DATA_CSV.exists():
        ref_df = load_referee_data(REFEREE_DATA_CSV)
        if not ref_df.empty:
            ref_histories = build_referee_histories(ref_df, df)
            print(f"  Built histories for {len(ref_histories):,} referees")
    else:
        print(f"\nReferee data not found at {REFEREE_DATA_CSV}")
        print("Run 'python -m nba_ml.collect_referees' for referee features.")
        print("Continuing without referee features...\n")

    # ── Build features ──
    features_df = build_training_features(
        df, player_metrics,
        ref_df if not ref_df.empty else None,
        ref_histories if ref_histories else None,
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Feature Engineering Summary")
    print(f"{'=' * 60}")
    print(f"Total games with features: {len(features_df):,}")
    print(f"Number of columns: {len(features_df.columns)}")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")

    feature_cols = [c for c in features_df.columns
                    if c not in ["game_id", "date", "season", "home_team", "away_team",
                                 "home_score", "away_score", "home_win", "total_score",
                                 "home_margin", "spread_line", "total_line",
                                 "home_cover", "total_over"]]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in sorted(feature_cols)[:30]:
        print(f"  - {c}")
    if len(feature_cols) > 30:
        print(f"  ... and {len(feature_cols) - 30} more")

    # Show feature categories
    categories = {
        "Rolling stats": len([c for c in feature_cols if any(f"_l{w}" in c for w in ROLLING_WINDOWS)]),
        "Travel/location": len([c for c in feature_cols if "travel" in c or "timezone" in c]),
        "Momentum": len([c for c in feature_cols if "momentum" in c or "trend" in c]),
        "Fatigue": len([c for c in feature_cols if "fatigue" in c or "rest" in c or "b2b" in c or "games_last" in c or "road_trip" in c]),
        "Head-to-head": len([c for c in feature_cols if "h2h" in c]),
        "Player-derived": len([c for c in feature_cols if "scorer" in c or "bench" in c or "hhi" in c or "depth" in c or "star_power" in c or "balance" in c]),
        "Referee": len([c for c in feature_cols if "ref_" in c]),
        "Matchup derived": len([c for c in feature_cols if "diff" in c or "projected" in c or "pace_mismatch" in c]),
    }
    print(f"\nFeature categories:")
    for cat, count in categories.items():
        status = f"{count} features" if count > 0 else "not available (run collection)"
        print(f"  {cat}: {status}")

    # Save
    features_df.to_csv(TRAINING_FEATURES_CSV, index=False)
    print(f"\nSaved to: {TRAINING_FEATURES_CSV}")
    print(f"File size: {TRAINING_FEATURES_CSV.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
