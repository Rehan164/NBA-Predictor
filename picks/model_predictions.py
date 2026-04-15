"""
Model Predictions — Run our trained ML models and format output.

Loads the trained ensemble models (spread, totals, moneyline) and
player prop models, generates predictions, and packages everything
into a structured format for Claude analysis.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from nba_ml.config import MODELS_DIR, TRAINING_FEATURES_CSV, DATA_DIR, PLAYER_GAME_LOGS_CSV

PLAYER_PROPS_DIR = MODELS_DIR / "player_props"
ET = timezone(timedelta(hours=-5))


def load_models() -> Dict[str, dict]:
    """Load all trained team-level models."""
    models = {}
    for name in ["moneyline_model", "spread_model", "totals_model"]:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {name} not found at {path}")
    return models


def load_player_prop_models() -> Dict[str, dict]:
    """Load player prop models (PTS, REB, AST)."""
    models = {}
    for stat in ["pts", "reb", "ast"]:
        path = PLAYER_PROPS_DIR / f"player_{stat}_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[stat] = pickle.load(f)
            print(f"  Loaded player_{stat}_model")
        else:
            print(f"  WARNING: player_{stat}_model not found")
    return models


def load_feature_data() -> pd.DataFrame:
    """Load the training features for building matchup features."""
    if not TRAINING_FEATURES_CSV.exists():
        print(f"  WARNING: Features not found at {TRAINING_FEATURES_CSV}")
        return pd.DataFrame()
    return pd.read_csv(TRAINING_FEATURES_CSV)


def build_matchup_features(home_team: str, away_team: str, feature_df: pd.DataFrame) -> dict:
    """Build feature vector from most recent team data."""
    features = {}

    home_as_home = feature_df[feature_df["home_team"] == home_team].sort_values("date").tail(1)
    if not home_as_home.empty:
        for col in home_as_home.columns:
            if col.startswith("home_") and col not in ["home_team", "home_score", "home_win", "home_margin"]:
                features[col] = home_as_home[col].values[0]

    away_as_away = feature_df[feature_df["away_team"] == away_team].sort_values("date").tail(1)
    if not away_as_away.empty:
        for col in away_as_away.columns:
            if col.startswith("away_") and col not in ["away_team", "away_score"]:
                features[col] = away_as_away[col].values[0]

    # Derived matchup features
    if "home_ppg_l10" in features and "away_ppg_l10" in features:
        features["projected_total"] = (
            features.get("home_ppg_l10", 100) + features.get("away_ppg_l10", 100) +
            features.get("home_opp_ppg_l10", 100) + features.get("away_opp_ppg_l10", 100)
        ) / 2
        features["margin_diff_l10"] = features.get("home_margin_l10", 0) - features.get("away_margin_l10", 0)
        features["margin_diff_l5"] = features.get("home_margin_l5", 0) - features.get("away_margin_l5", 0)
        features["win_pct_diff_l10"] = features.get("home_win_pct_l10", 0.5) - features.get("away_win_pct_l10", 0.5)
        features["win_pct_diff_l20"] = features.get("home_win_pct_l20", 0.5) - features.get("away_win_pct_l20", 0.5)
        features["off_efficiency_diff"] = features.get("home_off_efficiency_l10", 50) - features.get("away_off_efficiency_l10", 50)
        features["def_efficiency_diff"] = features.get("home_def_efficiency_l10", 50) - features.get("away_def_efficiency_l10", 50)
        features["rest_advantage"] = features.get("home_rest_days", 1) - features.get("away_rest_days", 1)
        features["streak_diff"] = features.get("home_streak", 0) - features.get("away_streak", 0)
        features["season_win_pct_diff"] = features.get("home_season_win_pct", 0.5) - features.get("away_season_win_pct", 0.5)
        features["momentum_diff"] = features.get("home_margin_momentum", 0) - features.get("away_margin_momentum", 0)
        features["consistency_diff"] = features.get("away_pts_std_l10", 10) - features.get("home_pts_std_l10", 10)
        features["fatigue_diff"] = features.get("away_games_last_7d", 3) - features.get("home_games_last_7d", 3)
        features["pace_mismatch"] = abs(
            (features.get("home_ppg_l10", 100) + features.get("home_opp_ppg_l10", 100)) -
            (features.get("away_ppg_l10", 100) + features.get("away_opp_ppg_l10", 100))
        )

    try:
        from nba_ml.build_features import get_travel_distance, get_timezone_diff
        features["travel_distance"] = get_travel_distance(away_team, home_team)
        features["timezone_diff"] = get_timezone_diff(away_team, home_team)
        features["away_travel_fatigue"] = features.get("travel_distance", 0) * features.get("away_b2b", 0)
    except ImportError:
        features["travel_distance"] = 0
        features["timezone_diff"] = 0
        features["away_travel_fatigue"] = 0

    return features


def _predict_with_model(model_data, x):
    """Get predictions from single model or ensemble."""
    model_type = model_data.get("model_type", "classifier")

    if model_type == "ensemble_classifier":
        models = model_data["models"]
        meta = model_data["meta_learner"]
        base_preds = np.column_stack([
            models["xgb"].predict_proba(x)[:, 1],
            models["lgb"].predict_proba(x)[:, 1],
            models["cat"].predict_proba(x)[:, 1],
        ])
        prob = float(meta.predict_proba(base_preds)[0, 1])
        calibrator = model_data.get("calibrator")
        if calibrator is not None:
            prob = float(calibrator.predict([prob])[0])
        return None, prob

    elif model_type == "ensemble_regressor":
        models = model_data["models"]
        meta = model_data["meta_learner"]
        base_preds = np.column_stack([
            models["xgb"].predict(x),
            models["lgb"].predict(x),
            models["cat"].predict(x),
        ])
        return float(meta.predict(base_preds)[0]), None

    elif model_type == "classifier":
        prob = float(model_data["model"].predict_proba(x)[0, 1])
        return None, prob

    else:
        return float(model_data["model"].predict(x)[0]), None


def predict_game(
    home_team: str,
    away_team: str,
    models: Dict[str, dict],
    feature_df: pd.DataFrame,
    injuries: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Generate predictions for a single game.

    Returns structured prediction dict for Claude analysis.
    """
    features = build_matchup_features(home_team, away_team, feature_df)
    if not features:
        return {"error": f"No data for {away_team} @ {home_team}"}

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "predictions": {},
    }

    for model_name, model_data in models.items():
        feature_cols = model_data["feature_columns"]
        x = pd.DataFrame([features])
        missing = [c for c in feature_cols if c not in x.columns]
        if missing:
            missing_df = pd.DataFrame({c: [np.nan] for c in missing})
            x = pd.concat([x, missing_df], axis=1)
        x = x[feature_cols].fillna(0)

        if model_name == "moneyline_model":
            _, prob = _predict_with_model(model_data, x)
            result["predictions"]["moneyline"] = {
                "home_win_prob": round(prob, 4),
                "away_win_prob": round(1 - prob, 4),
                "pick": home_team if prob > 0.5 else away_team,
                "confidence": round(max(prob, 1 - prob) * 100, 1),
            }

        elif model_name == "spread_model":
            margin, _ = _predict_with_model(model_data, x)

            # Apply injury adjustments
            if injuries:
                from nba_ml.injuries import apply_injury_adjustments
                base_total = features.get("projected_total", 220)
                adj = apply_injury_adjustments(home_team, away_team, margin, base_total, injuries)
                margin = adj["adjusted_spread"]
                result["injury_spread_adj"] = round(adj["spread_adjustment"], 1)

            result["predictions"]["spread"] = {
                "predicted_margin": round(margin, 1),
                "favored": home_team if margin > 0 else away_team,
                "by_points": round(abs(margin), 1),
            }

        elif model_name == "totals_model":
            raw_pred, _ = _predict_with_model(model_data, x)

            if model_data.get("target_type") == "deviation":
                projected = features.get("projected_total", 220)
                predicted_total = projected + raw_pred
            else:
                predicted_total = raw_pred

            if injuries:
                from nba_ml.injuries import apply_injury_adjustments
                base_margin = result.get("predictions", {}).get("spread", {}).get("predicted_margin", 0)
                adj = apply_injury_adjustments(home_team, away_team, base_margin, predicted_total, injuries)
                predicted_total = adj["adjusted_total"]
                result["injury_total_adj"] = round(adj["total_adjustment"], 1)

            result["predictions"]["totals"] = {
                "predicted_total": round(predicted_total, 1),
            }

    return result


def get_recent_player_stats(team: str, n_games: int = 10) -> List[Dict]:
    """Get recent stats for key players on a team."""
    if not PLAYER_GAME_LOGS_CSV.exists():
        return []

    df = pd.read_csv(PLAYER_GAME_LOGS_CSV)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    team_logs = df[df["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE", ascending=False)

    # Get recent games
    recent_game_ids = team_logs["GAME_ID"].unique()[:n_games]
    recent = team_logs[team_logs["GAME_ID"].isin(recent_game_ids)]

    # Aggregate by player
    player_stats = (
        recent.groupby("PLAYER_NAME")
        .agg(
            games=("GAME_ID", "nunique"),
            avg_pts=("PTS", "mean"),
            avg_reb=("REB", "mean"),
            avg_ast=("AST", "mean"),
            avg_min=("MIN", "mean"),
        )
        .reset_index()
    )

    # Top players by minutes
    top_players = player_stats[player_stats["avg_min"] >= 15].nlargest(8, "avg_min")

    return [
        {
            "name": row["PLAYER_NAME"],
            "avg_pts": round(row["avg_pts"], 1),
            "avg_reb": round(row["avg_reb"], 1),
            "avg_ast": round(row["avg_ast"], 1),
            "avg_min": round(row["avg_min"], 1),
            "games": int(row["games"]),
        }
        for _, row in top_players.iterrows()
    ]


def predict_player_props(
    player_name: str,
    team: str,
    opponent: str,
    prop_models: Dict[str, dict],
    player_logs: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Predict player props (PTS, REB, AST) using trained models.

    Returns dict with predicted values for each stat.
    """
    if player_logs is None:
        if not PLAYER_GAME_LOGS_CSV.exists():
            return {}
        player_logs = pd.read_csv(PLAYER_GAME_LOGS_CSV)
        player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])

    # Get this player's recent games
    p_logs = player_logs[
        player_logs["PLAYER_NAME"] == player_name
    ].sort_values("GAME_DATE", ascending=False).head(20)

    if len(p_logs) < 5:
        return {"error": f"Not enough data for {player_name}"}

    # Build basic features from recent game logs
    features = {
        "pts_avg_l5": p_logs.head(5)["PTS"].mean(),
        "pts_avg_l10": p_logs.head(10)["PTS"].mean(),
        "reb_avg_l5": p_logs.head(5)["REB"].mean(),
        "reb_avg_l10": p_logs.head(10)["REB"].mean(),
        "ast_avg_l5": p_logs.head(5)["AST"].mean(),
        "ast_avg_l10": p_logs.head(10)["AST"].mean(),
        "min_avg_l5": p_logs.head(5)["MIN"].mean(),
        "min_avg_l10": p_logs.head(10)["MIN"].mean(),
        "pts_std_l10": p_logs.head(10)["PTS"].std(),
    }

    result = {
        "player": player_name,
        "team": team,
        "opponent": opponent,
        "recent_stats": features,
        "predictions": {},
    }

    # Use simple rolling averages as baseline predictions
    # (The full model needs the build_player_features pipeline features)
    result["predictions"]["points"] = round(features["pts_avg_l10"], 1)
    result["predictions"]["rebounds"] = round(features["reb_avg_l10"], 1)
    result["predictions"]["assists"] = round(features["ast_avg_l10"], 1)

    # If we have trained models and full features, use them
    for stat_key, model_data in prop_models.items():
        feature_cols = model_data.get("feature_columns", [])
        # Try to build feature vector if we have all needed columns
        x = pd.DataFrame([features])
        available = [c for c in feature_cols if c in x.columns]
        if len(available) >= len(feature_cols) * 0.5:
            missing = [c for c in feature_cols if c not in x.columns]
            if missing:
                for c in missing:
                    x[c] = 0
            x = x[feature_cols].fillna(0)
            pred = float(model_data["model"].predict(x)[0])
            stat_name = {"pts": "points", "reb": "rebounds", "ast": "assists"}.get(stat_key, stat_key)
            result["predictions"][stat_name] = round(pred, 1)

    return result
