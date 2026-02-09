"""
NBA ML Prediction Script

Generates predictions for today's games using trained models.
Pulls live odds from ESPN and shows spread/total/ML picks with payouts.

Usage:
    python -m nba_ml.predict
"""

import math
import pickle
import platform
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .config import MODELS_DIR, TRAINING_FEATURES_CSV, DATA_DIR

# Eastern timezone for NBA schedule
ET = timezone(timedelta(hours=-5))

# Player prop models directory
PLAYER_PROPS_DIR = MODELS_DIR / "player_props"


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def safe_int(val, default=-110):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1.0
    return (100 / abs(odds)) + 1.0


def calc_payout(wager: float, odds: int):
    dec = american_to_decimal(odds)
    payout = round(wager * dec, 2)
    profit = round(payout - wager, 2)
    return profit, payout


def edge_to_confidence(edge_points: float) -> float:
    """Convert a point edge into a confidence percentage (50-90 range)."""
    # 0 pts → 50%, 3 pts → 66%, 5 pts → 73%, 10 pts → 84%
    return 50 + 40 * (1 - math.exp(-abs(edge_points) / 5))


def confidence_label(conf: float) -> str:
    if conf >= 75:
        return "STRONG"
    if conf >= 65:
        return "MODERATE"
    if conf >= 57:
        return "LEAN"
    return "LOW"


def format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        et_time = dt.astimezone(ET)
        if platform.system() == "Windows":
            return et_time.strftime("%#I:%M %p ET")
        return et_time.strftime("%-I:%M %p ET")
    except Exception:
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            et_time = dt.astimezone(ET)
            return et_time.strftime("%I:%M %p ET").lstrip("0")
        except Exception:
            return iso_str


# ═════════════════════════════════════════════════════════════════════════════
# ESPN DATA
# ═════════════════════════════════════════════════════════════════════════════

def fetch_json(url: str):
    import requests
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  API error: {e}")
        return None


def get_todays_games_espn():
    """Fetch today's games with odds from ESPN."""
    today = datetime.now(ET).strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}"

    data = fetch_json(url)
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]

        home = away = None
        for c in comp["competitors"]:
            if c["homeAway"] == "home":
                home = c
            else:
                away = c
        if not home or not away:
            continue

        status_name = comp.get("status", {}).get("type", {}).get("name", "")
        if status_name == "STATUS_FINAL":
            continue

        odds_list = comp.get("odds", [])
        odds = odds_list[0] if odds_list else {}

        ml = odds.get("moneyline", {})
        ps = odds.get("pointSpread", {})
        total = odds.get("total", {})

        games.append({
            "event_id": event["id"],
            "home_team": home["team"]["abbreviation"],
            "away_team": away["team"]["abbreviation"],
            "home_name": home["team"]["displayName"],
            "away_name": away["team"]["displayName"],
            "game_time": event["date"],
            "spread_details": odds.get("details"),
            "spread_val": safe_float(odds.get("spread")),
            "over_under": safe_float(odds.get("overUnder")),
            "home_ml": safe_int(ml.get("home", {}).get("close", {}).get("odds"), None),
            "away_ml": safe_int(ml.get("away", {}).get("close", {}).get("odds"), None),
            "home_spread_odds": safe_int(ps.get("home", {}).get("close", {}).get("odds")),
            "away_spread_odds": safe_int(ps.get("away", {}).get("close", {}).get("odds")),
            "over_odds": safe_int(total.get("over", {}).get("close", {}).get("odds")),
            "under_odds": safe_int(total.get("under", {}).get("close", {}).get("odds")),
        })

    return games


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str) -> dict:
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


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

    from .build_features import get_travel_distance, get_timezone_diff
    features["travel_distance"] = get_travel_distance(away_team, home_team)
    features["timezone_diff"] = get_timezone_diff(away_team, home_team)
    features["away_travel_fatigue"] = features.get("travel_distance", 0) * features.get("away_b2b", 0)

    return features


def prepare_features(features: dict, feature_cols: list) -> pd.DataFrame:
    """Build a single-row DataFrame aligned to model's feature columns."""
    x = pd.DataFrame([features])
    for col in feature_cols:
        if col not in x.columns:
            x[col] = np.nan
    return x[feature_cols].fillna(0)


def predict_game(home_team: str, away_team: str, game: dict,
                  models: dict, feature_df: pd.DataFrame) -> dict:
    """Generate predictions for a single game with live odds."""
    features = build_matchup_features(home_team, away_team, feature_df)

    if not features:
        return {"error": f"Insufficient data for {away_team} @ {home_team}"}

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "spread_val": game.get("spread_val"),
        "spread_details": game.get("spread_details"),
        "over_under": game.get("over_under"),
        "home_ml": game.get("home_ml"),
        "away_ml": game.get("away_ml"),
        "home_spread_odds": game.get("home_spread_odds", -110),
        "away_spread_odds": game.get("away_spread_odds", -110),
        "over_odds": game.get("over_odds", -110),
        "under_odds": game.get("under_odds", -110),
        "game_time": game.get("game_time", ""),
        "picks": [],
    }

    for model_name, model_data in models.items():
        model = model_data["model"]
        model_type = model_data.get("model_type", "classifier")
        feature_cols = model_data["feature_columns"]

        x = prepare_features(features, feature_cols)

        if model_name == "moneyline_model":
            # Classification: predict probability of home win
            prob = model.predict_proba(x)[0]
            home_prob = prob[1]

            if home_prob > 0.5:
                pick_team = home_team
                conf = home_prob * 100
                odds = game.get("home_ml") or -110
            else:
                pick_team = away_team
                conf = (1 - home_prob) * 100
                odds = game.get("away_ml") or -110

            result["ml_home_prob"] = home_prob
            result["picks"].append({
                "bet_type": "ML",
                "pick": pick_team,
                "confidence": round(conf, 1),
                "odds": odds,
                "detail": f"{home_prob*100:.0f}% home win prob",
            })

        elif model_name == "spread_model":
            # Regression: predict home margin, compare to spread line
            predicted_margin = float(model.predict(x)[0])
            sv = game.get("spread_val")

            result["predicted_margin"] = predicted_margin

            if sv is not None:
                # edge = predicted_margin + spread_val
                # Positive edge → home covers, negative → away covers
                edge = predicted_margin + sv
                conf = edge_to_confidence(edge)

                if edge > 0:
                    pick_label = f"{home_team} {sv:+.1f}"
                    odds = game.get("home_spread_odds", -110)
                else:
                    pick_label = f"{away_team} {-sv:+.1f}"
                    odds = game.get("away_spread_odds", -110)

                result["picks"].append({
                    "bet_type": "Spread",
                    "pick": pick_label,
                    "confidence": round(conf, 1),
                    "odds": odds,
                    "detail": f"Predicted margin: {predicted_margin:+.1f}, Edge: {edge:+.1f}",
                })
            else:
                # No spread available, just show predicted margin direction
                conf = edge_to_confidence(predicted_margin)
                if predicted_margin > 0:
                    pick_label = f"{home_team} (spread)"
                else:
                    pick_label = f"{away_team} (spread)"

                result["picks"].append({
                    "bet_type": "Spread",
                    "pick": pick_label,
                    "confidence": round(conf, 1),
                    "odds": -110,
                    "detail": f"Predicted margin: {predicted_margin:+.1f}",
                })

        elif model_name == "totals_model":
            # Regression: predict total score, compare to O/U line
            predicted_total = float(model.predict(x)[0])
            ou_line = game.get("over_under")

            result["predicted_total"] = predicted_total

            if ou_line is not None:
                edge = predicted_total - ou_line
                conf = edge_to_confidence(edge)

                if edge > 0:
                    pick_label = f"OVER {ou_line}"
                    odds = game.get("over_odds", -110)
                else:
                    pick_label = f"UNDER {ou_line}"
                    odds = game.get("under_odds", -110)

                result["picks"].append({
                    "bet_type": "O/U",
                    "pick": pick_label,
                    "confidence": round(conf, 1),
                    "odds": odds,
                    "detail": f"Predicted total: {predicted_total:.0f}, Edge: {edge:+.1f}",
                })
            else:
                result["picks"].append({
                    "bet_type": "O/U",
                    "pick": f"Proj {predicted_total:.0f}",
                    "confidence": 50.0,
                    "odds": -110,
                    "detail": f"Predicted total: {predicted_total:.0f} (no line available)",
                })

    result["picks"].sort(key=lambda p: p["confidence"], reverse=True)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# PLAYER PROPS
# ═════════════════════════════════════════════════════════════════════════════

def load_player_prop_models() -> dict:
    """Load player prop models if available."""
    models = {}
    if not PLAYER_PROPS_DIR.exists():
        return models

    for stat in ["pts", "reb", "ast"]:
        path = PLAYER_PROPS_DIR / f"player_{stat}_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[stat] = pickle.load(f)

    return models


def get_recent_players(player_csv: Path, teams: set, n_per_team: int = 8) -> pd.DataFrame:
    """Get top players for given teams based on recent minutes."""
    if not player_csv.exists():
        return pd.DataFrame()

    plr = pd.read_csv(player_csv)
    plr["GAME_DATE"] = pd.to_datetime(plr["GAME_DATE"])

    for col in ["PTS", "REB", "AST", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "TOV", "PLUS_MINUS"]:
        if col in plr.columns:
            plr[col] = pd.to_numeric(plr[col], errors="coerce").fillna(0)

    # Get most recent season's data
    cutoff = plr["GAME_DATE"].max() - timedelta(days=60)
    recent = plr[plr["GAME_DATE"] >= cutoff].copy()

    # Filter to teams playing today
    recent = recent[recent["TEAM_ABBREVIATION"].isin(teams)]

    if recent.empty:
        return pd.DataFrame()

    # Top N players per team by average minutes
    avg_min = recent.groupby(["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]).agg({
        "MIN": "mean", "PTS": "mean", "REB": "mean", "AST": "mean",
        "FGM": "mean", "FGA": "mean", "FG3M": "mean", "FG3A": "mean",
        "FTM": "mean", "FTA": "mean", "TOV": "mean", "PLUS_MINUS": "mean",
        "GAME_DATE": "count",
    }).rename(columns={"GAME_DATE": "games_played"}).reset_index()

    # Only players with 5+ games in the window
    avg_min = avg_min[avg_min["games_played"] >= 5]

    top_players = (
        avg_min.sort_values(["TEAM_ABBREVIATION", "MIN"], ascending=[True, False])
        .groupby("TEAM_ABBREVIATION")
        .head(n_per_team)
    )

    return top_players, recent


def build_player_prop_features(player_row, player_history: pd.DataFrame,
                                opponent: str, is_home: int,
                                opp_def_stats: dict) -> dict:
    """Build features for a single player prop prediction."""
    pid = player_row["PLAYER_ID"]
    p_games = player_history[player_history["PLAYER_ID"] == pid].sort_values("GAME_DATE")

    if len(p_games) < 5:
        return {}

    feat = {
        "is_home": is_home,
        "is_starter": 1 if player_row["MIN"] >= 24 else 0,
    }

    # Rolling averages (from recent data, already pre-aggregated)
    for stat in ["PTS", "REB", "AST", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "TOV", "PLUS_MINUS"]:
        if stat in p_games.columns:
            for w in [3, 5, 10, 20]:
                vals = p_games[stat].tail(w)
                if len(vals) >= min(w, 3):
                    feat[f"{stat}_avg_l{w}"] = vals.mean()
                    if stat in ["PTS", "REB", "AST"]:
                        feat[f"{stat}_std_l{w}"] = vals.std() if len(vals) >= 2 else 0
                        feat[f"{stat}_max_l{w}"] = vals.max()
                        feat[f"{stat}_min_l{w}"] = vals.min()

    # Momentum
    for stat in ["PTS", "REB", "AST"]:
        avg_3 = p_games[stat].tail(3).mean() if len(p_games) >= 3 else None
        avg_10 = p_games[stat].tail(10).mean() if len(p_games) >= 10 else None
        if avg_3 is not None and avg_10 is not None:
            feat[f"{stat}_momentum"] = avg_3 - avg_10

    # Season average
    feat["PTS_season_avg"] = p_games["PTS"].mean()
    feat["REB_season_avg"] = p_games["REB"].mean()
    feat["AST_season_avg"] = p_games["AST"].mean()
    feat["MIN_season_avg"] = p_games["MIN"].mean()

    # Usage proxy
    min_safe = max(player_row["MIN"], 1)
    feat["usage_proxy"] = player_row.get("FGA", 0) / min_safe * 48

    # Rest days (approximate from last game)
    if len(p_games) >= 2:
        last_date = p_games["GAME_DATE"].iloc[-1]
        prev_date = p_games["GAME_DATE"].iloc[-2]
        feat["rest_days"] = min((last_date - prev_date).days, 7)
    else:
        feat["rest_days"] = 2

    # Opponent defensive stats
    for k, v in opp_def_stats.items():
        feat[k] = v

    return feat


def predict_player_props(prop_models: dict, player_csv: Path,
                          games: list, feature_df: pd.DataFrame) -> list:
    """Generate player prop predictions for today's games."""
    if not prop_models or not player_csv.exists():
        return []

    teams_today = set()
    game_matchups = {}
    for g in games:
        teams_today.add(g["home_team"])
        teams_today.add(g["away_team"])
        game_matchups[g["home_team"]] = {"opponent": g["away_team"], "is_home": 1}
        game_matchups[g["away_team"]] = {"opponent": g["home_team"], "is_home": 0}

    result = get_recent_players(player_csv, teams_today)
    if isinstance(result, tuple):
        top_players, recent_data = result
    else:
        return []

    if top_players.empty:
        return []

    # Build simple opponent defensive stats from feature_df
    opp_def_cache = {}
    for team in teams_today:
        team_games = feature_df[
            (feature_df["home_team"] == team) | (feature_df["away_team"] == team)
        ].sort_values("date").tail(10)

        if team_games.empty:
            opp_def_cache[team] = {}
            continue

        # When this team was on defense (opponent scored against them)
        home_mask = team_games["home_team"] == team
        pts_allowed = []
        for _, row in team_games.iterrows():
            if row["home_team"] == team:
                pts_allowed.append(row.get("away_score", 110) if "away_score" in team_games.columns else 110)
            else:
                pts_allowed.append(row.get("home_score", 110) if "home_score" in team_games.columns else 110)

        opp_def_cache[team] = {
            "opp_def_pts_allowed_l10": np.mean(pts_allowed) if pts_allowed else 110,
        }

    predictions = []

    for _, player in top_players.iterrows():
        team = player["TEAM_ABBREVIATION"]
        matchup = game_matchups.get(team)
        if not matchup:
            continue

        opponent = matchup["opponent"]
        is_home = matchup["is_home"]
        opp_def = opp_def_cache.get(opponent, {})

        # Build features from player's recent game history
        p_history = recent_data[recent_data["PLAYER_ID"] == player["PLAYER_ID"]]
        feat = build_player_prop_features(player, p_history, opponent, is_home, opp_def)

        if not feat:
            continue

        for stat, stat_model_data in prop_models.items():
            stat_model = stat_model_data["model"]
            stat_cols = stat_model_data["feature_columns"]

            x = pd.DataFrame([feat])
            for col in stat_cols:
                if col not in x.columns:
                    x[col] = np.nan
            x = x[stat_cols].fillna(0)

            predicted = float(stat_model.predict(x)[0])
            avg_recent = player.get(stat.upper(), 0)

            predictions.append({
                "player_name": player["PLAYER_NAME"],
                "team": team,
                "opponent": opponent,
                "stat": stat.upper(),
                "predicted": round(predicted, 1),
                "avg_recent": round(avg_recent, 1),
                "minutes": round(player["MIN"], 1),
                "is_starter": "Yes" if player["MIN"] >= 24 else "No",
            })

    return predictions


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def display_game(pred: dict):
    """Display predictions for a single game."""
    if "error" in pred:
        print(f"\n  {pred['error']}")
        return

    ht, at = pred["home_team"], pred["away_team"]
    time_str = format_time(pred["game_time"]) if pred.get("game_time") else ""

    print(f"\n  {at} @ {ht}  {time_str}")
    print("  " + "-" * 62)

    # Show current lines
    spread = pred.get("spread_details") or "TBD"
    ou = pred.get("over_under") or "TBD"
    hml = pred.get("home_ml")
    aml = pred.get("away_ml")
    hml_str = f"{hml:+d}" if hml else "-"
    aml_str = f"{aml:+d}" if aml else "-"

    print(f"  Lines:  Spread: {spread}  |  O/U: {ou}  |  ML: {ht} {hml_str} / {at} {aml_str}")

    # Show model outputs
    pm = pred.get("predicted_margin")
    pt = pred.get("predicted_total")
    if pm is not None or pt is not None:
        parts = []
        if pm is not None:
            parts.append(f"Model Margin: {pm:+.1f}")
        if pt is not None:
            parts.append(f"Model Total: {pt:.0f}")
        print(f"  Model:  {' | '.join(parts)}")

    print()
    for pick in pred["picks"]:
        odds_str = f"{pick['odds']:+d}" if pick["odds"] else "-110"
        label = confidence_label(pick["confidence"])
        marker = " <<" if pick["confidence"] >= 60 else ""
        print(f"    {pick['bet_type']:8}  {pick['pick']:<28} "
              f"Conf: {pick['confidence']:5.1f}% [{label:8}]  ({odds_str}){marker}")


def display_top_picks(all_picks: list, wager: float = 100.0):
    """Display top 20 bets ranked by confidence with $100 payouts."""
    print("\n  " + "=" * 72)
    print(f"  TOP BETS — Ranked by Confidence (${wager:.0f} wager)")
    print("  " + "=" * 72)

    if not all_picks:
        print("\n  No picks available for today.")
        return

    # Sort by confidence, take top 20
    ranked = sorted(all_picks, key=lambda p: p["confidence"], reverse=True)[:20]

    print(f"\n  {'#':>3}  {'Game':<18} {'Type':<8} {'Pick':<28} {'Conf':>6}  {'Odds':>6}  {'Profit':>9}  {'Payout':>9}")
    print("  " + "-" * 100)

    for i, pick in enumerate(ranked, 1):
        odds = pick["odds"] or -110
        odds_str = f"{odds:+d}"
        profit, payout = calc_payout(wager, odds)
        label = confidence_label(pick["confidence"])
        marker = " *" if pick["confidence"] >= 60 else ""

        print(f"  {i:>3}  {pick['game_label']:<18} {pick['bet_type']:<8} "
              f"{pick['pick']:<28} {pick['confidence']:5.1f}%  {odds_str:>6}  "
              f"${profit:>8,.2f}  ${payout:>8,.2f}{marker}")

    print("  " + "-" * 100)
    print(f"  * = Recommended (60%+ confidence)")

    # Show recommended parlay (best pick per game, top 3-4)
    best_per_game = {}
    for p in ranked:
        gid = p.get("game_id")
        if gid and (gid not in best_per_game or p["confidence"] > best_per_game[gid]["confidence"]):
            best_per_game[gid] = p

    parlay_legs = sorted(best_per_game.values(), key=lambda p: p["confidence"], reverse=True)
    parlay_legs = [p for p in parlay_legs if p["confidence"] >= 57][:4]

    if len(parlay_legs) >= 2:
        combined = 1.0
        for leg in parlay_legs:
            combined *= american_to_decimal(leg["odds"] or -110)
        parlay_profit = round(wager * combined - wager, 2)
        parlay_payout = round(wager * combined, 2)

        print(f"\n  SUGGESTED PARLAY ({len(parlay_legs)} legs, 1 per game):")
        for i, leg in enumerate(parlay_legs, 1):
            odds_str = f"{leg['odds']:+d}" if leg["odds"] else "-110"
            print(f"    Leg {i}: {leg['game_label']:<18} {leg['pick']:<28} ({odds_str})")
        print(f"\n    Combined Odds: {combined:.2f}x  |  "
              f"${wager:.0f} → ${parlay_payout:,.2f} profit ${parlay_profit:,.2f}")


def display_player_props(props: list):
    """Display player prop predictions."""
    if not props:
        return

    print("\n  " + "=" * 72)
    print("  PLAYER PROP PREDICTIONS")
    print("  " + "=" * 72)

    # Group by team
    by_team = {}
    for p in props:
        by_team.setdefault(p["team"], []).append(p)

    for team, players in sorted(by_team.items()):
        print(f"\n  {team}:")

        # Get unique players sorted by minutes
        seen = set()
        team_players = []
        for p in sorted(players, key=lambda x: -x["minutes"]):
            if p["player_name"] not in seen:
                seen.add(p["player_name"])
                team_players.append(p)

        for player in team_players[:6]:
            name = player["player_name"]
            starter = "S" if player["is_starter"] == "Yes" else "B"
            mins = player["minutes"]

            # Find all stat predictions for this player
            player_props = [p for p in players if p["player_name"] == name]
            stat_strs = []
            for pp in player_props:
                stat_strs.append(f"{pp['stat']}: {pp['predicted']:.1f} (avg {pp['avg_recent']:.1f})")

            print(f"    [{starter}] {name:<22} {mins:.0f} min  |  {', '.join(stat_strs)}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  " + "=" * 72)
    print("  NBA ML PREDICTIONS")
    print("  " + "=" * 72)
    fmt = '%B %d, %Y  %#I:%M %p ET' if platform.system() == "Windows" else '%B %d, %Y  %-I:%M %p ET'
    print(f"  {datetime.now(ET).strftime(fmt)}")

    # ── Load team models ──
    print("\n  Loading models...")
    models = {}
    for model_name in ["moneyline_model", "spread_model", "totals_model"]:
        try:
            model_data = load_model(model_name)
            models[model_name] = model_data
            mtype = model_data.get("model_type", "classifier")
            metrics = model_data.get("metrics", {})

            if mtype == "classifier":
                acc = metrics.get("accuracy", 0)
                print(f"    {model_name}: loaded ({mtype}, accuracy: {acc:.1%})")
            else:
                mae = metrics.get("mae", 0)
                dir_acc = metrics.get("direction_accuracy", 0)
                print(f"    {model_name}: loaded ({mtype}, MAE: {mae:.1f} pts, direction: {dir_acc:.1%})")
        except FileNotFoundError:
            print(f"    {model_name}: NOT FOUND")

    if not models:
        print("\n  No models found. Run 'python -m nba_ml.train_models' first.")
        return

    # ── Load player prop models ──
    prop_models = load_player_prop_models()
    if prop_models:
        print(f"    Player props: {', '.join(prop_models.keys())} models loaded")
    else:
        print("    Player props: not available (run train_player_props)")

    # ── Load features ──
    print("\n  Loading feature data...")
    if not TRAINING_FEATURES_CSV.exists():
        print(f"  Feature file not found: {TRAINING_FEATURES_CSV}")
        return

    feature_df = pd.read_csv(TRAINING_FEATURES_CSV)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    print(f"    {len(feature_df):,} historical games loaded")

    # ── Fetch today's games ──
    print("\n  Fetching today's games from ESPN...")
    games = get_todays_games_espn()

    if not games:
        print("  No games found for today.")
        return

    print(f"    Found {len(games)} games")

    # ── Generate team predictions ──
    all_predictions = []
    all_picks = []

    for game in games:
        pred = predict_game(
            game["home_team"], game["away_team"],
            game, models, feature_df
        )
        all_predictions.append(pred)

        if "error" not in pred:
            for pick in pred["picks"]:
                pick["game_label"] = f"{pred['away_team']} @ {pred['home_team']}"
                pick["game_id"] = game.get("event_id")
                all_picks.append(pick)

    # ── Display game-by-game predictions ──
    print("\n  " + "=" * 72)
    print("  GAME PREDICTIONS")
    print("  " + "=" * 72)

    for pred in all_predictions:
        display_game(pred)

    # ── Display top 20 bets ──
    display_top_picks(all_picks)

    # ── Player props ──
    from .config import PLAYER_GAME_LOGS_CSV
    player_props = predict_player_props(
        prop_models, PLAYER_GAME_LOGS_CSV, games, feature_df
    )
    display_player_props(player_props)

    print()


if __name__ == "__main__":
    main()
