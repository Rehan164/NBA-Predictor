"""
NBA ML Prediction Script

Generates predictions for today's games using trained models.
Pulls live odds from ESPN and shows spread/total/ML picks with payouts.

Usage:
    python -m nba_ml.predict
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .config import (
    MODELS_DIR,
    TRAINING_FEATURES_CSV,
    MIN_CONFIDENCE_SPREAD,
    MIN_CONFIDENCE_TOTAL,
    MIN_CONFIDENCE_ML,
)

# Eastern timezone for NBA schedule
ET = timezone(timedelta(hours=-5))


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


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1.0
    return (100 / abs(odds)) + 1.0


def calc_single(wager: float, odds: int):
    dec = american_to_decimal(odds)
    payout = round(wager * dec, 2)
    profit = round(payout - wager, 2)
    return profit, payout


def calc_parlay(wager: float, legs: list):
    combined = 1.0
    for leg in legs:
        combined *= american_to_decimal(leg["odds"])
    payout = round(wager * combined, 2)
    profit = round(payout - wager, 2)
    return round(combined, 3), profit, payout


def get_wager() -> float:
    while True:
        raw = input("\n  How much would you like to wager? $").strip().replace(",", "")
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Please enter a valid dollar amount.")


def confidence_label(conf: float) -> str:
    if conf >= 70:
        return "STRONG"
    if conf >= 60:
        return "MODERATE"
    if conf >= 55:
        return "LEAN"
    return "LOW"


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

        # Skip completed games
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
            "spread_val": odds.get("spread"),
            "over_under": odds.get("overUnder"),
            "home_ml": safe_int(ml.get("home", {}).get("close", {}).get("odds"), None),
            "away_ml": safe_int(ml.get("away", {}).get("close", {}).get("odds"), None),
            "home_spread_odds": safe_int(ps.get("home", {}).get("close", {}).get("odds")),
            "away_spread_odds": safe_int(ps.get("away", {}).get("close", {}).get("odds")),
            "over_odds": safe_int(total.get("over", {}).get("close", {}).get("odds")),
            "under_odds": safe_int(total.get("under", {}).get("close", {}).get("odds")),
        })

    return games


def format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        et_time = dt.astimezone(ET)
        import platform
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

    # Home team features from their last home game
    home_as_home = feature_df[feature_df["home_team"] == home_team].sort_values("date").tail(1)
    if not home_as_home.empty:
        for col in home_as_home.columns:
            if col.startswith("home_") and col not in ["home_team", "home_score", "home_win", "home_margin"]:
                features[col] = home_as_home[col].values[0]

    # Away team features from their last away game
    away_as_away = feature_df[feature_df["away_team"] == away_team].sort_values("date").tail(1)
    if not away_as_away.empty:
        for col in away_as_away.columns:
            if col.startswith("away_") and col not in ["away_team", "away_score"]:
                features[col] = away_as_away[col].values[0]

    # Matchup-level features from any recent game with both teams' data
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

    # Travel features
    from .build_features import get_travel_distance, get_timezone_diff
    features["travel_distance"] = get_travel_distance(away_team, home_team)
    features["timezone_diff"] = get_timezone_diff(away_team, home_team)
    features["away_travel_fatigue"] = features.get("travel_distance", 0) * features.get("away_b2b", 0)

    return features


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
        feature_cols = model_data["feature_columns"]

        X = pd.DataFrame([features])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_cols].fillna(0)

        prob = model.predict_proba(X)[0]

        if model_name == "moneyline_model":
            home_prob = prob[1]
            result["ml_home_prob"] = home_prob
            result["ml_away_prob"] = 1 - home_prob

            if home_prob > 0.5:
                pick_team = home_team
                conf = home_prob
                odds = game.get("home_ml") or -110
            else:
                pick_team = away_team
                conf = 1 - home_prob
                odds = game.get("away_ml") or -110

            result["picks"].append({
                "bet_type": "Moneyline",
                "pick": pick_team,
                "confidence": round(conf * 100, 1),
                "odds": odds,
            })

        elif model_name == "spread_model":
            home_cover_prob = prob[1]
            sv = game.get("spread_val")

            if home_cover_prob > 0.5:
                # Take home side
                if sv is not None:
                    pick_label = f"{home_team} {sv:+.1f}"
                else:
                    pick_label = f"{home_team} (spread)"
                odds = game.get("home_spread_odds", -110)
                conf = home_cover_prob
            else:
                # Take away side — flip the spread sign
                if sv is not None:
                    pick_label = f"{away_team} {-sv:+.1f}"
                else:
                    pick_label = f"{away_team} (spread)"
                odds = game.get("away_spread_odds", -110)
                conf = 1 - home_cover_prob

            result["picks"].append({
                "bet_type": "Spread",
                "pick": pick_label,
                "confidence": round(conf * 100, 1),
                "odds": odds,
            })

        elif model_name == "totals_model":
            over_prob = prob[1]
            ou_line = game.get("over_under")

            if over_prob > 0.5:
                ou_str = f"OVER {ou_line}" if ou_line else "OVER"
                odds = game.get("over_odds", -110)
                conf = over_prob
            else:
                ou_str = f"UNDER {ou_line}" if ou_line else "UNDER"
                odds = game.get("under_odds", -110)
                conf = 1 - over_prob

            result["picks"].append({
                "bet_type": "O/U",
                "pick": ou_str,
                "confidence": round(conf * 100, 1),
                "odds": odds,
            })

    # Sort picks by confidence
    result["picks"].sort(key=lambda p: p["confidence"], reverse=True)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def display_game_predictions(pred: dict):
    """Display predictions for a single game."""
    if "error" in pred:
        print(f"\n  {pred['error']}")
        return

    ht, at = pred["home_team"], pred["away_team"]
    time_str = format_time(pred["game_time"]) if pred.get("game_time") else ""

    print(f"\n  {at} @ {ht}  {time_str}")
    print("  " + "-" * 54)

    # Show current lines
    spread = pred.get("spread_details") or "TBD"
    ou = pred.get("over_under") or "TBD"
    hml = pred.get("home_ml")
    aml = pred.get("away_ml")
    hml_str = f"{hml:+d}" if hml else "-"
    aml_str = f"{aml:+d}" if aml else "-"

    print(f"  Lines:  Spread: {spread}  |  O/U: {ou}  |  ML: {ht} {hml_str} / {at} {aml_str}")
    print()

    for pick in pred["picks"]:
        odds_str = f"{pick['odds']:+d}" if pick["odds"] else "-110"
        label = confidence_label(pick["confidence"])
        marker = " <<" if pick["confidence"] >= 57 else ""
        print(f"    {pick['bet_type']:10}  {pick['pick']:<28} Conf: {pick['confidence']:5.1f}%  [{label:8}]  ({odds_str}){marker}")


def display_single(pick: dict, wager: float):
    """Display a single bet with payout."""
    odds = pick["odds"] or -110
    profit, payout = calc_single(wager, odds)
    odds_str = f"{odds:+d}"

    print()
    print("  " + "=" * 54)
    print(f"  TOP PICK: {pick['pick']}  ({pick['bet_type']})")
    print(f"  Confidence: {pick['confidence']:.0f}%  [{confidence_label(pick['confidence'])}]")
    print("  " + "=" * 54)
    print()
    print(f"  Wager:            ${wager:,.2f}  at  {odds_str}")
    print(f"  Potential Profit: ${profit:,.2f}")
    print(f"  Total Payout:     ${payout:,.2f}")
    print("  " + "=" * 54)


def display_parlay(legs: list, wager: float):
    """Display a parlay with payout."""
    combined, profit, payout = calc_parlay(wager, legs)

    print()
    print("  " + "=" * 54)
    print(f"  PARLAY: {len(legs)} Legs")
    print("  " + "=" * 54)
    print()
    for i, leg in enumerate(legs, 1):
        odds_str = f"{leg['odds']:+d}" if leg["odds"] else "-110"
        print(f"  Leg {i}: {leg['pick']:<28} ({leg['bet_type']})  "
              f"Conf: {leg['confidence']:.0f}%  Odds: {odds_str}")
    print()
    print(f"  Combined Decimal Odds: {combined:.3f}")
    print(f"  Wager:                 ${wager:,.2f}")
    print(f"  Potential Profit:      ${profit:,.2f}")
    print(f"  Total Payout:          ${payout:,.2f}")
    print()
    print(f"  All {len(legs)} legs must hit to win.")
    print("  " + "=" * 54)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Main prediction routine."""
    print()
    print("  " + "=" * 54)
    print("  NBA ML PREDICTIONS")
    print("  " + "=" * 54)
    import platform
    fmt = '%B %d, %Y  %#I:%M %p ET' if platform.system() == "Windows" else '%B %d, %Y  %-I:%M %p ET'
    print(f"  Date: {datetime.now(ET).strftime(fmt)}")

    # ── Load models ──
    print("\n  Loading models...")
    models = {}
    for model_name in ["moneyline_model", "spread_model", "totals_model"]:
        try:
            models[model_name] = load_model(model_name)
            metrics = models[model_name].get("metrics", {})
            acc = metrics.get("accuracy", 0)
            print(f"    {model_name}: loaded (accuracy: {acc:.1%})")
        except FileNotFoundError:
            print(f"    {model_name}: NOT FOUND")

    if not models:
        print("\n  No models found. Run 'python -m nba_ml.train_models' first.")
        return

    # ── Load features ──
    print("\n  Loading feature data...")
    if not TRAINING_FEATURES_CSV.exists():
        print(f"  Feature file not found: {TRAINING_FEATURES_CSV}")
        print("  Run 'python -m nba_ml.build_features' first.")
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

    # ── Generate predictions ──
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

    # ── Display all predictions ──
    print("\n  " + "=" * 54)
    print("  GAME PREDICTIONS")
    print("  " + "=" * 54)

    for pred in all_predictions:
        display_game_predictions(pred)

    # ── Recommended bets (high confidence) ──
    strong_picks = [p for p in all_picks if p["confidence"] >= 57]
    strong_picks.sort(key=lambda p: p["confidence"], reverse=True)

    print("\n  " + "=" * 54)
    print(f"  RECOMMENDED BETS ({len(strong_picks)} picks above 57% confidence)")
    print("  " + "=" * 54)

    if not strong_picks:
        print("\n  No high-confidence picks for today.")
        print("  Consider lowering thresholds or waiting for more games.")
        return

    for i, pick in enumerate(strong_picks, 1):
        odds_str = f"{pick['odds']:+d}" if pick["odds"] else "-110"
        label = confidence_label(pick["confidence"])
        print(f"\n  #{i}  {pick['game_label']}")
        print(f"      {pick['bet_type']:10} {pick['pick']:<28} {pick['confidence']:.0f}% [{label}]  ({odds_str})")

    # ── Interactive betting ──
    print("\n  " + "-" * 54)
    print("  What would you like to do?")
    print("  1) Place a single bet (top pick)")
    print("  2) Build a parlay")
    print("  3) Show payout for a specific pick")
    print("  4) Exit")
    print()

    while True:
        choice = input("  Choice (1-4): ").strip()

        if choice == "1":
            if not strong_picks:
                print("  No recommended picks available.")
                continue
            wager = get_wager()
            display_single(strong_picks[0], wager)

        elif choice == "2":
            # One pick per game — take the highest-confidence pick for each game
            best_per_game = {}
            for p in strong_picks:
                gid = p.get("game_id")
                if gid and (gid not in best_per_game or p["confidence"] > best_per_game[gid]["confidence"]):
                    best_per_game[gid] = p
            parlay_pool = list(best_per_game.values())
            parlay_pool.sort(key=lambda p: p["confidence"], reverse=True)

            if len(parlay_pool) < 2:
                print("  Not enough games with picks for a parlay (need 2+ different games).")
                continue

            print(f"\n  Available legs (best pick per game, max 1 per game):")
            for i, pick in enumerate(parlay_pool, 1):
                odds_str = f"{pick['odds']:+d}" if pick["odds"] else "-110"
                print(f"  {i}) {pick['game_label']}  —  {pick['pick']} ({pick['bet_type']})  "
                      f"Conf: {pick['confidence']:.0f}%  {odds_str}")

            print(f"\n  Enter leg numbers separated by commas (e.g. 1,2,3):")
            while True:
                raw = input("  Legs: ").strip()
                try:
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    if len(indices) >= 2 and all(0 <= i < len(parlay_pool) for i in indices):
                        # Verify no duplicate games
                        selected = [parlay_pool[i] for i in indices]
                        game_ids = [s["game_id"] for s in selected]
                        if len(game_ids) == len(set(game_ids)):
                            break
                        else:
                            print("  Cannot have multiple legs from the same game.")
                            continue
                except (ValueError, IndexError):
                    pass
                print(f"  Enter at least 2 valid leg numbers (1-{len(parlay_pool)}), comma-separated.")

            wager = get_wager()
            display_parlay(selected, wager)

        elif choice == "3":
            print()
            for i, pick in enumerate(strong_picks, 1):
                odds_str = f"{pick['odds']:+d}" if pick["odds"] else "-110"
                print(f"  {i}) {pick['pick']} ({pick['bet_type']}) {odds_str}")

            while True:
                raw = input(f"\n  Pick # (1-{len(strong_picks)}): ").strip()
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(strong_picks):
                        break
                except ValueError:
                    pass
                print(f"  Enter 1-{len(strong_picks)}.")

            wager = get_wager()
            display_single(strong_picks[idx], wager)

        elif choice == "4":
            print("\n  Good luck!")
            break

        else:
            print("  Enter 1, 2, 3, or 4.")

        print()
        again = input("  Do another? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Good luck!")
            break


if __name__ == "__main__":
    main()
