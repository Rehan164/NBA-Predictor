"""
Backtest NBA games against model predictions.

Fetches completed games from ESPN (with betting lines from pickcenter),
runs model predictions, and compares predicted vs actual results.

Shows ML, ATS (spread), and O/U hit rates per day + full week aggregate.

Usage:
    python -m nba_ml.backtest_yesterday              # yesterday only
    python -m nba_ml.backtest_yesterday --week        # last 7 days
    python -m nba_ml.backtest_yesterday --calibrate   # 14-day backtest + save bias corrections
    python -m nba_ml.backtest_yesterday --date=20260220
"""

import json
import sys
import pickle
import platform
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .config import MODELS_DIR, TRAINING_FEATURES_CSV
from .predict import (
    load_model,
    build_matchup_features,
    prepare_features,
    _predict_with_model,
    fetch_json,
    safe_float,
    safe_int,
    american_to_decimal,
)

ET = timezone(timedelta(hours=-5))


# =====================================================================
# FETCH COMPLETED GAMES  (with betting lines from pickcenter)
# =====================================================================

def fetch_pickcenter(event_id: str) -> dict:
    """Fetch spread/total/ML lines from ESPN summary pickcenter."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    data = fetch_json(url)
    if not data:
        return {}

    picks = data.get("pickcenter", [])
    if not picks:
        return {}

    # Use the first provider (usually DraftKings)
    p = picks[0]
    home_odds = p.get("homeTeamOdds", {})
    away_odds = p.get("awayTeamOdds", {})

    return {
        "spread_val": safe_float(p.get("spread")),         # home spread (positive = home underdog)
        "over_under": safe_float(p.get("overUnder")),
        "home_ml": safe_int(home_odds.get("moneyLine"), None),
        "away_ml": safe_int(away_odds.get("moneyLine"), None),
        "spread_details": p.get("details", ""),
    }


def get_games_for_date(date_str: str) -> list:
    """Fetch completed games + betting lines for a given date (YYYYMMDD)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
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
        if status_name != "STATUS_FINAL":
            continue

        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))
        event_id = event["id"]

        # Get betting lines from pickcenter
        lines = fetch_pickcenter(event_id)

        games.append({
            "event_id": event_id,
            "home_team": home["team"]["abbreviation"],
            "away_team": away["team"]["abbreviation"],
            "home_name": home["team"]["displayName"],
            "away_name": away["team"]["displayName"],
            "home_score": home_score,
            "away_score": away_score,
            "total_score": home_score + away_score,
            "home_margin": home_score - away_score,
            "home_win": 1 if home_score > away_score else 0,
            "spread_val": lines.get("spread_val"),
            "over_under": lines.get("over_under"),
            "home_ml": lines.get("home_ml"),
            "away_ml": lines.get("away_ml"),
            "spread_details": lines.get("spread_details", ""),
        })

    return games


# =====================================================================
# RUN PREDICTIONS & COMPARE
# =====================================================================

def backtest_games(games: list, models: dict, feature_df: pd.DataFrame) -> list:
    """Run model predictions on completed games and compare with actuals."""
    results = []

    for game in games:
        home = game["home_team"]
        away = game["away_team"]

        features = build_matchup_features(home, away, feature_df)
        if not features:
            results.append({**game, "error": "Insufficient feature data"})
            continue

        result = {**game}

        # Moneyline prediction
        if "moneyline_model" in models:
            model_data = models["moneyline_model"]
            x = prepare_features(features, model_data["feature_columns"])
            _, home_prob = _predict_with_model(model_data, x)

            predicted_winner = home if home_prob > 0.5 else away
            actual_winner = home if game["home_win"] == 1 else away

            result["ml_home_prob"] = home_prob
            result["ml_predicted_winner"] = predicted_winner
            result["ml_actual_winner"] = actual_winner
            result["ml_correct"] = predicted_winner == actual_winner
            result["ml_confidence"] = max(home_prob, 1 - home_prob) * 100

        # Spread prediction
        if "spread_model" in models:
            model_data = models["spread_model"]
            x = prepare_features(features, model_data["feature_columns"])
            predicted_margin, _ = _predict_with_model(model_data, x)

            result["predicted_margin"] = predicted_margin
            result["actual_margin"] = game["home_margin"]
            result["margin_error"] = abs(predicted_margin - game["home_margin"])

            # ATS check against the actual line
            spread_val = game.get("spread_val")
            if spread_val is not None:
                # spread_val is from home perspective
                # Model edge: model says home wins by predicted_margin, line says spread_val
                model_edge = predicted_margin + spread_val
                if model_edge > 0:
                    pick_home_cover = True
                else:
                    pick_home_cover = False

                actual_home_cover = game["home_margin"] + spread_val > 0
                result["ats_correct"] = pick_home_cover == actual_home_cover
                result["spread_line"] = spread_val
                result["spread_edge"] = model_edge
                result["ats_pick"] = f"{home} {spread_val:+.1f}" if pick_home_cover else f"{away} {-spread_val:+.1f}"

        # Totals prediction
        if "totals_model" in models:
            model_data = models["totals_model"]
            x = prepare_features(features, model_data["feature_columns"])
            predicted_total, _ = _predict_with_model(model_data, x)

            result["predicted_total"] = predicted_total
            result["actual_total"] = game["total_score"]
            result["total_error"] = abs(predicted_total - game["total_score"])

            ou_line = game.get("over_under")
            if ou_line is not None:
                model_over = predicted_total > ou_line
                actual_over = game["total_score"] > ou_line
                result["ou_correct"] = model_over == actual_over
                result["ou_line"] = ou_line
                result["ou_edge"] = predicted_total - ou_line
                result["ou_pick"] = f"OVER {ou_line}" if model_over else f"UNDER {ou_line}"

        results.append(result)

    return results


# =====================================================================
# DISPLAY (single day)
# =====================================================================

def display_day(results: list, date_label: str):
    """Display one day's backtest results."""

    print(f"\n  {'='*78}")
    print(f"  {date_label}")
    print(f"  {'='*78}")

    if not results:
        print("  No completed games.")
        return {}, []

    # Column headers
    print(f"\n  {'Game':<22} {'Score':>9}  {'ML':>4}  {'Spread':>22}  {'O/U':>22}")
    print(f"  {'-'*22} {'-'*9}  {'-'*4}  {'-'*22}  {'-'*22}")

    ml_hits = ml_total = 0
    ats_hits = ats_total = 0
    ou_hits = ou_total = 0
    all_bets = []

    for r in results:
        if "error" in r:
            print(f"  {r['away_team']} @ {r['home_team']:<16} -- Error")
            continue

        home = r["home_team"]
        away = r["away_team"]
        label = f"{away} @ {home}"
        score = f"{r['away_score']}-{r['home_score']}"

        # ML
        ml_str = ""
        if "ml_correct" in r:
            ml_total += 1
            if r["ml_correct"]:
                ml_hits += 1
                ml_str = "YES"
            else:
                ml_str = "NO"

            # Collect for top bets
            all_bets.append({
                "game": label,
                "type": "ML",
                "pick": r["ml_predicted_winner"],
                "confidence": r["ml_confidence"],
                "correct": r["ml_correct"],
                "detail": f"{r['ml_home_prob']*100:.0f}% home",
            })

        # ATS
        ats_str = ""
        if "ats_correct" in r:
            ats_total += 1
            line_str = f"{r['spread_details']}"
            if r["ats_correct"]:
                ats_hits += 1
                ats_str = f"{line_str:<16} YES"
            else:
                ats_str = f"{line_str:<16}  NO"

            all_bets.append({
                "game": label,
                "type": "ATS",
                "pick": r["ats_pick"],
                "confidence": 50 + abs(r["spread_edge"]) * 3,  # rough confidence from edge
                "correct": r["ats_correct"],
                "detail": f"edge {r['spread_edge']:+.1f}",
            })
        elif "spread_val" not in r or r.get("spread_val") is None:
            ats_str = "no line"

        # O/U
        ou_str = ""
        if "ou_correct" in r:
            ou_total += 1
            ou_line = r["ou_line"]
            direction = "O" if r["ou_edge"] > 0 else "U"
            if r["ou_correct"]:
                ou_hits += 1
                ou_str = f"{direction} {ou_line:<15} YES"
            else:
                ou_str = f"{direction} {ou_line:<15}  NO"

            all_bets.append({
                "game": label,
                "type": "O/U",
                "pick": r["ou_pick"],
                "confidence": 50 + abs(r["ou_edge"]) * 2,
                "correct": r["ou_correct"],
                "detail": f"edge {r['ou_edge']:+.1f}",
            })
        elif "over_under" not in r or r.get("over_under") is None:
            ou_str = "no line"

        print(f"  {label:<22} {score:>9}  {ml_str:>4}  {ats_str:>22}  {ou_str:>22}")

    # Day totals
    print(f"  {'-'*78}")

    def fmt_record(hits, total):
        if total == 0:
            return "-- "
        pct = hits / total * 100
        return f"{hits}/{total} ({pct:.0f}%)"

    print(f"  ML: {fmt_record(ml_hits, ml_total):>14}    "
          f"ATS: {fmt_record(ats_hits, ats_total):>14}    "
          f"O/U: {fmt_record(ou_hits, ou_total):>14}")

    day_stats = {
        "ml_hits": ml_hits, "ml_total": ml_total,
        "ats_hits": ats_hits, "ats_total": ats_total,
        "ou_hits": ou_hits, "ou_total": ou_total,
    }

    return day_stats, all_bets


def display_top_bets(all_bets: list, n: int = 6):
    """Show top N bets by confidence and whether they hit."""
    if not all_bets:
        return

    ranked = sorted(all_bets, key=lambda b: b["confidence"], reverse=True)[:n]

    print(f"\n  {'='*78}")
    print(f"  TOP {n} BETS BY CONFIDENCE")
    print(f"  {'='*78}")
    print(f"  {'#':>3}  {'Game':<22} {'Type':<5} {'Pick':<24} {'Conf':>5}  {'Result':>6}")
    print(f"  {'-'*3}  {'-'*22} {'-'*5} {'-'*24} {'-'*5}  {'-'*6}")

    hits = 0
    for i, b in enumerate(ranked, 1):
        result = "HIT" if b["correct"] else "MISS"
        if b["correct"]:
            hits += 1
        print(f"  {i:>3}  {b['game']:<22} {b['type']:<5} {b['pick']:<24} "
              f"{b['confidence']:5.1f}  {result:>6}")

    pct = hits / len(ranked) * 100 if ranked else 0
    print(f"  {'-'*78}")
    print(f"  Top {n} record: {hits}/{len(ranked)} ({pct:.0f}%)")


def display_week_summary(week_stats: list):
    """Show aggregated week summary across multiple days."""
    if not week_stats:
        return

    total_ml_hits = sum(d["ml_hits"] for d in week_stats)
    total_ml = sum(d["ml_total"] for d in week_stats)
    total_ats_hits = sum(d["ats_hits"] for d in week_stats)
    total_ats = sum(d["ats_total"] for d in week_stats)
    total_ou_hits = sum(d["ou_hits"] for d in week_stats)
    total_ou = sum(d["ou_total"] for d in week_stats)

    def fmt(hits, total):
        if total == 0:
            return "--"
        return f"{hits}/{total} ({hits/total*100:.1f}%)"

    print(f"\n  {'='*78}")
    print(f"  WEEK TOTALS ({len(week_stats)} days)")
    print(f"  {'='*78}")
    print(f"    Moneyline:    {fmt(total_ml_hits, total_ml)}")
    print(f"    ATS (Spread): {fmt(total_ats_hits, total_ats)}")
    print(f"    Over/Under:   {fmt(total_ou_hits, total_ou)}")

    combined_hits = total_ml_hits + total_ats_hits + total_ou_hits
    combined_total = total_ml + total_ats + total_ou
    print(f"    Combined:     {fmt(combined_hits, combined_total)}")
    print()


# =====================================================================
# CALIBRATION (bias correction from backtest results)
# =====================================================================

def compute_calibration(all_results: list) -> dict:
    """Compute systematic biases from backtest results."""
    margin_errors = []  # predicted - actual (positive = model predicts too high)
    total_errors = []
    home_preds = []     # model's home win prob
    home_actuals = []   # actual home wins (0/1)

    for r in all_results:
        if "error" in r:
            continue
        if "predicted_margin" in r:
            margin_errors.append(r["predicted_margin"] - r["actual_margin"])
        if "predicted_total" in r:
            total_errors.append(r["predicted_total"] - r["actual_total"])
        if "ml_home_prob" in r:
            home_preds.append(r["ml_home_prob"])
            home_actuals.append(r["home_win"])

    cal = {"sample_size": len(all_results), "computed_at": datetime.now(ET).isoformat()}

    if margin_errors:
        cal["margin_bias"] = round(float(np.mean(margin_errors)), 2)

    if total_errors:
        cal["total_bias"] = round(float(np.mean(total_errors)), 2)

    if home_preds and home_actuals:
        avg_pred_home = np.mean(home_preds)
        avg_actual_home = np.mean(home_actuals)
        # Shift = how much to adjust home_prob (negative = model overrates home)
        cal["ml_home_shift"] = round(float(avg_actual_home - avg_pred_home), 4)

    return cal


def save_calibration(cal: dict):
    """Save calibration to models/calibration.json."""
    cal_path = MODELS_DIR / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\n  Calibration saved to: {cal_path}")
    print(f"    Margin bias:   {cal.get('margin_bias', 0):+.2f} pts (will subtract from predictions)")
    print(f"    Total bias:    {cal.get('total_bias', 0):+.2f} pts (will subtract from predictions)")
    print(f"    ML home shift: {cal.get('ml_home_shift', 0):+.4f} (will add to home_prob)")
    print(f"    Sample size:   {cal.get('sample_size', 0)} games")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print()
    print(f"  {'='*78}")
    print(f"  NBA MODEL BACKTEST")
    print(f"  {'='*78}")

    # Parse args
    target_date = None
    week_mode = False
    calibrate_mode = False
    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            target_date = arg.split("=", 1)[1]
        elif arg == "--week":
            week_mode = True
        elif arg == "--calibrate":
            calibrate_mode = True

    if target_date is None:
        yesterday = datetime.now(ET) - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    # Calibrate mode: override to 14-day window
    if calibrate_mode:
        week_mode = True
        print("  Mode: CALIBRATE (14-day backtest + save bias corrections)")

    # Load models
    print("\n  Loading models...")
    models = {}
    for model_name in ["moneyline_model", "spread_model", "totals_model"]:
        try:
            model_data = load_model(model_name)
            models[model_name] = model_data
            mtype = model_data.get("model_type", "classifier")
            tag = "ensemble" if "ensemble" in mtype else "single"
            print(f"    {model_name}: loaded ({tag})")
        except FileNotFoundError:
            print(f"    {model_name}: NOT FOUND")

    if not models:
        print("\n  No models found. Run 'python -m nba_ml.train_models' first.")
        return

    # Load features
    print("\n  Loading feature data...")
    if not TRAINING_FEATURES_CSV.exists():
        print(f"  Feature file not found: {TRAINING_FEATURES_CSV}")
        return

    feature_df = pd.read_csv(TRAINING_FEATURES_CSV)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    print(f"    {len(feature_df):,} historical games loaded")

    # Build list of dates to process
    if calibrate_mode:
        base = datetime.strptime(target_date, "%Y%m%d")
        dates = [(base - timedelta(days=i)).strftime("%Y%m%d") for i in range(14)]
        dates.reverse()
    elif week_mode:
        base = datetime.strptime(target_date, "%Y%m%d")
        dates = [(base - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
        dates.reverse()
    else:
        dates = [target_date]

    # Process each day
    week_stats = []
    all_week_bets = []
    all_results = []  # raw results for calibration

    for date_str in dates:
        dt = datetime.strptime(date_str, "%Y%m%d")
        date_label = dt.strftime("%A, %B %d, %Y")

        print(f"\n  Fetching games for {date_label}...")
        games = get_games_for_date(date_str)

        if not games:
            print(f"  No completed games.")
            continue

        print(f"    Found {len(games)} completed games")
        results = backtest_games(games, models, feature_df)
        all_results.extend(results)

        day_stats, day_bets = display_day(results, date_label)

        if day_stats:
            week_stats.append(day_stats)
            all_week_bets.extend(day_bets)

    # Top 6 bets (from all days processed)
    display_top_bets(all_week_bets, n=6)

    # Week summary (only if multiple days)
    if len(week_stats) > 1:
        display_week_summary(week_stats)
    elif len(week_stats) == 1:
        print()

    # Save calibration if requested
    if calibrate_mode and all_results:
        cal = compute_calibration(all_results)
        save_calibration(cal)


if __name__ == "__main__":
    main()
