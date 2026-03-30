"""
NBA Picks Orchestrator — Claude-Powered Betting Analysis.

Full pipeline:
  1. Fetch today's games and odds (The-Odds-API + ESPN)
  2. Pull injury reports (ESPN)
  3. Detect injury-driven performance patterns
  4. Run our trained ML models for predictions
  5. Fetch player prop lines
  6. Send everything to Claude for analysis
  7. Iterate until we hit target number of picks

Usage:
    python -m picks.main                    # Full analysis
    python -m picks.main --picks 3          # Target 3 picks
    python -m picks.main --no-props         # Skip player props (saves API quota)
    python -m picks.main --no-odds          # Use ESPN odds only (no Odds API)
    python -m picks.main --rebuild-patterns # Rebuild pattern database
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file from project root
load_dotenv(PROJECT_ROOT / ".env")

from picks.injury_report import get_full_injury_report, get_out_players, format_injury_report
from picks.patterns import load_cached_patterns, find_active_patterns, format_patterns, build_pattern_database
from picks.model_predictions import (
    load_models, load_player_prop_models, load_feature_data,
    predict_game, get_recent_player_stats, predict_player_props,
    format_predictions_for_claude,
)
from picks.odds_api import get_game_odds, get_player_props, get_best_odds, format_prop_summary
from picks.claude_analyst import iterative_analysis


ET = timezone(timedelta(hours=-5))


def fetch_espn_games():
    """Fetch today's games from ESPN (free, no API key needed)."""
    import requests

    today = datetime.now(ET).strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR: Could not fetch ESPN schedule: {e}")
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
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status == "STATUS_FINAL":
            continue

        games.append({
            "event_id": event["id"],
            "home_team": home["team"]["abbreviation"],
            "away_team": away["team"]["abbreviation"],
            "home_name": home["team"]["displayName"],
            "away_name": away["team"]["displayName"],
            "game_time": event["date"],
        })

    return games


def run(args):
    """Main pipeline."""
    print("=" * 70)
    print("  NBA PICKS — Claude-Powered Betting Analysis")
    print(f"  {datetime.now(ET).strftime('%A, %B %d, %Y %I:%M %p ET')}")
    print("=" * 70)

    # ── Step 1: Get today's games ──────────────────────────────────────────
    print("\n[1/6] Fetching today's games...")
    espn_games = fetch_espn_games()
    if not espn_games:
        print("  No games found for today. Exiting.")
        return

    print(f"  Found {len(espn_games)} games:")
    for g in espn_games:
        print(f"    {g['away_name']} @ {g['home_name']}")

    teams_playing = set()
    for g in espn_games:
        teams_playing.add(g["home_team"])
        teams_playing.add(g["away_team"])

    # ── Step 2: Fetch odds ─────────────────────────────────────────────────
    odds_data = []
    player_props_data = {}

    if not args.no_odds:
        print("\n[2/6] Fetching odds from The-Odds-API...")
        try:
            raw_odds = get_game_odds()
            if raw_odds:
                odds_data = get_best_odds(raw_odds)
                print(f"  Got odds for {len(odds_data)} games")

                # Fetch player props if requested
                if not args.no_props:
                    print("  Fetching player props (this uses more API quota)...")
                    for game_odds in raw_odds[:len(espn_games)]:
                        event_id = game_odds["id"]
                        props = get_player_props(event_id)
                        if props:
                            game_key = f"{game_odds['away_team']} @ {game_odds['home_team']}"
                            player_props_data[game_key] = props
                            print(f"    Got props for {len(props)} players in {game_key}")
            else:
                print("  No odds data returned (check API key)")
        except Exception as e:
            print(f"  WARNING: Odds API failed: {e}")
            print("  Continuing with ESPN odds only...")
    else:
        print("\n[2/6] Skipping Odds API (--no-odds)")

    # ── Step 3: Fetch injuries ─────────────────────────────────────────────
    print("\n[3/6] Fetching injury reports...")
    injury_report = get_full_injury_report(team_filter=list(teams_playing))
    out_players = get_out_players(injury_report)
    injury_text = format_injury_report(injury_report, teams=list(teams_playing))
    print(injury_text)

    # ── Step 4: Load patterns ──────────────────────────────────────────────
    print("\n[4/6] Loading injury-driven patterns...")
    if args.rebuild_patterns:
        all_patterns = build_pattern_database()
    else:
        all_patterns = load_cached_patterns()

    active_patterns = find_active_patterns(all_patterns, out_players)
    patterns_text = format_patterns(active_patterns)
    if active_patterns:
        print(f"  {len(active_patterns)} active patterns for today's injuries")
    else:
        print("  No active injury patterns for today's games")

    # ── Step 5: Run our ML models ──────────────────────────────────────────
    print("\n[5/6] Running model predictions...")
    models = load_models()
    feature_df = load_feature_data()

    if not models:
        print("  ERROR: No trained models found. Run training first:")
        print("    python -m nba_ml.train_models")
        print("    python -m nba_ml.train_player_props")
        return

    game_predictions = []
    for game in espn_games:
        pred = predict_game(
            game["home_team"],
            game["away_team"],
            models,
            feature_df,
            injuries=out_players,
        )
        game_predictions.append(pred)

    predictions_text = format_predictions_for_claude(game_predictions, odds_data)
    print(predictions_text)

    # Player stats for context
    player_context_lines = []
    prop_models = load_player_prop_models()

    for game in espn_games:
        for team_key in ["home_team", "away_team"]:
            team = game[team_key]
            stats = get_recent_player_stats(team)
            if stats:
                player_context_lines.append(f"\n{team} key players (last 10 games):")
                for p in stats:
                    player_context_lines.append(
                        f"  {p['name']}: {p['avg_pts']} PTS / {p['avg_reb']} REB / "
                        f"{p['avg_ast']} AST in {p['avg_min']} MIN ({p['games']} games)"
                    )

    player_context = "\n".join(player_context_lines)

    # Format player props from Odds API
    props_text = ""
    if player_props_data:
        props_parts = []
        for game_key, props in player_props_data.items():
            props_parts.append(f"\n{game_key}:")
            props_parts.append(format_prop_summary(props))
        props_text = "\n".join(props_parts)

    # Combine player context and props
    full_player_text = ""
    if player_context:
        full_player_text += "PLAYER RECENT STATS:\n" + player_context
    if props_text:
        full_player_text += "\n\nPLAYER PROP LINES (from sportsbooks):\n" + props_text

    # ── Step 6: Claude Analysis ────────────────────────────────────────────
    print(f"\n[6/6] Sending to Claude for analysis (target: {args.picks} picks)...")

    try:
        analysis, picks = iterative_analysis(
            model_predictions=predictions_text,
            injury_report=injury_text,
            active_patterns=patterns_text,
            player_props_summary=full_player_text,
            odds_summary="",
            target_picks=args.picks,
            max_iterations=args.max_iterations,
        )

        # Print the full analysis
        print("\n" + "=" * 70)
        print("  CLAUDE'S ANALYSIS")
        print("=" * 70)
        print(analysis)

        # Summary
        print("\n" + "=" * 70)
        print(f"  FINAL SLATE: {len(picks)} picks")
        print("=" * 70)
        for i, pick in enumerate(picks, 1):
            print(f"  {i}. {pick['description']}")

        if len(picks) < args.picks:
            print(f"\n  Note: Claude selected {len(picks)}/{args.picks} picks.")
            print("  Claude may have found insufficient value to fill the slate.")

    except ImportError:
        print("\n  ERROR: anthropic package not installed.")
        print("  Run: pip install anthropic")
        print("\n  Without Claude, here are the raw model predictions:")
        print(predictions_text)

    except ValueError as e:
        print(f"\n  ERROR: {e}")
        print("\n  Without Claude, here are the raw model predictions:")
        print(predictions_text)

    except Exception as e:
        print(f"\n  ERROR: Claude analysis failed: {e}")
        print("\n  Falling back to raw model predictions:")
        print(predictions_text)


def main():
    parser = argparse.ArgumentParser(description="NBA Picks — Claude-Powered Betting Analysis")
    parser.add_argument("--picks", type=int, default=5, help="Target number of picks (default: 5)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max Claude iterations (default: 3)")
    parser.add_argument("--no-props", action="store_true", help="Skip player props (saves Odds API quota)")
    parser.add_argument("--no-odds", action="store_true", help="Skip The-Odds-API (use ESPN odds only)")
    parser.add_argument("--rebuild-patterns", action="store_true", help="Rebuild pattern database from scratch")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
