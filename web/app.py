"""
NBA Predictor Web Dashboard.

Usage:
    python -m web.app
    Open http://localhost:5000
"""

import sys
import threading
import unicodedata
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from flask import Flask, jsonify, render_template, request
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _norm(name: str) -> str:
    """Lowercase + strip diacritics so 'Jokić' == 'Jokic'."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii").lower()

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

app = Flask(__name__)
ET = timezone(timedelta(hours=-5))

# ── ESPN abbr → NBA API/CSV abbr (they differ for several teams) ──────────
ESPN_TO_CSV_ABBR = {
    "SA":   "SAS",
    "GS":   "GSW",
    "NO":   "NOP",
    "NY":   "NYK",
    "OC":   "OKC",
    "UTAH": "UTA",
    "BK":   "BKN",
    "UTA":  "UTA",   # sometimes ESPN sends full abbr too
}

def _csv_team(espn_abbr: str) -> str:
    """Convert ESPN team abbreviation to the one used in player_game_logs CSV."""
    return ESPN_TO_CSV_ABBR.get(espn_abbr, espn_abbr)


# ── ESPN team ID map ──────────────────────────────────────────────────────
ESPN_TEAM_IDS = {
    "ATL": 1,  "BOS": 2,  "BKN": 17, "CHA": 30, "CHI": 4,
    "CLE": 5,  "DAL": 6,  "DEN": 7,  "DET": 8,  "GSW": 9,
    "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29,
    "MIA": 14, "MIL": 15, "MIN": 16, "NOP": 3,  "NYK": 18,
    "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21, "POR": 22,
    "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27,
    "NO": 3,   "GS": 9,   "NY": 18,  "SA": 24,  "UTAH": 26, "BK": 17,
}

# ── Caches ────────────────────────────────────────────────────────────────
_player_logs: Optional[pd.DataFrame] = None
_today_games = None
_today_games_date = None
_props_cache = None
_props_cache_date = None
_injuries_cache = None
_injuries_cache_date = None
_headshots: dict = {}
_headshots_teams: set = set()

# ── Analysis job state ────────────────────────────────────────────────────
_analysis_job = {"status": "idle", "log": [], "result": None, "error": None}
_analysis_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────

def _fetch_team_headshots(team_abbr: str):
    global _headshots, _headshots_teams
    if team_abbr in _headshots_teams:
        return
    _headshots_teams.add(team_abbr)
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return
    import requests as req
    try:
        r = req.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster",
            timeout=8,
        )
        r.raise_for_status()
        for athlete in r.json().get("athletes", []):
            name = athlete.get("displayName", "")
            url = athlete.get("headshot", {}).get("href", "")
            if name and url:
                _headshots[_norm(name)] = url
    except Exception:
        pass


def _get_headshot(name: str) -> str:
    return _headshots.get(_norm(name), "")


def _get_player_logs() -> Optional[pd.DataFrame]:
    global _player_logs
    if _player_logs is None:
        csv_path = PROJECT_ROOT / "data" / "nba_player_game_logs.csv"
        if csv_path.exists():
            print("Loading player game logs...")
            _player_logs = pd.read_csv(csv_path, low_memory=False)
            _player_logs["GAME_DATE"] = pd.to_datetime(_player_logs["GAME_DATE"])
            print(f"  Loaded {len(_player_logs):,} rows")
    return _player_logs


def _fetch_espn_games():
    import requests as req
    today = datetime.now(ET).strftime("%Y%m%d")
    try:
        r = req.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}",
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        home = away = None
        for c in comp["competitors"]:
            (home if c["homeAway"] == "home" else None)
            if c["homeAway"] == "home":
                home = c
            else:
                away = c
        if not home or not away:
            continue

        status_obj = comp.get("status", {})
        # Parse ESPN/DraftKings odds (free, no API key needed)
        market = {}
        odds_list = comp.get("odds", [])
        if odds_list:
            o = odds_list[0]
            ml = o.get("moneyline", {})
            ps = o.get("pointSpread", {})
            tot = o.get("total", {})
            away_fav = o.get("awayTeamOdds", {}).get("favorite", False)

            away_ml = ml.get("away", {}).get("close", {}).get("odds", "")
            home_ml = ml.get("home", {}).get("close", {}).get("odds", "")
            spread_away = ps.get("away", {}).get("close", {}).get("line", "")
            spread_odds_away = ps.get("away", {}).get("close", {}).get("odds", "")
            over_line = tot.get("over", {}).get("close", {}).get("line", "")
            over_odds = tot.get("over", {}).get("close", {}).get("odds", "")
            under_odds = tot.get("under", {}).get("close", {}).get("odds", "")

            market = {
                "away_ml": away_ml,
                "home_ml": home_ml,
                "spread": spread_away,          # from away team's perspective (negative = away favored)
                "spread_odds": spread_odds_away,
                "total": over_line.lstrip("ou") if over_line else "",
                "over_odds": over_odds,
                "under_odds": under_odds,
                "away_favorite": away_fav,
            }

        games.append({
            "event_id": event["id"],
            "home_team": home["team"]["abbreviation"],
            "away_team": away["team"]["abbreviation"],
            "home_name": home["team"]["displayName"],
            "away_name": away["team"]["displayName"],
            "home_logo": home["team"].get("logo", ""),
            "away_logo": away["team"].get("logo", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
            "game_time": event["date"],
            "status": status_obj.get("type", {}).get("name", ""),
            "status_display": status_obj.get("type", {}).get("shortDetail", ""),
            "market": market,
        })
    return games


def _get_today_games():
    global _today_games, _today_games_date
    today = datetime.now(ET).date().isoformat()
    if _today_games_date == today and _today_games is not None:
        return _today_games
    _today_games = _fetch_espn_games()
    _today_games_date = today
    return _today_games


def _get_injuries():
    """Fetch and cache today's injury report as flat name→record dict."""
    global _injuries_cache, _injuries_cache_date
    today = datetime.now(ET).date().isoformat()
    if _injuries_cache_date == today and _injuries_cache is not None:
        return _injuries_cache

    games = _get_today_games()
    teams = {g["home_team"] for g in games} | {g["away_team"] for g in games}
    flat = {}
    try:
        from picks.injury_report import get_full_injury_report
        report = get_full_injury_report(team_filter=list(teams))
        for team_data in report.values():
            for inj in team_data.get("all_injuries", []):
                flat[_norm(inj["name"])] = {
                    "status": inj["status"],
                    "injury": inj.get("injury", ""),
                    "is_out": inj.get("is_out", False),
                    "is_doubtful": inj.get("is_doubtful", False),
                    "is_questionable": inj.get("is_questionable", False),
                }
    except Exception:
        pass

    _injuries_cache = flat
    _injuries_cache_date = today
    return flat


def _calc_hot_streak(logs_df: pd.DataFrame, player_name: str) -> dict:
    """
    Returns {pts, reb, ast} booleans — True if last-5 avg is 20%+ above last-20 avg.
    Also returns per-stat streak counts (consecutive games above avg).
    """
    p = logs_df[logs_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE", ascending=False)
    result = {"pts": False, "reb": False, "ast": False, "details": {}}
    if len(p) < 6:
        return result

    l5 = p.head(5)
    l20 = p.head(20)

    for stat, col in [("pts", "PTS"), ("reb", "REB"), ("ast", "AST")]:
        avg5 = float(l5[col].mean())
        avg20 = float(l20[col].mean())
        if avg20 > 0 and avg5 >= avg20 * 1.20:
            result[stat] = True
            result["details"][stat] = {"l5": round(avg5, 1), "l20": round(avg20, 1)}

    return result


def _estimate_prob(pick_desc: str) -> float:
    u = pick_desc.upper()
    if "HIGH" in u:
        return 0.65
    if "MODERATE" in u or "MEDIUM" in u:
        return 0.58
    if "LOW" in u:
        return 0.52
    return 0.55


def _prob_to_american(p: float) -> int:
    if p <= 0 or p >= 1:
        return 0
    if p >= 0.5:
        return int(round(-100 * p / (1 - p) / 5) * 5)
    return int(round(100 * (1 - p) / p / 5) * 5)


# ── Analysis background job ───────────────────────────────────────────────

def _run_analysis_bg(target_picks: int):
    global _analysis_job
    log = _analysis_job["log"]

    def emit(msg):
        log.append(msg)

    try:
        emit("Fetching today's games...")
        games = _get_today_games()
        if not games:
            raise RuntimeError("No games found for today.")
        emit(f"Found {len(games)} games: " + ", ".join(f"{g['away_team']}@{g['home_team']}" for g in games))

        teams = {g["home_team"] for g in games} | {g["away_team"] for g in games}

        # Odds
        odds_data = []
        raw_odds = []
        player_props_data = {}
        emit("Fetching odds...")
        try:
            from picks.odds_api import get_game_odds, get_player_props, get_best_odds, format_prop_summary
            raw_odds = get_game_odds() or []
            odds_data = get_best_odds(raw_odds) if raw_odds else []
            emit(f"Got odds for {len(odds_data)} games")

            emit("Fetching player props...")
            for go in raw_odds[:len(games)]:
                props = get_player_props(go["id"])
                if props:
                    key = f"{go['away_team']} @ {go['home_team']}"
                    player_props_data[key] = props
                    emit(f"  {key}: {len(props)} players")
        except Exception as e:
            emit(f"Odds unavailable: {e}")

        # Injuries
        emit("Fetching injury report...")
        try:
            from picks.injury_report import get_full_injury_report, get_out_players, format_injury_report
            injury_report = get_full_injury_report(team_filter=list(teams))
            out_players = get_out_players(injury_report)
            injury_text = format_injury_report(injury_report, teams=list(teams))
            total_out = sum(len(v) for v in out_players.values())
            emit(f"  {total_out} players OUT")
        except Exception as e:
            injury_report = {}
            out_players = {}
            injury_text = "No injury data available."
            emit(f"Injuries unavailable: {e}")

        # Patterns
        emit("Loading injury patterns...")
        try:
            from picks.patterns import load_cached_patterns, find_active_patterns, format_patterns
            all_patterns = load_cached_patterns()
            active_patterns = find_active_patterns(all_patterns, out_players)
            patterns_text = format_patterns(active_patterns)
            emit(f"  {len(active_patterns)} active patterns")
        except Exception as e:
            patterns_text = ""
            emit(f"Patterns unavailable: {e}")

        # ML models
        emit("Running ML model predictions...")
        try:
            from picks.model_predictions import (
                load_models, load_feature_data, predict_game,
                get_recent_player_stats, format_predictions_for_claude,
            )
            models = load_models()
            feature_df = load_feature_data()
            game_predictions = []
            for g in games:
                pred = predict_game(g["home_team"], g["away_team"], models, feature_df, injuries=out_players)
                game_predictions.append(pred)
            predictions_text = format_predictions_for_claude(game_predictions, odds_data)
            emit("  Models ran successfully")
        except Exception as e:
            predictions_text = "Model predictions unavailable."
            emit(f"Models error: {e}")
            game_predictions = []

        # Player context
        player_context_lines = []
        try:
            from picks.model_predictions import get_recent_player_stats
            for g in games:
                for tk in ["home_team", "away_team"]:
                    team = g[tk]
                    stats = get_recent_player_stats(team)
                    if stats:
                        player_context_lines.append(f"\n{team} recent:")
                        for p in stats:
                            player_context_lines.append(
                                f"  {p['name']}: {p['avg_pts']} PTS / {p['avg_reb']} REB / {p['avg_ast']} AST"
                            )
        except Exception:
            pass

        props_text = ""
        if player_props_data:
            from picks.odds_api import format_prop_summary
            parts = []
            for gk, props in player_props_data.items():
                parts.append(f"\n{gk}:")
                parts.append(format_prop_summary(props))
            props_text = "\n".join(parts)

        full_player_text = ""
        if player_context_lines:
            full_player_text += "PLAYER RECENT STATS:\n" + "\n".join(player_context_lines)
        if props_text:
            full_player_text += "\n\nPLAYER PROP LINES:\n" + props_text

        # Claude
        emit(f"Sending to Claude (target: {target_picks} picks)...")
        from picks.claude_analyst import iterative_analysis
        analysis, picks = iterative_analysis(
            model_predictions=predictions_text,
            injury_report=injury_text,
            active_patterns=patterns_text,
            player_props_summary=full_player_text,
            odds_summary="",
            target_picks=target_picks,
            max_iterations=3,
        )
        emit(f"Claude selected {len(picks)} picks")

        # Build structured picks with probability estimates
        structured_picks = []
        for pick in picks:
            desc = pick["description"]
            prob = _estimate_prob(desc)
            structured_picks.append({
                "description": desc,
                "probability": round(prob * 100, 1),
                "odds": _prob_to_american(prob),
            })

        # Parlay legs
        parlay_legs = []
        running_prob = 1.0
        for i, p in enumerate(structured_picks, 1):
            running_prob *= p["probability"] / 100
            parlay_legs.append({
                "legs": i,
                "probability": round(running_prob * 100, 1),
                "odds": _prob_to_american(running_prob),
            })

        _analysis_job["result"] = {
            "analysis": analysis,
            "picks": structured_picks,
            "parlay_legs": parlay_legs,
        }
        _analysis_job["status"] = "done"
        emit("Done.")

    except Exception as e:
        import traceback
        _analysis_job["error"] = str(e)
        _analysis_job["log"].append(f"ERROR: {e}")
        _analysis_job["status"] = "error"


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


@app.route("/api/games")
def api_games():
    games = _get_today_games()
    if not games:
        return jsonify([])

    result = []
    try:
        from picks.model_predictions import load_models, load_feature_data, predict_game
        models = load_models()
        feature_df = load_feature_data()
        for g in games:
            try:
                pred = predict_game(g["home_team"], g["away_team"], models, feature_df)
                predictions = pred.get("predictions", {})
            except Exception:
                predictions = {}
            result.append({**g, "predictions": predictions})
    except Exception as e:
        for g in games:
            result.append({**g, "predictions": {}})

    return jsonify(result)


@app.route("/api/players")
def api_players():
    games = _get_today_games()
    if not games:
        return jsonify([])

    logs = _get_player_logs()
    if logs is None:
        return jsonify([])

    team_to_game = {}
    for g in games:
        team_to_game[g["home_team"]] = g
        team_to_game[g["away_team"]] = g

    # Fetch headshots in parallel
    threads = [threading.Thread(target=_fetch_team_headshots, args=(t,)) for t in team_to_game]
    for t in threads: t.start()
    for t in threads: t.join()

    # Fetch injuries
    injuries = _get_injuries()

    players = []
    for team in sorted(team_to_game.keys()):
        game = team_to_game[team]
        opponent = game["away_team"] if team == game["home_team"] else game["home_team"]

        csv_team = _csv_team(team)
        team_logs = logs[logs["TEAM_ABBREVIATION"] == csv_team].sort_values("GAME_DATE", ascending=False)
        recent_ids = team_logs["GAME_ID"].unique()[:10]
        recent = team_logs[team_logs["GAME_ID"].isin(recent_ids)]
        if recent.empty:
            continue

        agg = (
            recent.groupby("PLAYER_NAME")
            .agg(games=("GAME_ID", "nunique"), avg_pts=("PTS", "mean"),
                 avg_reb=("REB", "mean"), avg_ast=("AST", "mean"), avg_min=("MIN", "mean"))
            .reset_index()
        )
        top = agg[agg["avg_min"] >= 15].nlargest(6, "avg_min")

        for _, row in top.iterrows():
            name = row["PLAYER_NAME"]
            inj = injuries.get(name.lower(), {})
            hot = _calc_hot_streak(logs, name)

            players.append({
                "name": name,
                "team": team,
                "opponent": opponent,
                "game_time": game["game_time"],
                "avg_pts": round(float(row["avg_pts"]), 1),
                "avg_reb": round(float(row["avg_reb"]), 1),
                "avg_ast": round(float(row["avg_ast"]), 1),
                "avg_min": round(float(row["avg_min"]), 1),
                "games": int(row["games"]),
                "headshot": _get_headshot(name),
                "injury_status": inj.get("status", ""),
                "injury_desc": inj.get("injury", ""),
                "is_out": inj.get("is_out", False),
                "is_doubtful": inj.get("is_doubtful", False),
                "is_questionable": inj.get("is_questionable", False),
                "hot_pts": hot["pts"],
                "hot_reb": hot["reb"],
                "hot_ast": hot["ast"],
                "hot_details": hot.get("details", {}),
            })

    players.sort(key=lambda x: x["avg_min"], reverse=True)
    return jsonify(players)


@app.route("/api/player/<path:name>/history")
def api_player_history(name):
    stat = request.args.get("stat", "pts").upper()
    line = request.args.get("line", type=float)
    n_games = request.args.get("n", default=20, type=int)
    stat_col = {"PTS": "PTS", "REB": "REB", "AST": "AST"}.get(stat, "PTS")

    logs = _get_player_logs()
    if logs is None:
        return jsonify({"error": "No player data"}), 404

    player_logs = logs[logs["PLAYER_NAME"] == name].sort_values("GAME_DATE", ascending=False)
    if player_logs.empty:
        matches = logs[logs["PLAYER_NAME"].str.contains(name, case=False, na=False)]["PLAYER_NAME"].unique()
        if len(matches):
            player_logs = logs[logs["PLAYER_NAME"] == matches[0]].sort_values("GAME_DATE", ascending=False)
        if player_logs.empty:
            return jsonify({"error": f"Player '{name}' not found"}), 404

    recent = player_logs.head(n_games)
    games_data = []
    for _, row in recent.iterrows():
        value = float(row[stat_col]) if not pd.isna(row[stat_col]) else 0.0
        matchup = str(row.get("MATCHUP", ""))
        team_abbr = str(row.get("TEAM_ABBREVIATION", ""))
        opp = matchup.replace(f"{team_abbr} vs. ", "").replace(f"{team_abbr} @ ", "@").strip()
        if opp == matchup:
            opp = matchup[-3:] if len(matchup) >= 3 else matchup
        games_data.append({
            "date": row["GAME_DATE"].strftime("%m/%d"),
            "opponent": opp,
            "value": round(value, 1),
            "hit": (value >= line) if line is not None else None,
            "home": "vs." in matchup,
        })

    games_data.reverse()
    hits = [g for g in games_data if g["hit"] is True]
    total = [g for g in games_data if g["hit"] is not None]

    return jsonify({
        "player": name,
        "stat": stat,
        "line": line,
        "games": games_data,
        "hit_rate": round(len(hits) / len(total) * 100, 1) if total else None,
        "hit_count": len(hits),
        "total_games": len(total),
        "avg": round(float(recent[stat_col].mean()), 1),
    })


@app.route("/api/game/<event_id>/players")
def api_game_players(event_id):
    """Both teams' rosters with avg + predicted minutes for a game."""
    games = _get_today_games()
    game = next((g for g in games if g["event_id"] == event_id), None)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    logs = _get_player_logs()
    if logs is None:
        return jsonify({"error": "No player data"}), 500

    injuries = _get_injuries()

    # Try to get actual projected starters from ESPN game summary
    espn_starters = _fetch_espn_starters(event_id)

    result = {}
    for side in ("away", "home"):
        team = game[f"{side}_team"]

        # Fetch headshots (may already be cached)
        _fetch_team_headshots(team)

        csv_team = _csv_team(team)
        team_logs = logs[logs["TEAM_ABBREVIATION"] == csv_team].sort_values("GAME_DATE", ascending=False)
        recent_ids = team_logs["GAME_ID"].unique()[:15]
        recent = team_logs[team_logs["GAME_ID"].isin(recent_ids)]
        if recent.empty:
            result[side] = {"team": team, "name": game[f"{side}_name"], "logo": game[f"{side}_logo"], "players": []}
            continue

        agg = (
            recent.groupby("PLAYER_NAME")
            .agg(avg_pts=("PTS", "mean"), avg_reb=("REB", "mean"),
                 avg_ast=("AST", "mean"), avg_min=("MIN", "mean"),
                 games=("GAME_ID", "nunique"))
            .reset_index()
        )

        players = []
        for _, row in agg[agg["avg_min"] >= 6].sort_values("avg_min", ascending=False).iterrows():
            name = row["PLAYER_NAME"]
            inj = injuries.get(name.lower(), {})
            players.append({
                "name": name,
                "avg_pts": round(float(row["avg_pts"]), 1),
                "avg_reb": round(float(row["avg_reb"]), 1),
                "avg_ast": round(float(row["avg_ast"]), 1),
                "avg_min": round(float(row["avg_min"]), 1),
                "headshot": _get_headshot(name),
                "is_out": inj.get("is_out", False),
                "is_doubtful": inj.get("is_doubtful", False),
                "is_questionable": inj.get("is_questionable", False),
                "injury_desc": inj.get("injury", ""),
            })

        # Predicted minutes
        # Rules:
        #  - Only 70% of OUT players' minutes get redistributed (rest = call-ups/deep bench)
        #  - Only rotation players (≥15 min avg) absorb the extra minutes
        #  - Hard cap at 37 min (realistic NBA max for a heavy-minute game)
        MAX_MINS = 37.0
        REDISTRIBUTE_FACTOR = 0.70

        out_mins = sum(p["avg_min"] for p in players if p["is_out"])
        rotation = [p for p in players if not p["is_out"] and p["avg_min"] >= 15]
        rotation_total = sum(p["avg_min"] for p in rotation) or 1
        redistributable = out_mins * REDISTRIBUTE_FACTOR

        for p in players:
            if p["is_out"]:
                p["pred_min"] = 0.0
            elif p["avg_min"] >= 15:
                share = p["avg_min"] / rotation_total
                p["pred_min"] = round(min(MAX_MINS, p["avg_min"] + redistributable * share), 1)
            else:
                p["pred_min"] = p["avg_min"]  # deep bench unchanged

        # Projected starters: ESPN data if available, else top-5 by pred minutes
        if espn_starters.get(team):
            starter_names = set(espn_starters[team])
        else:
            starter_names = {p["name"] for p in sorted((p for p in players if not p["is_out"]), key=lambda x: x["pred_min"], reverse=True)[:5]}

        for p in players:
            p["starter"] = p["name"] in starter_names

        result[side] = {
            "team": team,
            "name": game[f"{side}_name"],
            "logo": game[f"{side}_logo"],
            "players": players,
        }

    return jsonify(result)


def _fetch_espn_starters(event_id: str) -> dict:
    """Try to get announced starters from ESPN game summary. Returns {team_abbr: [names]}."""
    import requests as req
    try:
        r = req.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}",
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        starters = {}
        for roster in data.get("rosters", []):
            abbr = roster.get("team", {}).get("abbreviation", "")
            names = [
                e["athlete"]["displayName"]
                for e in roster.get("roster", [])
                if e.get("starter") and e.get("athlete", {}).get("displayName")
            ]
            if abbr and names:
                starters[abbr] = names
        return starters
    except Exception:
        return {}


@app.route("/api/injuries")
def api_injuries():
    return jsonify(_get_injuries())


@app.route("/api/props")
def api_props():
    global _props_cache, _props_cache_date
    today = datetime.now(ET).date().isoformat()
    if _props_cache_date == today and _props_cache is not None:
        return jsonify(_props_cache)
    try:
        from picks.odds_api import get_game_odds, get_player_props
        raw_odds = get_game_odds() or []
        games = _get_today_games()
        all_props = {}
        for go in raw_odds[:len(games)]:
            props = get_player_props(go["id"])
            if props:
                for player_name, markets in props.items():
                    if player_name not in all_props:
                        all_props[player_name] = {}
                    for mk, data in markets.items():
                        stat = mk.replace("player_", "").lower()
                        if stat in ("points", "rebounds", "assists", "threes"):
                            short = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "tpm"}[stat]
                            over_price = under_price = None
                            for bk in ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]:
                                if bk in data.get("books", {}):
                                    over_price  = data["books"][bk].get("over")
                                    under_price = data["books"][bk].get("under")
                                    break
                            all_props[player_name][short] = {
                                "line":  data.get("line"),
                                "over":  over_price,
                                "under": under_price,
                            }
        _props_cache = all_props
        _props_cache_date = today
        return jsonify(all_props)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh")
def api_refresh():
    global _today_games, _today_games_date, _injuries_cache, _injuries_cache_date
    _today_games = _today_games_date = None
    _injuries_cache = _injuries_cache_date = None
    return jsonify({"ok": True})


@app.route("/api/live/headshots")
def api_live_headshots():
    """Pre-load headshots for all teams playing today and return the full cache."""
    games = _get_today_games()
    for game in games:
        for abbr in [game.get("away_team", ""), game.get("home_team", "")]:
            if abbr:
                _fetch_team_headshots(abbr)
                _fetch_team_headshots(_csv_team(abbr))
    return jsonify(_headshots)


@app.route("/api/live/players")
def api_live_players():
    """Return active player names for autocomplete (players who played this season)."""
    logs = _get_player_logs()
    if logs is None:
        return jsonify([])
    recent = logs[logs["GAME_DATE"] >= "2024-10-01"]["PLAYER_NAME"].dropna().unique()
    return jsonify(sorted(recent.tolist()))


@app.route("/live")
def live_page():
    return render_template("live.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


@app.route("/money")
def money_page():
    return render_template("money.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


@app.route("/api/live/games")
def api_live_games():
    """Today's games with live scores — always fresh, no cache."""
    global _today_games, _today_games_date
    _today_games = None   # force re-fetch for live data
    _today_games_date = None
    return jsonify(_get_today_games())


@app.route("/api/live/boxscore/<event_id>")
def api_live_boxscore(event_id):
    """Live player stats + game clock from ESPN."""
    import requests as req
    try:
        r = req.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}",
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ── Game status ───────────────────────────────────────────────────────
    hc = data.get("header", {}).get("competitions", [{}])[0]
    st = hc.get("status", {})
    period      = st.get("period", 0)
    clock       = st.get("displayClock", "")
    status_type = st.get("type", {}).get("name", "")
    is_live     = status_type == "STATUS_IN_PROGRESS"
    is_final    = status_type == "STATUS_FINAL"

    scores = {}
    for comp in hc.get("competitors", []):
        scores[comp.get("team", {}).get("abbreviation", "")] = comp.get("score", "0")

    # Win probability — use live feed first, fall back to pre-game BPI projection
    home_win_prob = None
    wp_list = data.get("winprobability", [])
    if wp_list:
        try:
            home_win_prob = float(wp_list[-1].get("homeWinPercentage", None))
            if home_win_prob is not None:
                home_win_prob = round(home_win_prob * 100, 1)  # ESPN returns 0-1 scale
        except (TypeError, ValueError):
            home_win_prob = None
    if home_win_prob is None:
        predictor = hc.get("predictor", {})
        if predictor:
            try:
                home_win_prob = float(predictor.get("homeTeam", {}).get("gameProjection", None))
            except (TypeError, ValueError):
                home_win_prob = None

    # ── Pre-load headshots for both teams (synchronous; cached after first call) ──
    for comp in hc.get("competitors", []):
        abbr = comp.get("team", {}).get("abbreviation", "")
        if abbr:
            _fetch_team_headshots(abbr)
            _fetch_team_headshots(_csv_team(abbr))

    # ── Player stats ──────────────────────────────────────────────────────
    players = {}
    for team_block in data.get("boxscore", {}).get("players", []):
        team_abbr = team_block.get("team", {}).get("abbreviation", "")
        for grp in team_block.get("statistics", []):
            keys = grp.get("keys", [])

            # Build index map once per group
            idx = {k: i for i, k in enumerate(keys)}
            min_i = idx.get("minutes", -1)
            pts_i = idx.get("points", -1)
            reb_i = idx.get("rebounds", idx.get("totalRebounds", -1))
            ast_i = idx.get("assists", -1)
            tpm_i = next((idx[k] for k in idx if "threePoint" in k and "Made" in k.split("-")[0]), -1)

            for ath in grp.get("athletes", []):
                name = ath.get("athlete", {}).get("displayName", "")
                stats = ath.get("stats", [])
                if not name or not stats:
                    continue

                def _si(i):
                    if i < 0 or i >= len(stats): return 0
                    try: return int(str(stats[i]).split("-")[0])
                    except: return 0

                def _min(i):
                    if i < 0 or i >= len(stats): return 0.0
                    v = str(stats[i])
                    if ":" in v:
                        try:
                            p = v.split(":")
                            return float(p[0]) + float(p[1]) / 60
                        except: return 0.0
                    try: return float(v)
                    except: return 0.0

                min_played = _min(min_i)
                min_str    = stats[min_i] if min_i >= 0 and min_i < len(stats) else "0:00"

                # Normalize key so accented names (Jokić→Jokic) match NBA API autocomplete
                norm_name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii")
                players[norm_name] = {
                    "team":     team_abbr,
                    "pts":      _si(pts_i),
                    "reb":      _si(reb_i),
                    "ast":      _si(ast_i),
                    "tpm":      _si(tpm_i),
                    "min":      round(min_played, 2),
                    "min_str":  str(min_str),
                    "active":   ath.get("active", True),
                    "starter":  ath.get("starter", False),
                    "headshot": _get_headshot(name),
                }

    # ── Enrich with season avg minutes (for projection) ───────────────────
    logs = _get_player_logs()
    for name, p in players.items():
        if logs is not None:
            p_logs = logs[logs["PLAYER_NAME"] == name].sort_values("GAME_DATE", ascending=False).head(10)
            if not p_logs.empty:
                p["avg_min"] = round(float(p_logs["MIN"].mean()), 1)
                continue
        p["avg_min"] = 32.0 if p["starter"] else 20.0

    return jsonify({
        "event_id":    event_id,
        "period":      period,
        "clock":       clock,
        "is_live":     is_live,
        "is_final":    is_final,
        "scores":      scores,
        "players":     players,
        "home_win_prob": home_win_prob,
    })


@app.route("/api/analysis/start", methods=["POST"])
def api_analysis_start():
    global _analysis_job
    with _analysis_lock:
        if _analysis_job["status"] == "running":
            return jsonify({"error": "Analysis already running"}), 409
        target = request.json.get("picks", 5) if request.is_json else 5
        _analysis_job = {"status": "running", "log": [], "result": None, "error": None}
        threading.Thread(target=_run_analysis_bg, args=(target,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/analysis/status")
def api_analysis_status():
    return jsonify(_analysis_job)


@app.route("/api/analysis/reset", methods=["POST"])
def api_analysis_reset():
    global _analysis_job
    with _analysis_lock:
        _analysis_job = {"status": "idle", "log": [], "result": None, "error": None}
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Loading player data...")
    _get_player_logs()
    print("Server: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
