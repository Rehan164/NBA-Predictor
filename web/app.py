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
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
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

# ── Model training job state ─────────────────────────────────────────────
_model_job = {"status": "idle", "log": [], "error": None}
_model_lock = threading.Lock()

# ── Player model training job state ──────────────────────────────────
_player_model_job = {"status": "idle", "log": [], "error": None}
_player_model_lock = threading.Lock()


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
            before = len(_player_logs)
            _player_logs = _player_logs.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
            dupes = before - len(_player_logs)
            if dupes:
                print(f"  Dropped {dupes:,} duplicate rows")
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


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


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
                            books = []
                            for bk in ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]:
                                if bk in data.get("books", {}):
                                    bd = data["books"][bk]
                                    books.append({
                                        "key": bk,
                                        "title": bd.get("title", bk.title()),
                                        "line": bd.get("line", data.get("line")),
                                        "over": bd.get("over"),
                                        "under": bd.get("under"),
                                    })
                            all_props[player_name][short] = {
                                "line": books[0]["line"] if books else data.get("line"),
                                "books": books,
                            }
        _props_cache = all_props
        _props_cache_date = today
        return jsonify(all_props)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh")
def api_refresh():
    global _today_games, _today_games_date, _injuries_cache, _injuries_cache_date, _props_cache, _props_cache_date
    _today_games = _today_games_date = None
    _injuries_cache = _injuries_cache_date = None
    _props_cache = _props_cache_date = None
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


# ── Model Page ────────────────────────────────────────────────────────────

@app.route("/model")
def model_page():
    return render_template("model.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


@app.route("/api/model/status")
def api_model_status():
    try:
        from nba_ml.advanced_model import get_model_status
        status = get_model_status()
    except Exception as e:
        status = {"trained": False, "error": str(e)}

    # Merge training-in-progress flag
    with _model_lock:
        status["training_in_progress"] = _model_job["status"] == "running"

    return jsonify(status)


@app.route("/api/model/train", methods=["POST"])
def api_model_train():
    global _model_job
    with _model_lock:
        if _model_job["status"] == "running":
            return jsonify({"error": "Training already in progress"}), 409
        _model_job = {"status": "running", "log": [], "error": None, "progress": {}}

    def _run():
        global _model_job
        try:
            import torch
            import numpy as np
            from nba_ml.advanced_model import train_model, RANDOM_STATE

            torch.manual_seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)

            def progress(msg):
                _model_job["log"].append(msg)

            model, metrics, norm_params = train_model(
                progress_callback=progress,
                progress_state=_model_job["progress"],
            )
            _model_job["log"].append("TRAINING COMPLETE")
            _model_job["status"] = "done"
        except Exception as e:
            import traceback
            _model_job["error"] = str(e)
            _model_job["log"].append(f"ERROR: {e}")
            _model_job["log"].append(traceback.format_exc())
            _model_job["status"] = "error"

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/model/train/status")
def api_model_train_status():
    return jsonify(_model_job)



@app.route("/api/model/learn", methods=["POST"])
def api_model_learn():
    """Fine-tune model on yesterday's actual results."""
    try:
        from nba_ml.advanced_model import learn_from_results
        target_date = None
        if request.is_json and request.json.get("date"):
            target_date = request.json["date"]
        result = learn_from_results(target_date=target_date)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/model/predict")
def api_model_predict():
    try:
        from nba_ml.advanced_model import predict_today
        preds = predict_today()
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/update-data", methods=["POST"])
def api_model_update_data():
    try:
        from nba_ml.collect_data import update_current_season as update_games
        from nba_ml.collect_players import update_current_season as update_players

        update_games()
        update_players()

        # Invalidate player logs cache
        global _player_logs
        _player_logs = None

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Player Model Page ────────────────────────────────────────────────

@app.route("/player-model")
def player_model_page():
    return render_template("player_model.html", date=datetime.now(ET).strftime("%A, %B %d, %Y"))


@app.route("/api/player-model/status")
def api_player_model_status():
    try:
        from nba_ml.player_props_nn import get_model_status
        status = get_model_status()
    except Exception as e:
        status = {"trained": False, "error": str(e)}

    with _player_model_lock:
        status["training_in_progress"] = _player_model_job["status"] == "running"

    return jsonify(status)


@app.route("/api/player-model/train", methods=["POST"])
def api_player_model_train():
    global _player_model_job
    with _player_model_lock:
        if _player_model_job["status"] == "running":
            return jsonify({"error": "Training already in progress"}), 409
        _player_model_job = {"status": "running", "log": [], "error": None, "progress": {}}

    def _run():
        global _player_model_job
        try:
            import torch
            import numpy as np
            from nba_ml.player_props_nn import train_model, ENSEMBLE_SEEDS

            torch.manual_seed(ENSEMBLE_SEEDS[0])
            np.random.seed(ENSEMBLE_SEEDS[0])

            def progress(msg):
                _player_model_job["log"].append(msg)

            train_model(
                progress_callback=progress,
                progress_state=_player_model_job.get("progress"),
            )
            _player_model_job["log"].append("TRAINING COMPLETE")
            _player_model_job["status"] = "done"
        except Exception as e:
            import traceback
            _player_model_job["error"] = str(e)
            _player_model_job["log"].append(f"ERROR: {e}")
            _player_model_job["log"].append(traceback.format_exc())
            _player_model_job["status"] = "error"

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/player-model/train/status")
def api_player_model_train_status():
    return jsonify(_player_model_job)


@app.route("/api/player-model/predict")
def api_player_model_predict():
    try:
        from nba_ml.player_props_nn import predict_today
        preds = predict_today()
        return jsonify(preds)
    except Exception as e:
        import traceback
        return jsonify([{"error": str(e)}]), 500


if __name__ == "__main__":
    print("Loading player data...")
    _get_player_logs()
    print("Server: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
