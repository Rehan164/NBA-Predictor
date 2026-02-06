"""
NBA Betting Advisor - Analyzes today's NBA games and recommends bets.

Uses a weighted rating model built from ESPN standings, schedule, and
historical ATS/O/U data.  Supports single bets and parlays.

Data sourced from ESPN's public API (no API key required).

DISCLAIMER: For entertainment purposes only. Bet responsibly.
"""

import sys
from datetime import datetime, timezone, timedelta
from tabulate import tabulate
from nba_tracker import (
    fetch_json,
    get_team_schedule,
    get_event_odds,
    parse_games,
    determine_ats,
    determine_ou,
    format_date,
    format_time,
    BASE,
)

# ── Constants ────────────────────────────────────────────────────────────────

STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
SCOREBOARD_URL = f"{BASE}/scoreboard"
NUM_RECENT = 15  # past games to scan for ATS / O/U trends

_odds_cache: dict = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def clamp(val, lo=0, hi=100):
    return max(lo, min(hi, val))


def parse_record(rec: str):
    """'20-5' -> (20, 5)"""
    try:
        parts = rec.split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def parse_streak(s: str):
    """`W5` -> +5, `L3` -> -3"""
    try:
        return int(s[1:]) * (1 if s[0] == "W" else -1)
    except Exception:
        return 0


def safe_int(val, default=-110):
    """Convert a string like '-110' to int, or return default."""
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


def confidence_label(conf: float) -> str:
    if conf >= 80:
        return "STRONG"
    if conf >= 65:
        return "MODERATE"
    if conf >= 50:
        return "LEAN"
    return "LOW"


def get_event_odds_cached(event_id):
    if event_id in _odds_cache:
        return _odds_cache[event_id]
    result = get_event_odds(event_id)
    _odds_cache[event_id] = result
    return result


# ── Section A: Data Fetching ─────────────────────────────────────────────────

def get_todays_games():
    """Fetch today's scoreboard and return a list of game dicts with odds."""
    # Use US Eastern time so "today" matches the NBA schedule
    et = timezone(timedelta(hours=-5))
    today = datetime.now(et).strftime("%Y%m%d")
    data = fetch_json(f"{SCOREBOARD_URL}?dates={today}")
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

        # Skip games already completed
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
            "home_id": home["team"]["id"],
            "away_id": away["team"]["id"],
            "home_abbr": home["team"]["abbreviation"],
            "away_abbr": away["team"]["abbreviation"],
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


def get_standings_data():
    """Return {team_id: stats_dict} from the standings endpoint."""
    data = fetch_json(STANDINGS_URL)
    if not data:
        return {}

    result = {}
    for conf in data.get("children", []):
        for entry in conf.get("standings", {}).get("entries", []):
            team = entry.get("team", {})
            tid = str(team.get("id", ""))

            # Build name -> value map from stats list
            sm = {}
            for stat in entry.get("stats", []):
                name = stat.get("name") or stat.get("abbreviation", "")
                if "displayValue" in stat:
                    sm[name] = stat["displayValue"]
                elif "value" in stat:
                    sm[name] = stat["value"]
                elif "summary" in stat:
                    sm[name] = stat["summary"]

            result[tid] = {
                "abbr": team.get("abbreviation", ""),
                "name": team.get("displayName", ""),
                "wins": int(float(sm.get("wins", 0))),
                "losses": int(float(sm.get("losses", 0))),
                "win_pct": float(sm.get("winPercent", 0)),
                "avg_pts_for": float(sm.get("avgPointsFor", 0)),
                "avg_pts_against": float(sm.get("avgPointsAgainst", 0)),
                "differential": float(sm.get("differential", 0)),
                "streak": str(sm.get("streak", "W0")),
                "playoff_seed": int(float(sm.get("playoffSeed", 0))),
                "home_record": str(sm.get("Home", "0-0")),
                "road_record": str(sm.get("Road", "0-0")),
                "last_ten": str(sm.get("Last Ten Games", "0-0")),
            }
    return result


def get_team_trends(team_id):
    """Compute recent ATS/O/U trends and scoring splits for a team."""
    schedule = get_team_schedule(team_id)
    if not schedule:
        return _empty_trends()

    past, _ = parse_games(schedule, team_id)
    if not past:
        return _empty_trends()

    recent = past[-NUM_RECENT:]

    covers = misses = ou_overs = ou_unders = 0
    totals = []
    home_scored, home_allowed = [], []
    away_scored, away_allowed = [], []
    recent_wins = sum(1 for g in recent if g["won"])

    for g in recent:
        details, spread_val, over_under = get_event_odds_cached(g["event_id"])
        ats = determine_ats(g, spread_val, details)
        ou = determine_ou(g, over_under)

        if ats == "Cover":
            covers += 1
        elif ats == "Miss":
            misses += 1
        if ou == "Over":
            ou_overs += 1
        elif ou == "Under":
            ou_unders += 1

        totals.append(g["total"])

        if g["is_home"]:
            home_scored.append(g["team_score"])
            home_allowed.append(g["opp_score"])
        else:
            away_scored.append(g["team_score"])
            away_allowed.append(g["opp_score"])

    ats_rated = covers + misses
    ou_rated = ou_overs + ou_unders

    return {
        "ats_covers": covers,
        "ats_misses": misses,
        "ats_pct": covers / max(ats_rated, 1),
        "ou_overs": ou_overs,
        "ou_unders": ou_unders,
        "ou_over_pct": ou_overs / max(ou_rated, 1),
        "avg_total": sum(totals) / max(len(totals), 1),
        "home_avg_scored": sum(home_scored) / max(len(home_scored), 1),
        "home_avg_allowed": sum(home_allowed) / max(len(home_allowed), 1),
        "away_avg_scored": sum(away_scored) / max(len(away_scored), 1),
        "away_avg_allowed": sum(away_allowed) / max(len(away_allowed), 1),
        "recent_wins": recent_wins,
        "recent_losses": len(recent) - recent_wins,
        "recent_pct": recent_wins / max(len(recent), 1),
    }


def _empty_trends():
    return {k: 0 for k in [
        "ats_covers", "ats_misses", "ats_pct",
        "ou_overs", "ou_unders", "ou_over_pct",
        "avg_total",
        "home_avg_scored", "home_avg_allowed",
        "away_avg_scored", "away_avg_allowed",
        "recent_wins", "recent_losses", "recent_pct",
    ]}


# ── Injuries & Starters ─────────────────────────────────────────────────────

_summary_cache: dict = {}


def _get_summary_cached(event_id):
    if event_id in _summary_cache:
        return _summary_cache[event_id]
    data = fetch_json(f"{BASE}/summary?event={event_id}")
    _summary_cache[event_id] = data
    return data


def get_game_injuries(event_id):
    """Return {team_id: [injury_dict, ...]} for a game."""
    data = _get_summary_cached(event_id)
    if not data:
        return {}

    result = {}
    for group in data.get("injuries", []):
        tid = str(group.get("team", {}).get("id", ""))
        entries = []
        for inj in group.get("injuries", []):
            athlete = inj.get("athlete", {})
            status_obj = inj.get("type", {})
            details = inj.get("details", {})
            abbr = status_obj.get("abbreviation", "?")

            # Normalise status labels
            status_map = {"O": "OUT", "D": "DOUBTFUL", "DD": "DTD",
                          "Q": "QUES", "P": "PROB"}
            entries.append({
                "name": athlete.get("displayName", "Unknown"),
                "pos": athlete.get("position", {}).get("abbreviation", ""),
                "status": status_map.get(abbr, abbr),
                "status_code": abbr,
                "injury": (f"{details.get('side', '')} "
                           f"{details.get('type', 'Unknown')}").strip(),
            })
        result[tid] = entries
    return result


def _find_team_player_group(box, team_id):
    """Find the player group matching team_id in a boxscore."""
    for pgroup in box.get("players", []):
        if str(pgroup.get("team", {}).get("id", "")) == str(team_id):
            return pgroup
    return None


def _athlete_to_starter(entry):
    """Convert a boxscore athlete entry to a starter dict."""
    athlete = entry.get("athlete", {})
    raw_stats = entry.get("stats", [])
    pos = ""
    if isinstance(athlete.get("position"), dict):
        pos = athlete["position"].get("abbreviation", "")
    return {
        "name": athlete.get("displayName", "Unknown"),
        "pos": pos,
        "minutes": raw_stats[0] if raw_stats else "-",
        "points": raw_stats[2] if len(raw_stats) > 2 else "-",
    }


def _parse_starters_from_boxscore(box, team_id):
    """Extract starter list from a boxscore player group."""
    pgroup = _find_team_player_group(box, team_id)
    if not pgroup:
        return []
    stats = pgroup.get("statistics", [])
    if not stats:
        return []
    return [_athlete_to_starter(a) for a in stats[0].get("athletes", [])
            if a.get("starter")]


def get_probable_starters(team_id):
    """Derive probable starters from the most recent completed game boxscore."""
    schedule = get_team_schedule(team_id)
    if not schedule:
        return []

    past, _ = parse_games(schedule, team_id)
    if not past:
        return []

    data = _get_summary_cached(past[-1]["event_id"])
    if not data:
        return []

    return _parse_starters_from_boxscore(data.get("boxscore", {}), team_id)


def compute_injury_penalty(starters, injuries):
    """Return a negative adjustment (0 to -10) based on starters who are out."""
    if not injuries:
        return 0, []
    injured_names = {i["name"] for i in injuries
                     if i["status_code"] in ("O", "D")}
    starter_names = {s["name"] for s in starters}
    out_starters = injured_names & starter_names
    # Also count non-starter outs as minor
    dtd_names = {i["name"] for i in injuries if i["status_code"] == "DD"}
    dtd_starters = dtd_names & starter_names

    penalty = len(out_starters) * -2.5 + len(dtd_starters) * -1.0
    penalty = max(penalty, -10)

    detail_parts = []
    if out_starters:
        detail_parts.append(f"{len(out_starters)} starter(s) OUT")
    if dtd_starters:
        detail_parts.append(f"{len(dtd_starters)} starter(s) DTD")
    return penalty, detail_parts


# ── Section B: Rating Model ─────────────────────────────────────────────────

def build_team_rating(standings, trends, is_home, injury_penalty=0,
                      injury_detail=None):
    """Return power_score, spread_score, ou_score and a factors dict.

    `factors` maps factor name -> (normalized_score_0_100, raw_display_str).
    """
    factors = {}

    # 1. Season win %
    f_win = standings["win_pct"] * 100
    record = f"{standings['wins']}-{standings['losses']}"
    factors["Season Win %"] = (f_win, f"{standings['win_pct']:.3f} ({record})")

    # 2. Venue win %
    rec = standings["home_record"] if is_home else standings["road_record"]
    w, l = parse_record(rec)
    f_venue = (w / max(w + l, 1)) * 100
    label = "Home Record" if is_home else "Road Record"
    factors[label] = (f_venue, rec)

    # 3. Last 10
    l10w, l10l = parse_record(standings["last_ten"])
    f_l10 = (l10w / max(l10w + l10l, 1)) * 100
    factors["Last 10"] = (f_l10, standings["last_ten"])

    # 4. Point differential
    diff = standings["differential"]
    f_diff = clamp((diff + 15) / 30 * 100)
    factors["Pt Diff/Game"] = (f_diff, f"{diff:+.1f}")

    # 5. Avg points for
    ppg = standings["avg_pts_for"]
    f_ppg = clamp((ppg - 95) / 30 * 100)
    factors["Avg PF"] = (f_ppg, f"{ppg:.1f}")

    # 6. Avg points against (inverted — lower is better)
    opp = standings["avg_pts_against"]
    f_opp = clamp((125 - opp) / 30 * 100)
    factors["Avg PA"] = (f_opp, f"{opp:.1f}")

    # 7. Streak
    streak = parse_streak(standings["streak"])
    f_streak = clamp((streak + 10) / 20 * 100)
    factors["Streak"] = (f_streak, standings["streak"])

    # 8. Recent ATS %
    f_ats = trends["ats_pct"] * 100
    ats_rec = f"{trends['ats_covers']}-{trends['ats_misses']}"
    factors["Recent ATS"] = (f_ats, f"{trends['ats_pct']:.0%} ({ats_rec})")

    # 9. Recent SU %
    f_recent = trends["recent_pct"] * 100
    su_rec = f"{trends['recent_wins']}-{trends['recent_losses']}"
    factors["Recent Record"] = (f_recent, f"{su_rec}")

    # ── Power score (moneyline) ──
    power = (
        f_win * 0.20 + f_venue * 0.10 + f_l10 * 0.10
        + f_diff * 0.15 + f_ppg * 0.05 + f_opp * 0.05
        + f_streak * 0.10 + f_ats * 0.15 + f_recent * 0.10
    )

    # ── Spread score (ATS-focused) ──
    spread_score = (
        f_ats * 0.30 + f_diff * 0.20 + f_recent * 0.15
        + f_streak * 0.15 + f_venue * 0.10 + f_l10 * 0.10
    )

    # ── O/U score (higher = more likely over) ──
    f_ou_trend = trends["ou_over_pct"] * 100
    f_avg_total = clamp((trends["avg_total"] - 210) / 30 * 100)
    if is_home:
        venue_total = trends["home_avg_scored"] + trends["home_avg_allowed"]
    else:
        venue_total = trends["away_avg_scored"] + trends["away_avg_allowed"]
    f_venue_pace = clamp((venue_total - 200) / 40 * 100)

    ou_score = (
        f_ou_trend * 0.35 + f_avg_total * 0.25
        + f_ppg * 0.20 + f_venue_pace * 0.20
    )

    # ── Injury adjustment ──
    if injury_penalty != 0:
        power += injury_penalty
        spread_score += injury_penalty
        detail_str = ", ".join(injury_detail) if injury_detail else "injuries"
        factors["Injuries"] = (injury_penalty, detail_str)

    return {
        "power": round(power, 1),
        "spread": round(spread_score, 1),
        "ou": round(ou_score, 1),
        "factors": factors,
    }


# ── Section C: Pick Generation ───────────────────────────────────────────────

def _top_diffs(home_factors, away_factors, home_abbr, away_abbr, n=3):
    """Return human-readable strings for the top-n diverging factors."""
    diffs = []
    for key in home_factors:
        if key in away_factors:
            h_score, h_raw = home_factors[key]
            a_score, a_raw = away_factors[key]
            diffs.append((key, h_score, a_score, h_raw, a_raw,
                          abs(h_score - a_score)))
    diffs.sort(key=lambda x: x[5], reverse=True)

    lines = []
    for fname, hs, as_, hr, ar, _ in diffs[:n]:
        better = home_abbr if hs > as_ else away_abbr
        lines.append(f"{fname}: {home_abbr} {hr}  vs  {away_abbr} {ar}")
    return lines


def generate_spread_pick(game, home_r, away_r):
    if game["spread_val"] is None:
        return None

    model_diff = home_r["spread"] - away_r["spread"]
    market_edge = -game["spread_val"]  # positive if home favored
    market_scaled = market_edge * 1.5
    edge = model_diff - market_scaled

    sv = game["spread_val"]
    if edge >= 0:
        # Take home side
        pick = f"{game['home_abbr']} {sv:+.1f}" if sv != 0 else game["spread_details"]
        odds = game["home_spread_odds"]
        side = "home"
    else:
        # Take away side — flip the spread sign
        pick = f"{game['away_abbr']} {-sv:+.1f}"
        odds = game["away_spread_odds"]
        side = "away"

    conf = clamp(50 + abs(edge) * 1.5, 30, 95)
    reasons = _top_diffs(home_r["factors"], away_r["factors"],
                         game["home_abbr"], game["away_abbr"])

    return {
        "game_id": game["event_id"],
        "game_label": f"{game['away_abbr']} @ {game['home_abbr']}",
        "bet_type": "Spread",
        "pick": pick,
        "confidence": round(conf, 1),
        "odds": odds,
        "reasoning": reasons,
        "side": side,
    }


def generate_ou_pick(game, home_r, away_r):
    if game["over_under"] is None:
        return None

    combined = (home_r["ou"] + away_r["ou"]) / 2
    edge = combined - 50

    if edge >= 0:
        pick = f"Over {game['over_under']}"
        odds = game["over_odds"]
    else:
        pick = f"Under {game['over_under']}"
        odds = game["under_odds"]

    conf = clamp(50 + abs(edge) * 1.2, 30, 95)

    ha, aa = game["home_abbr"], game["away_abbr"]
    h_pf = home_r["factors"].get("Avg PF", (0, "?"))[1]
    a_pf = away_r["factors"].get("Avg PF", (0, "?"))[1]
    reasons = [
        f"Combined O/U model: {combined:.0f}/100 ({'over' if edge >= 0 else 'under'} lean)",
        f"Avg PF: {ha} {h_pf}  vs  {aa} {a_pf}",
        f"{ha} O/U score: {home_r['ou']:.0f} | {aa} O/U score: {away_r['ou']:.0f}",
    ]

    return {
        "game_id": game["event_id"],
        "game_label": f"{aa} @ {ha}",
        "bet_type": "O/U",
        "pick": pick,
        "confidence": round(conf, 1),
        "odds": odds,
        "reasoning": reasons,
        "side": "over" if edge >= 0 else "under",
    }


def generate_ml_pick(game, home_r, away_r):
    if game["home_ml"] is None and game["away_ml"] is None:
        return None

    diff = home_r["power"] - away_r["power"]

    if diff >= 0:
        pick = f"{game['home_abbr']} ML"
        odds = game["home_ml"]
        side = "home"
    else:
        pick = f"{game['away_abbr']} ML"
        odds = game["away_ml"]
        side = "away"

    if odds is None:
        odds = -110

    conf = clamp(50 + abs(diff) * 0.8, 25, 95)
    if odds < -200:
        conf *= 0.85
        conf = clamp(conf, 25, 95)

    reasons = [f"Power rating: {game['home_abbr']} {home_r['power']:.0f}  vs  "
               f"{game['away_abbr']} {away_r['power']:.0f}"]
    reasons += _top_diffs(home_r["factors"], away_r["factors"],
                          game["home_abbr"], game["away_abbr"], 2)

    return {
        "game_id": game["event_id"],
        "game_label": f"{game['away_abbr']} @ {game['home_abbr']}",
        "bet_type": "ML",
        "pick": pick,
        "confidence": round(conf, 1),
        "odds": odds,
        "reasoning": reasons,
        "side": side,
    }


def generate_all_picks(games, standings):
    """Analyze every game and return all picks sorted by confidence.

    Also fetches injuries/starters per game and displays matchup details.
    Returns the sorted picks list.
    """
    picks = []
    total = len(games)
    analyzed_teams = {}   # cache trends per team_id
    starters_cache = {}   # cache starters per team_id

    print("  Loading injury reports & probable starters...\n")

    for idx, game in enumerate(games):
        label = f"{game['away_abbr']} @ {game['home_abbr']}"
        print(f"\r  Analyzing game {idx+1}/{total}: {label:<25s}", end="", flush=True)

        # Fetch / cache trends
        for tid in (game["home_id"], game["away_id"]):
            if tid not in analyzed_teams:
                analyzed_teams[tid] = get_team_trends(tid)
            if tid not in starters_cache:
                starters_cache[tid] = get_probable_starters(tid)

        # Injuries for this specific game
        inj_map = get_game_injuries(game["event_id"])
        home_inj = inj_map.get(game["home_id"], [])
        away_inj = inj_map.get(game["away_id"], [])

        home_starters = starters_cache[game["home_id"]]
        away_starters = starters_cache[game["away_id"]]

        # Display matchup detail
        print("\r" + " " * 60 + "\r", end="")
        display_matchup_detail(game, home_starters, away_starters,
                               home_inj, away_inj)

        # Compute injury penalties
        h_pen, h_det = compute_injury_penalty(home_starters, home_inj)
        a_pen, a_det = compute_injury_penalty(away_starters, away_inj)

        home_st = standings.get(game["home_id"], _empty_standings())
        away_st = standings.get(game["away_id"], _empty_standings())
        home_tr = analyzed_teams[game["home_id"]]
        away_tr = analyzed_teams[game["away_id"]]

        home_r = build_team_rating(home_st, home_tr, True, h_pen, h_det)
        away_r = build_team_rating(away_st, away_tr, False, a_pen, a_det)

        for gen in (generate_spread_pick, generate_ou_pick, generate_ml_pick):
            p = gen(game, home_r, away_r)
            if p:
                picks.append(p)

    picks.sort(key=lambda p: p["confidence"], reverse=True)
    return picks


def _empty_standings():
    return {
        "abbr": "?", "name": "?",
        "wins": 0, "losses": 0, "win_pct": 0.0,
        "avg_pts_for": 110, "avg_pts_against": 110,
        "differential": 0, "streak": "W0", "playoff_seed": 0,
        "home_record": "0-0", "road_record": "0-0", "last_ten": "0-0",
    }


# ── Section D: Payout ────────────────────────────────────────────────────────

def calc_single(wager, odds):
    dec = american_to_decimal(odds)
    payout = round(wager * dec, 2)
    profit = round(payout - wager, 2)
    return profit, payout


def calc_parlay(wager, legs):
    combined = 1.0
    for leg in legs:
        combined *= american_to_decimal(leg["odds"])
    payout = round(wager * combined, 2)
    profit = round(payout - wager, 2)
    return round(combined, 3), profit, payout


# ── Section E: Display ───────────────────────────────────────────────────────

def display_disclaimer():
    print()
    print("  ================================================================")
    print("  DISCLAIMER: This tool is for entertainment purposes only.")
    print("  Sports betting involves risk. Past performance does not")
    print("  guarantee future results. Please bet responsibly.")
    print("  ================================================================")


def display_todays_games(games):
    et = timezone(timedelta(hours=-5))
    today_str = datetime.now(et).strftime("%B %d, %Y")
    print(f"\n  Today's NBA Games - {today_str}")
    print(f"  {'-' * 62}")

    rows = []
    for i, g in enumerate(games, 1):
        spread = g["spread_details"] or "TBD"
        ou = g["over_under"] or "TBD"
        hml = g["home_ml"] if g["home_ml"] is not None else "-"
        aml = g["away_ml"] if g["away_ml"] is not None else "-"
        t = format_time(g["game_time"])
        matchup = f"{g['away_abbr']} @ {g['home_abbr']}"
        rows.append([i, t, matchup, spread, ou, hml, aml])

    headers = ["#", "Time", "Matchup", "Spread", "O/U", "Home ML", "Away ML"]
    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="center",
                   numalign="center"))


def _display_team_starters_injuries(abbr, starters, injuries):
    """Print starters and injury list for one team."""
    injury_lookup = {i["name"]: i for i in injuries
                     if i["status_code"] in ("O", "D", "DD")}

    if starters:
        print(f"  {abbr} Probable Starters:")
        for s in starters:
            flag = ""
            inj = injury_lookup.get(s["name"])
            if inj:
                flag = f"  ** {inj['status']} **"
            print(f"    {s['pos']:<3s} {s['name']:<24s} "
                  f"{s['minutes']:>3s} min  {s['points']:>3s} pts{flag}")
    else:
        print(f"  {abbr} Starters: unavailable")

    if injuries:
        print(f"  {abbr} Injuries:")
        for inj in injuries:
            print(f"    [{inj['status']:<4s}] {inj['name']} "
                  f"({inj['pos']}) - {inj['injury']}")
    else:
        print(f"  {abbr} Injuries: none reported")
    print()


def display_matchup_detail(game, home_starters, away_starters,
                           home_injuries, away_injuries):
    """Print starters and injury report for a single matchup."""
    ha, aa = game["home_abbr"], game["away_abbr"]
    t = format_time(game["game_time"])
    print(f"\n  {aa} @ {ha}  |  {t}")
    print(f"  {'-' * 50}")

    _display_team_starters_injuries(ha, home_starters, home_injuries)
    _display_team_starters_injuries(aa, away_starters, away_injuries)


def display_all_picks(picks):
    print(f"\n  All Picks Ranked by Confidence")
    print(f"  {'-' * 62}")

    rows = []
    for i, p in enumerate(picks, 1):
        odds_str = f"{p['odds']:+d}" if p["odds"] else "-"
        rows.append([
            i,
            p["game_label"],
            p["bet_type"],
            p["pick"],
            f"{p['confidence']:.0f}%",
            confidence_label(p["confidence"]),
            odds_str,
        ])

    headers = ["#", "Game", "Type", "Pick", "Conf", "Grade", "Odds"]
    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="center"))


def display_single(pick, wager):
    profit, payout = calc_single(wager, pick["odds"])
    odds_str = f"{pick['odds']:+d}" if pick["odds"] else "-110"

    print()
    print("  " + "=" * 54)
    print(f"  TOP PICK: {pick['pick']}  ({pick['bet_type']})")
    print(f"  Confidence: {pick['confidence']:.0f}/100  [{confidence_label(pick['confidence'])}]")
    print("  " + "=" * 54)
    print()
    print("  Why this pick:")
    for r in pick["reasoning"]:
        print(f"    - {r}")
    print()
    print(f"  Wager:            ${wager:,.2f}  at  {odds_str}")
    print(f"  Potential Profit: ${profit:,.2f}")
    print(f"  Total Payout:     ${payout:,.2f}")
    print("  " + "=" * 54)


def display_parlay(legs, wager):
    combined, profit, payout = calc_parlay(wager, legs)

    print()
    print("  " + "=" * 54)
    print(f"  PARLAY: {len(legs)} Legs")
    print("  " + "=" * 54)
    print()
    for i, leg in enumerate(legs, 1):
        odds_str = f"{leg['odds']:+d}" if leg["odds"] else "-110"
        print(f"  Leg {i}: {leg['pick']:<25s} ({leg['bet_type']})  "
              f"Conf: {leg['confidence']:.0f}%  Odds: {odds_str}")
    print()
    print(f"  Combined Decimal Odds: {combined:.3f}")
    print(f"  Wager:                 ${wager:,.2f}")
    print(f"  Potential Profit:      ${profit:,.2f}")
    print(f"  Total Payout:          ${payout:,.2f}")
    print()
    print(f"  All {len(legs)} legs must hit to win.")
    print("  " + "=" * 54)

    # Show reasoning for each leg
    print("\n  Reasoning:")
    for i, leg in enumerate(legs, 1):
        print(f"\n  Leg {i} - {leg['pick']}:")
        for r in leg["reasoning"]:
            print(f"    - {r}")


# ── Section F: Interactive Flow ──────────────────────────────────────────────

def select_parlay_legs(picks, num_legs):
    """Auto-select top picks ensuring each from a different game, conf >= 55."""
    used_games = set()
    legs = []
    for p in picks:
        if p["game_id"] in used_games:
            continue
        if p["confidence"] < 55 and len(legs) >= 2:
            continue
        legs.append(p)
        used_games.add(p["game_id"])
        if len(legs) == num_legs:
            break
    return legs


def get_wager():
    while True:
        raw = input("  How much would you like to wager? $").strip().replace(",", "")
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Please enter a valid dollar amount.")


def main():
    print("\n  NBA Betting Advisor")
    print("  ===================")
    display_disclaimer()

    # ── Load data ──
    print("\n  Loading today's games...", end="", flush=True)
    games = get_todays_games()
    if not games:
        print("\n  No NBA games with odds found for today.")
        sys.exit(0)
    print(f" {len(games)} games found.")

    print("  Loading standings...", end="", flush=True)
    standings = get_standings_data()
    if not standings:
        print("\n  Failed to load standings data.")
        sys.exit(1)
    print(" Done.")

    # ── Show today's slate ──
    display_todays_games(games)

    # ── Analyze ──
    print("\n  Analyzing matchups (fetching historical data, injuries, starters)...\n")
    picks = generate_all_picks(games, standings)

    if not picks:
        print("  No picks could be generated. Odds may not be available yet.")
        sys.exit(0)

    display_all_picks(picks)

    # ── Bet loop ──
    while True:
        print("\n  How would you like to bet?")
        print("    1. Single bet (best pick)")
        print("    2. Parlay (combine 2-6 picks)")
        print("    3. Custom parlay (choose your own picks)")
        print("    4. Exit")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            wager = get_wager()
            display_single(picks[0], wager)

        elif choice == "2":
            max_legs = min(6, len({p["game_id"] for p in picks}))
            if max_legs < 2:
                print("  Not enough games for a parlay.")
                continue
            while True:
                raw = input(f"  How many legs? (2-{max_legs}): ").strip()
                try:
                    n = int(raw)
                    if 2 <= n <= max_legs:
                        break
                except ValueError:
                    pass
                print(f"  Enter a number between 2 and {max_legs}.")
            wager = get_wager()
            legs = select_parlay_legs(picks, n)
            if len(legs) < n:
                print(f"  Note: only {len(legs)} qualifying legs found "
                      f"(need conf >= 55 from different games).")
            display_parlay(legs, wager)

        elif choice == "3":
            print("\n  Enter pick numbers from the table above (comma-separated):")
            raw = input("  Picks: ").strip()
            try:
                indices = [int(x.strip()) - 1 for x in raw.split(",")]
                selected = [picks[i] for i in indices if 0 <= i < len(picks)]
            except (ValueError, IndexError):
                print("  Invalid selection.")
                continue
            if len(selected) < 2:
                print("  Need at least 2 picks for a parlay.")
                continue
            wager = get_wager()
            display_parlay(selected, wager)

        elif choice == "4":
            print("\n  Good luck! Remember to bet responsibly.\n")
            break

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
