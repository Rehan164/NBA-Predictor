"""
NBA Team Tracker - View past results and upcoming games with betting lines.

Data sourced from ESPN's public API (no API key required).
"""

import sys
import requests
from datetime import datetime, timezone
from tabulate import tabulate

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NUM_PAST_GAMES = 15
NUM_FUTURE_GAMES = 10


# ── API helpers ──────────────────────────────────────────────────────────────

def fetch_json(url):
    """GET a URL and return parsed JSON, or None on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"  [error] {e}")
        return None


def get_all_teams():
    """Return a list of (id, abbreviation, displayName) sorted by name."""
    data = fetch_json(f"{BASE}/teams?limit=50")
    if not data:
        return []
    teams = []
    for entry in data["sports"][0]["leagues"][0]["teams"]:
        t = entry["team"]
        teams.append((t["id"], t["abbreviation"], t["displayName"]))
    return sorted(teams, key=lambda x: x[2])


def get_team_schedule(team_id):
    """Return the raw schedule JSON for a team."""
    return fetch_json(f"{BASE}/teams/{team_id}/schedule")


def get_event_odds(event_id):
    """Fetch spread and over/under for a single event via the summary endpoint."""
    data = fetch_json(f"{BASE}/summary?event={event_id}")
    if not data:
        return None, None, None
    pc = data.get("pickcenter", [])
    if not pc:
        return None, None, None
    entry = pc[0]
    return entry.get("details"), entry.get("spread"), entry.get("overUnder")


# ── Parsing helpers ──────────────────────────────────────────────────────────

def parse_games(schedule_data, team_id):
    """Split schedule events into completed and upcoming game dicts."""
    past, future = [], []
    team_info = schedule_data.get("team", {})
    team_abbr = team_info.get("abbreviation", "???")

    for event in schedule_data.get("events", []):
        comp = event["competitions"][0]
        status = comp["status"]["type"]
        game_date = event["date"]

        # Identify home/away competitors
        home = away = None
        for c in comp["competitors"]:
            if c["homeAway"] == "home":
                home = c
            else:
                away = c

        if not home or not away:
            continue

        is_home = home["team"]["id"] == str(team_id)
        opponent = away["team"] if is_home else home["team"]
        opp_label = f"vs {opponent['abbreviation']}" if is_home else f"@ {opponent['abbreviation']}"

        game = {
            "event_id": event["id"],
            "date": game_date,
            "opponent": opp_label,
            "opponent_name": opponent["displayName"],
            "is_home": is_home,
            "team_abbr": team_abbr,
        }

        if status.get("completed"):
            home_score = home["score"]["value"] if isinstance(home["score"], dict) else float(home["score"])
            away_score = away["score"]["value"] if isinstance(away["score"], dict) else float(away["score"])

            team_score = home_score if is_home else away_score
            opp_score = away_score if is_home else home_score

            game["team_score"] = int(team_score)
            game["opp_score"] = int(opp_score)
            game["total"] = int(team_score + opp_score)
            game["won"] = team_score > opp_score
            past.append(game)
        elif status.get("name") == "STATUS_SCHEDULED":
            future.append(game)

    return past, future


def determine_ats(game, spread_val, details):
    """Determine if the selected team covered the spread."""
    if spread_val is None or details is None:
        return "N/A"

    margin = game["team_score"] - game["opp_score"]  # positive = team won

    # `details` looks like "BOS -7.5" or "DAL -3.5"
    # `spread_val` is the raw number from ESPN (positive means home is underdog)
    # We need to figure out the team's spread
    parts = details.split()
    if len(parts) < 2:
        return "N/A"

    fav_abbr = parts[0]
    try:
        line = float(parts[1])
    except ValueError:
        return "N/A"

    # line is negative (e.g., -7.5) for the favorite
    if fav_abbr == game["team_abbr"]:
        # Our team is favored: they need to win by more than abs(line)
        team_spread = line  # e.g., -7.5
    else:
        # Our team is the underdog: they get the points
        team_spread = -line  # e.g., +7.5

    adjusted_margin = margin + team_spread
    if adjusted_margin > 0:
        return "Cover"
    elif adjusted_margin < 0:
        return "Miss"
    else:
        return "Push"


def determine_ou(game, over_under):
    """Determine if the game went over or under."""
    if over_under is None:
        return "N/A"
    if game["total"] > over_under:
        return "Over"
    elif game["total"] < over_under:
        return "Under"
    else:
        return "Push"


# ── Display ──────────────────────────────────────────────────────────────────

def format_date(iso_str):
    """Convert ISO date string to 'Mon MM/DD' format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%a %m/%d")
    except Exception:
        return iso_str[:10]


def format_time(iso_str):
    """Convert ISO date string to 'h:MM PM ET' format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        from datetime import timedelta
        et = dt - timedelta(hours=5)  # rough UTC -> ET
        return et.strftime("%I:%M %p ET").lstrip("0")
    except Exception:
        return ""


def display_team_header(schedule_data):
    """Print team name and record."""
    team = schedule_data.get("team", {})
    name = team.get("displayName", "Unknown")
    record = ""
    rec_items = team.get("recordSummary", "")
    if rec_items:
        record = f" ({rec_items})"
    print(f"\n{'=' * 60}")
    print(f"  {name}{record}")
    print(f"{'=' * 60}")


def display_past_games(past_games):
    """Fetch odds and display a table of recent completed games."""
    if not past_games:
        print("\n  No completed games found.\n")
        return

    games = past_games[-NUM_PAST_GAMES:]
    print(f"\n  Last {len(games)} Games")
    print(f"  {'-' * 56}")

    rows = []
    total_games = len(games)
    for i, game in enumerate(games):
        pct = int((i + 1) / total_games * 100)
        print(f"\r  Fetching odds... {pct}%", end="", flush=True)

        details, spread_val, over_under = get_event_odds(game["event_id"])

        result = "W" if game["won"] else "L"
        score = f"{game['team_score']}-{game['opp_score']}"

        spread_display = details if details else "-"
        ou_display = str(over_under) if over_under else "-"
        ats = determine_ats(game, spread_val, details)
        ou_result = determine_ou(game, over_under)

        rows.append([
            format_date(game["date"]),
            game["opponent"],
            result,
            score,
            spread_display,
            ou_display,
            ats,
            ou_result,
        ])

    print("\r" + " " * 40 + "\r", end="")  # clear progress line

    headers = ["Date", "Opponent", "W/L", "Score", "Spread", "O/U", "ATS", "O/U Result"]
    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="center"))

    # Summary stats
    if rows:
        covers = sum(1 for r in rows if r[6] == "Cover")
        misses = sum(1 for r in rows if r[6] == "Miss")
        overs = sum(1 for r in rows if r[7] == "Over")
        unders = sum(1 for r in rows if r[7] == "Under")
        wins = sum(1 for r in rows if r[2] == "W")
        losses = sum(1 for r in rows if r[2] == "L")
        rated = sum(1 for r in rows if r[6] != "N/A" and r[6] != "-")

        print(f"\n  Record: {wins}-{losses}", end="")
        if rated:
            print(f"  |  ATS: {covers}-{misses}  |  O/U: {overs} Over, {unders} Under")
        else:
            print()


def display_future_games(future_games):
    """Fetch odds and display a table of upcoming games."""
    if not future_games:
        print("\n  No upcoming games found.\n")
        return

    games = future_games[:NUM_FUTURE_GAMES]
    print(f"\n  Next {len(games)} Upcoming Games")
    print(f"  {'-' * 56}")

    rows = []
    total_games = len(games)
    for i, game in enumerate(games):
        pct = int((i + 1) / total_games * 100)
        print(f"\r  Fetching lines... {pct}%", end="", flush=True)

        details, spread_val, over_under = get_event_odds(game["event_id"])

        spread_display = details if details else "TBD"
        ou_display = str(over_under) if over_under else "TBD"
        time_str = format_time(game["date"])

        rows.append([
            format_date(game["date"]),
            time_str,
            game["opponent"],
            game["opponent_name"],
            spread_display,
            ou_display,
        ])

    print("\r" + " " * 40 + "\r", end="")  # clear progress line

    headers = ["Date", "Time", "Opp", "Opponent", "Spread", "O/U"]
    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="center"))


# ── Team selection ───────────────────────────────────────────────────────────

def select_team(teams):
    """Interactive team picker. Supports number, abbreviation, or partial name."""
    print("\n  NBA Teams")
    print("  " + "-" * 40)

    for i, (tid, abbr, name) in enumerate(teams, 1):
        print(f"  {i:>2}. {abbr:<5} {name}")

    print()
    while True:
        choice = input("  Select a team (number, abbreviation, or name): ").strip()
        if not choice:
            continue

        # Try by number
        try:
            idx = int(choice)
            if 1 <= idx <= len(teams):
                return teams[idx - 1]
        except ValueError:
            pass

        # Try by abbreviation or partial name (case-insensitive)
        choice_lower = choice.lower()
        matches = [
            t for t in teams
            if t[1].lower() == choice_lower
            or choice_lower in t[2].lower()
        ]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"  Multiple matches: {', '.join(m[2] for m in matches)}")
            print("  Please be more specific.")
        else:
            print("  Team not found. Try again.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n  NBA Team Tracker")
    print("  ================")
    print("  Loading teams...", end="", flush=True)

    teams = get_all_teams()
    if not teams:
        print("\n  Failed to load NBA teams. Check your internet connection.")
        sys.exit(1)

    print(f"\r  Loaded {len(teams)} teams.   \n")

    while True:
        team_id, team_abbr, team_name = select_team(teams)
        print(f"\n  Loading data for {team_name}...")

        schedule = get_team_schedule(team_id)
        if not schedule:
            print("  Failed to load schedule. Try again.")
            continue

        display_team_header(schedule)
        past, future = parse_games(schedule, team_id)

        display_past_games(past)
        display_future_games(future)

        print()
        again = input("  Look up another team? (y/n): ").strip().lower()
        if again != "y":
            print("  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
