"""
Enhanced Injury Report System.

Fetches injury data from multiple sources:
  1. ESPN injury API (primary — free, no key)
  2. ESPN game-level roster status (pre-game scratches)

Categorizes players by impact level and returns structured data
for both team-level adjustments and player prop analysis.
"""

import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Status categories ranked by severity
STATUS_OUT = {"OUT", "INJURY_STATUS_OUT", "SUSPENSION"}
STATUS_DOUBTFUL = {"DOUBTFUL", "INJURY_STATUS_DOUBTFUL"}
STATUS_QUESTIONABLE = {"QUESTIONABLE", "INJURY_STATUS_QUESTIONABLE", "GAME_TIME_DECISION"}
STATUS_DAY_TO_DAY = {"DAY-TO-DAY", "INJURY_STATUS_DAY_TO_DAY", "DAY TO DAY"}


def fetch_espn_injuries() -> Dict[str, List[Dict]]:
    """
    Fetch full injury report from ESPN API with all status levels.

    Returns:
        Dict mapping team abbreviation -> list of injury records.
        Each record: {name, status, injury, position, is_out, is_doubtful, is_questionable}
    """
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        injuries_by_team = {}

        for team_obj in data.get("injuries", []):
            for injury in team_obj.get("injuries", []):
                athlete = injury.get("athlete", {})
                player_name = athlete.get("displayName", "Unknown")
                team_info = athlete.get("team", {})
                team_abbr = team_info.get("abbreviation")

                if not team_abbr:
                    continue

                status_raw = (injury.get("status") or "Unknown").upper().strip()

                record = {
                    "name": player_name,
                    "status": status_raw,
                    "injury": injury.get("longComment") or injury.get("shortComment") or "",
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "is_out": status_raw in STATUS_OUT or "OUT" in status_raw,
                    "is_doubtful": status_raw in STATUS_DOUBTFUL or "DOUBTFUL" in status_raw,
                    "is_questionable": status_raw in STATUS_QUESTIONABLE or "QUESTIONABLE" in status_raw,
                }

                if team_abbr not in injuries_by_team:
                    injuries_by_team[team_abbr] = []
                injuries_by_team[team_abbr].append(record)

        return injuries_by_team

    except Exception as e:
        print(f"  WARNING: Failed to fetch ESPN injuries: {e}")
        return {}


def fetch_game_day_status(event_id: str) -> Dict[str, str]:
    """
    Fetch game-day roster status from ESPN event endpoint.
    This catches last-minute scratches not in the injury report.

    Returns:
        Dict mapping player name -> status string
    """
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        statuses = {}
        for roster in data.get("rosters", []):
            for entry in roster.get("roster", []):
                athlete = entry.get("athlete", {})
                name = athlete.get("displayName", "")
                # Check for inactive/DNP status
                is_active = entry.get("active", True)
                if not is_active and name:
                    statuses[name] = "INACTIVE"

        return statuses

    except Exception:
        return {}


def get_full_injury_report(
    team_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Get comprehensive injury report with categorized players.

    Args:
        team_filter: Optional list of team abbreviations to filter

    Returns:
        Dict per team with categorized injury lists:
        {
            "LAL": {
                "out": [{"name": "LeBron James", "injury": "knee", ...}],
                "doubtful": [...],
                "questionable": [...],
                "all_injuries": [...],
            }
        }
    """
    print("  Fetching injury reports...")
    raw = fetch_espn_injuries()

    report = {}
    total_out = 0
    total_questionable = 0

    for team_abbr, injuries in raw.items():
        if team_filter and team_abbr not in team_filter:
            continue

        categorized = {
            "out": [],
            "doubtful": [],
            "questionable": [],
            "all_injuries": injuries,
        }

        for inj in injuries:
            if inj["is_out"]:
                categorized["out"].append(inj)
                total_out += 1
            elif inj["is_doubtful"]:
                categorized["doubtful"].append(inj)
            elif inj["is_questionable"]:
                categorized["questionable"].append(inj)
                total_questionable += 1

        report[team_abbr] = categorized

    print(f"  Found {total_out} OUT, {total_questionable} questionable across {len(report)} teams")
    return report


def get_out_players(report: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Extract just the OUT player names per team (for model adjustments)."""
    return {
        team: [p["name"] for p in data["out"]]
        for team, data in report.items()
        if data["out"]
    }


def format_injury_report(report: Dict[str, Dict], teams: Optional[List[str]] = None) -> str:
    """Format injury report as a readable string for Claude analysis."""
    lines = []
    teams_to_show = teams or sorted(report.keys())

    for team in teams_to_show:
        if team not in report:
            continue

        data = report[team]
        if not data["all_injuries"]:
            continue

        lines.append(f"\n{team}:")

        if data["out"]:
            for p in data["out"]:
                pos = f" ({p['position']})" if p["position"] else ""
                inj = f" - {p['injury']}" if p["injury"] else ""
                lines.append(f"  OUT: {p['name']}{pos}{inj}")

        if data["doubtful"]:
            for p in data["doubtful"]:
                pos = f" ({p['position']})" if p["position"] else ""
                lines.append(f"  DOUBTFUL: {p['name']}{pos}")

        if data["questionable"]:
            for p in data["questionable"]:
                pos = f" ({p['position']})" if p["position"] else ""
                lines.append(f"  QUESTIONABLE: {p['name']}{pos}")

    return "\n".join(lines) if lines else "No significant injuries reported."
