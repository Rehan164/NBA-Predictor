"""
NBA Injury Report Integration

Fetches and manages player injury data to adjust predictions.
Injuries significantly impact spread, totals, and player props.

Usage:
    from nba_ml.injuries import get_injury_report, apply_injury_adjustments
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime


# ═════════════════════════════════════════════════════════════════════════════
# MANUAL INJURY CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Star player impact values (points to adjust spread/total when OUT)
# Based on historical impact analysis
STAR_IMPACT = {
    # Lakers
    "LeBron James": {"spread": 4.5, "total": 6.0, "usage_shift": 0.15},
    "Anthony Davis": {"spread": 3.5, "total": 5.0, "usage_shift": 0.12},

    # Mavericks
    "Luka Doncic": {"spread": 5.5, "total": 7.0, "usage_shift": 0.18},
    "Kyrie Irving": {"spread": 3.0, "total": 4.5, "usage_shift": 0.10},

    # Warriors
    "Stephen Curry": {"spread": 5.0, "total": 6.5, "usage_shift": 0.16},

    # Bucks
    "Giannis Antetokounmpo": {"spread": 6.0, "total": 7.5, "usage_shift": 0.20},
    "Damian Lillard": {"spread": 3.5, "total": 5.0, "usage_shift": 0.12},

    # Nuggets
    "Nikola Jokic": {"spread": 5.5, "total": 7.0, "usage_shift": 0.18},

    # Celtics
    "Jayson Tatum": {"spread": 4.0, "total": 5.5, "usage_shift": 0.14},
    "Jaylen Brown": {"spread": 3.0, "total": 4.5, "usage_shift": 0.11},

    # Suns
    "Kevin Durant": {"spread": 4.5, "total": 6.0, "usage_shift": 0.15},
    "Devin Booker": {"spread": 3.5, "total": 5.0, "usage_shift": 0.13},

    # 76ers
    "Joel Embiid": {"spread": 5.5, "total": 7.0, "usage_shift": 0.17},

    # Thunder
    "Shai Gilgeous-Alexander": {"spread": 4.5, "total": 6.0, "usage_shift": 0.16},

    # Cavaliers
    "Donovan Mitchell": {"spread": 3.5, "total": 5.0, "usage_shift": 0.13},

    # Knicks
    "Jalen Brunson": {"spread": 3.5, "total": 5.0, "usage_shift": 0.14},

    # Add more stars as needed...
}

# Default impact for non-star starters
DEFAULT_STARTER_IMPACT = {"spread": 1.5, "total": 2.5, "usage_shift": 0.05}


# ═════════════════════════════════════════════════════════════════════════════
# INJURY REPORT FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def get_espn_injury_report() -> Dict[str, List[Dict]]:
    """
    Fetch current injury report from ESPN API.

    Returns:
        Dict mapping team abbreviations to list of injured players.
        Example: {"LAL": [{"name": "LeBron James", "status": "OUT"}, ...]}
    """
    try:
        # Use the global injuries endpoint
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        injuries_by_team = {}

        # Data is structured as array of team objects, each with "injuries" array
        for team_obj in data.get("injuries", []):
            team_name = team_obj.get("displayName", "")

            # Process injuries for this team
            for injury in team_obj.get("injuries", []):
                athlete = injury.get("athlete", {})
                player_name = athlete.get("displayName", "Unknown")

                # Get team abbreviation from athlete's team info (most reliable)
                team_info = athlete.get("team", {})
                team_abbr = team_info.get("abbreviation")

                if not team_abbr:
                    continue  # Skip if we can't get team abbreviation

                # Get status
                status = injury.get("status", "Unknown")

                # Get injury details
                details = injury.get("longComment") or injury.get("shortComment") or ""

                # Add to our dict
                if team_abbr not in injuries_by_team:
                    injuries_by_team[team_abbr] = []

                injuries_by_team[team_abbr].append({
                    "name": player_name,
                    "status": status,
                    "injury": details,
                })

        return injuries_by_team

    except Exception as e:
        print(f"  Warning: Could not fetch ESPN injury report: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_nba_injury_report() -> Dict[str, List[Dict]]:
    """
    Fetch injury report from NBA.com API (alternative source).

    Returns:
        Dict mapping team abbreviations to list of injured players.
    """
    try:
        # NBA.com injury endpoint
        url = "https://www.nba.com/stats/team/injuries"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
        }
        response = requests.get(url, headers=headers, timeout=10)

        # This is a simplified version - NBA.com may require more complex parsing
        # Consider using a dedicated NBA API library like nba_api
        return {}

    except Exception as e:
        print(f"  Warning: Could not fetch NBA injury report: {e}")
        return {}


def parse_manual_injuries(injury_text: str) -> Dict[str, List[str]]:
    """
    Parse manually entered injuries from command line or config.

    Format: "LAL:LeBron James,Anthony Davis|DAL:Luka Doncic"

    Returns:
        Dict mapping team to list of player names OUT
    """
    injuries = {}

    if not injury_text:
        return injuries

    teams = injury_text.split("|")
    for team_entry in teams:
        if ":" not in team_entry:
            continue

        team_abbr, players = team_entry.split(":", 1)
        team_abbr = team_abbr.strip().upper()
        player_list = [p.strip() for p in players.split(",")]

        injuries[team_abbr] = player_list

    return injuries


# ═════════════════════════════════════════════════════════════════════════════
# INJURY IMPACT CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def calculate_team_injury_impact(team: str, injuries: List[str]) -> Dict[str, float]:
    """
    Calculate the impact of injuries on team performance.

    Args:
        team: Team abbreviation (e.g., "LAL")
        injuries: List of player names who are OUT

    Returns:
        Dict with spread_adjustment, total_adjustment, usage_shift
    """
    total_spread_impact = 0.0
    total_total_impact = 0.0
    total_usage_shift = 0.0

    for player_name in injuries:
        impact = STAR_IMPACT.get(player_name, DEFAULT_STARTER_IMPACT)
        total_spread_impact += impact["spread"]
        total_total_impact += impact["total"]
        total_usage_shift += impact["usage_shift"]

    return {
        "spread_adjustment": total_spread_impact,
        "total_adjustment": total_total_impact,
        "usage_shift": total_usage_shift,
    }


def apply_injury_adjustments(
    home_team: str,
    away_team: str,
    base_spread: float,
    base_total: float,
    injuries: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Adjust predictions based on injury report.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        base_spread: Model's base spread prediction (positive = home favored)
        base_total: Model's base total prediction
        injuries: Dict mapping team -> list of OUT players

    Returns:
        Dict with adjusted_spread, adjusted_total, confidence_penalty
    """
    home_injuries = injuries.get(home_team, [])
    away_injuries = injuries.get(away_team, [])

    home_impact = calculate_team_injury_impact(home_team, home_injuries)
    away_impact = calculate_team_injury_impact(away_team, away_injuries)

    # Spread adjustment:
    # If home loses LeBron (-4.5), spread moves AGAINST home
    # If away loses Luka (-5.5), spread moves TOWARD home
    spread_adjustment = away_impact["spread_adjustment"] - home_impact["spread_adjustment"]
    adjusted_spread = base_spread + spread_adjustment

    # Total adjustment:
    # Both teams losing stars reduces total
    total_adjustment = -(home_impact["total_adjustment"] + away_impact["total_adjustment"])
    adjusted_total = base_total + total_adjustment

    # Confidence penalty: More injuries = less predictable
    total_injuries = len(home_injuries) + len(away_injuries)
    confidence_penalty = min(0.15, total_injuries * 0.03)  # Max 15% penalty

    return {
        "adjusted_spread": adjusted_spread,
        "adjusted_total": adjusted_total,
        "spread_adjustment": spread_adjustment,
        "total_adjustment": total_adjustment,
        "confidence_penalty": confidence_penalty,
        "home_injuries": home_injuries,
        "away_injuries": away_injuries,
    }


def get_usage_boost_candidates(
    team: str,
    injuries: List[str],
    roster: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Identify which players will see increased usage when stars are out.

    Args:
        team: Team abbreviation
        injuries: List of OUT players
        roster: Optional list of available players (if None, uses default top options)

    Returns:
        Dict mapping player name -> usage boost multiplier (1.15 = 15% more usage)
    """
    if not injuries:
        return {}

    # Calculate total usage shift from injuries
    total_usage_freed = sum(
        STAR_IMPACT.get(player, DEFAULT_STARTER_IMPACT)["usage_shift"]
        for player in injuries
    )

    # Team-specific beneficiaries (manual for now, could be data-driven)
    usage_beneficiaries = {
        "LAL": {
            # If LeBron is out:
            "Anthony Davis": 1.20,
            "Austin Reaves": 1.25,
            "D'Angelo Russell": 1.20,
        },
        "DAL": {
            # If Luka is out:
            "Kyrie Irving": 1.25,
            "Dereck Lively II": 1.15,
            "PJ Washington": 1.20,
        },
        # Add more teams...
    }

    team_beneficiaries = usage_beneficiaries.get(team, {})

    # Filter to only players not also injured
    result = {}
    for player, boost in team_beneficiaries.items():
        if player not in injuries:
            result[player] = boost

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def get_injury_report(manual_input: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get comprehensive injury report from all sources.

    Args:
        manual_input: Optional manual injury string to override API
                      Format: "LAL:LeBron James,AD|DAL:Luka Doncic"

    Returns:
        Dict mapping team -> list of OUT players
    """
    if manual_input:
        print("  Using manual injury input")
        return parse_manual_injuries(manual_input)

    print("  Fetching injury report from ESPN...")
    espn_injuries = get_espn_injury_report()

    if not espn_injuries:
        print("  No injury data retrieved from ESPN")
        return {}

    # Filter to only OUT players (not questionable/doubtful/day-to-day)
    # ESPN uses different status formats, so check for common "out" variants
    out_players = {}
    for team, injuries in espn_injuries.items():
        out_list = []
        for inj in injuries:
            status = inj.get("status", "").upper()
            # Check for various "out" statuses
            if status in ["OUT", "INJURY_STATUS_OUT"] or "OUT" in status:
                out_list.append(inj["name"])

        if out_list:
            out_players[team] = out_list

    return out_players


def print_injury_report(injuries: Dict[str, List[str]]):
    """Pretty print injury report."""
    if not injuries:
        print("  No significant injuries reported")
        return

    print("\n  INJURY REPORT (OUT):")
    print("  " + "-" * 50)
    for team, players in sorted(injuries.items()):
        print(f"  {team}:")
        for player in players:
            star_marker = " [STAR]" if player in STAR_IMPACT else ""
            print(f"    - {player}{star_marker}")
    print()
