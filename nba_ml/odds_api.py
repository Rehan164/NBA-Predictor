"""
Odds API wrapper for the Model page.

Thin re-export of picks.odds_api with a convenience helper that returns
today's games with best-available odds in a flat dict keyed by
(home_team_full_name, away_team_full_name).

Usage:
    from nba_ml.odds_api import fetch_todays_odds
"""

import os
from typing import Dict, List, Optional
import requests

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# NBA full-name → abbreviation for matching
_FULL_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]


def fetch_todays_odds() -> List[Dict]:
    """
    Fetch today's NBA odds and return a list of dicts:
    [
      {
        "home": "LAL", "away": "DAL",
        "spread": -4.5, "total": 224.5,
        "home_ml": -180, "away_ml": +155,
      }, ...
    ]
    """
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        return []

    try:
        r = requests.get(f"{BASE_URL}/sports/{SPORT}/odds", params={
            "apiKey": key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return []

    results = []
    for event in data:
        home_full = event.get("home_team", "")
        away_full = event.get("away_team", "")
        home_abbr = _FULL_TO_ABBR.get(home_full, home_full)
        away_abbr = _FULL_TO_ABBR.get(away_full, away_full)

        row = {
            "home": home_abbr, "away": away_abbr,
            "spread": None, "total": None,
            "home_ml": None, "away_ml": None,
        }

        for book_key in PREFERRED_BOOKS:
            bm = None
            for b in event.get("bookmakers", []):
                if b["key"] == book_key:
                    bm = b
                    break
            if not bm:
                continue

            for market in bm.get("markets", []):
                mk = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if mk == "spreads" and row["spread"] is None:
                    ho = outcomes.get(home_full, {})
                    if ho.get("point") is not None:
                        row["spread"] = ho["point"]

                if mk == "totals" and row["total"] is None:
                    ov = outcomes.get("Over", {})
                    if ov.get("point") is not None:
                        row["total"] = ov["point"]

                if mk == "h2h" and row["home_ml"] is None:
                    ho = outcomes.get(home_full, {})
                    ao = outcomes.get(away_full, {})
                    if ho.get("price") is not None:
                        row["home_ml"] = ho["price"]
                        row["away_ml"] = ao.get("price")

        results.append(row)

    return results
