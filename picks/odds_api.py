"""
Odds API Integration — Fetch live betting lines and player props.

Uses The-Odds-API (https://the-odds-api.com) for:
  - Game lines: spreads, totals, moneylines from multiple books
  - Player props: points, rebounds, assists O/U from DraftKings, FanDuel, etc.

Free tier: 500 requests/month. Player prop requests cost more quota.

Setup:
  export ODDS_API_KEY="your_key_here"
  Get a free key at: https://the-odds-api.com/#get-access
"""

import os
import requests
from typing import Dict, List, Optional

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Bookmakers to prioritize (in order of preference)
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]

# Player prop market keys
PROP_MARKETS = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "threes": "player_threes",
    "pts_reb_ast": "player_points_rebounds_assists",
}


def _get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        print("  WARNING: ODDS_API_KEY not set. Set it with: export ODDS_API_KEY='your_key'")
        print("  Get a free key at: https://the-odds-api.com/#get-access")
    return key


def _request(endpoint: str, params: dict) -> Optional[dict]:
    key = _get_api_key()
    if not key:
        return None

    params["apiKey"] = key
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=15)
        r.raise_for_status()

        # Track quota usage from response headers
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")
        print(f"  [Odds API] Quota: {used} used, {remaining} remaining")

        return r.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("  ERROR: Invalid ODDS_API_KEY. Check your key.")
        elif e.response.status_code == 429:
            print("  ERROR: Odds API quota exceeded. Wait until next month or upgrade.")
        else:
            print(f"  ERROR: Odds API HTTP {e.response.status_code}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Odds API request failed: {e}")
        return None


def get_game_odds(regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict]:
    """
    Fetch game-level odds (moneyline, spread, totals) for today's NBA games.

    Returns list of games with odds from multiple bookmakers.
    """
    data = _request(f"sports/{SPORT}/odds", {
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    })

    if not data:
        return []

    games = []
    for event in data:
        game = {
            "id": event["id"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "commence_time": event["commence_time"],
            "bookmakers": {},
        }

        for bm in event.get("bookmakers", []):
            book_key = bm["key"]
            book_data = {"title": bm["title"], "markets": {}}

            for market in bm.get("markets", []):
                market_key = market["key"]
                outcomes = {}
                for outcome in market.get("outcomes", []):
                    outcomes[outcome["name"]] = {
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                    }
                book_data["markets"][market_key] = outcomes

            game["bookmakers"][book_key] = book_data

        games.append(game)

    return games


def get_player_props(event_id: str, markets: Optional[List[str]] = None) -> Dict:
    """
    Fetch player prop odds for a specific game.

    Args:
        event_id: The-Odds-API event ID
        markets: List of prop markets to fetch (default: points, rebounds, assists)

    Returns:
        Dict with prop data organized by player and market.
    """
    if markets is None:
        markets = ["player_points", "player_rebounds", "player_assists"]

    data = _request(f"sports/{SPORT}/events/{event_id}/odds", {
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
    })

    if not data:
        return {}

    props_by_player = {}

    for bm in data.get("bookmakers", []):
        book_key = bm["key"]
        book_title = bm["title"]

        for market in bm.get("markets", []):
            market_key = market["key"]

            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "Unknown")
                line = outcome.get("point")
                price = outcome.get("price")
                side = outcome.get("name", "").lower()  # "Over" or "Under"

                if player_name not in props_by_player:
                    props_by_player[player_name] = {}

                if market_key not in props_by_player[player_name]:
                    props_by_player[player_name][market_key] = {
                        "line": line,
                        "books": {},
                    }

                if book_key not in props_by_player[player_name][market_key]["books"]:
                    props_by_player[player_name][market_key]["books"][book_key] = {
                        "title": book_title,
                        "line": line,
                    }

                props_by_player[player_name][market_key]["books"][book_key][side] = price
                # Update line from the preferred book if available
                if book_key in PREFERRED_BOOKS:
                    props_by_player[player_name][market_key]["line"] = line

    return props_by_player


def get_best_odds(game_odds: List[Dict]) -> List[Dict]:
    """
    Extract the best available odds across all bookmakers for each game.

    Returns simplified game data with consensus/best lines.
    """
    results = []

    for game in game_odds:
        best = {
            "id": game["id"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "commence_time": game["commence_time"],
            "spread": None,
            "spread_odds_home": None,
            "spread_odds_away": None,
            "total": None,
            "over_odds": None,
            "under_odds": None,
            "home_ml": None,
            "away_ml": None,
            "book_source": None,
        }

        # Try preferred books first, fall back to any available
        for book_key in PREFERRED_BOOKS + list(game["bookmakers"].keys()):
            if book_key not in game["bookmakers"]:
                continue

            bm = game["bookmakers"][book_key]
            markets = bm["markets"]

            # Spread
            if best["spread"] is None and "spreads" in markets:
                spreads = markets["spreads"]
                home = spreads.get(game["home_team"], {})
                away = spreads.get(game["away_team"], {})
                if home.get("point") is not None:
                    best["spread"] = home["point"]
                    best["spread_odds_home"] = home.get("price", -110)
                    best["spread_odds_away"] = away.get("price", -110)
                    best["book_source"] = bm["title"]

            # Totals
            if best["total"] is None and "totals" in markets:
                totals = markets["totals"]
                over = totals.get("Over", {})
                under = totals.get("Under", {})
                if over.get("point") is not None:
                    best["total"] = over["point"]
                    best["over_odds"] = over.get("price", -110)
                    best["under_odds"] = under.get("price", -110)

            # Moneyline
            if best["home_ml"] is None and "h2h" in markets:
                h2h = markets["h2h"]
                home = h2h.get(game["home_team"], {})
                away = h2h.get(game["away_team"], {})
                if home.get("price") is not None:
                    best["home_ml"] = home["price"]
                    best["away_ml"] = away.get("price")

        results.append(best)

    return results


def format_prop_summary(props_by_player: Dict) -> str:
    """Format player props into a readable summary string."""
    lines = []
    for player, markets in sorted(props_by_player.items()):
        parts = []
        for market_key, data in markets.items():
            stat_name = market_key.replace("player_", "").upper()
            line_val = data.get("line", "?")
            parts.append(f"{stat_name} {line_val}")
        lines.append(f"  {player}: {', '.join(parts)}")
    return "\n".join(lines)
