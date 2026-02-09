"""
Configuration constants for NBA ML betting system.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

# Output files
HISTORICAL_GAMES_CSV = DATA_DIR / "nba_historical_games.csv"
PLAYER_GAME_LOGS_CSV = DATA_DIR / "nba_player_game_logs.csv"
REFEREE_DATA_CSV = DATA_DIR / "nba_referee_data.csv"
TRAINING_FEATURES_CSV = DATA_DIR / "nba_training_features.csv"
PLAYER_FEATURES_CSV = DATA_DIR / "nba_player_features.csv"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Data Collection ──────────────────────────────────────────────────────────

START_SEASON = 2000  # First season to collect (2000 = 1999-2000 season)
END_SEASON = 2025    # Last season to collect (2025 = 2024-2025 season)

# Rate limiting for NBA API
NBA_API_DELAY = 0.6  # seconds between requests (slightly under 1/sec to be safe)

# ── Feature Engineering ──────────────────────────────────────────────────────

ROLLING_WINDOWS = [3, 5, 10, 15, 20]  # More granular windows for better patterns
MIN_GAMES_FOR_FEATURES = 10   # Minimum games before calculating features

# ── Model Training ───────────────────────────────────────────────────────────

# Chronological train/test split
TRAIN_END_DATE = "2022-07-01"    # Train on data before this date
TEST_START_DATE = "2022-10-01"  # Test on data after this date

# Cross-validation
CV_FOLDS = 7  # More folds for robust validation
RANDOM_STATE = 42

# Optuna multi-phase tuning
# Phase 1: QMC exploration (wide ranges, find promising regions)
# Phase 2: TPE exploitation (narrow around best, Bayesian optimization)
# Phase 3: CMA-ES polish (evolution strategy, fine-tune continuous params)
OPTUNA_PHASE1_TRIALS = 40   # QMC broad exploration
OPTUNA_PHASE2_TRIALS = 80   # TPE focused search
OPTUNA_PHASE3_TRIALS = 30   # CMA-ES final polish
OPTUNA_TRIALS = OPTUNA_PHASE1_TRIALS + OPTUNA_PHASE2_TRIALS + OPTUNA_PHASE3_TRIALS  # Total: 150

# Early stopping (prevents overfitting, allows more trees)
EARLY_STOPPING_ROUNDS = 50

# Sample weighting (recent games matter more)
USE_SAMPLE_WEIGHTS = True
WEIGHT_DECAY = 0.9995  # Per-game decay factor (older games weighted less)

# Feature selection
USE_FEATURE_SELECTION = True
FEATURE_IMPORTANCE_THRESHOLD = 0.001  # Remove features below this importance

# ── Betting Simulation ───────────────────────────────────────────────────────

JUICE = -110  # Standard vig on spread/total bets
BREAK_EVEN_PCT = 0.5238  # Win rate needed to break even at -110

# Minimum confidence to place a bet
MIN_CONFIDENCE_SPREAD = 0.54
MIN_CONFIDENCE_TOTAL = 0.54
MIN_CONFIDENCE_ML = 0.55

# ── Team Mappings ────────────────────────────────────────────────────────────

# Map team abbreviations to full names (for consistency across sources)
TEAM_ABBR_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "NJN": "Brooklyn Nets",  # Old name
    "CHA": "Charlotte Hornets",
    "CHH": "Charlotte Hornets",  # Old abbreviation
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "GS": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "VAN": "Memphis Grizzlies",  # Vancouver
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NO": "New Orleans Pelicans",
    "NOH": "New Orleans Pelicans",  # Hornets era
    "NOK": "New Orleans Pelicans",  # OKC temp
    "NYK": "New York Knicks",
    "NY": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "SEA": "Oklahoma City Thunder",  # Seattle SuperSonics
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "SA": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "UTAH": "Utah Jazz",
    "WAS": "Washington Wizards",
    "WSH": "Washington Wizards",
}

# NBA API team IDs
NBA_API_TEAM_IDS = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BKN": 1610612751,
    "CHA": 1610612766,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NOP": 1610612740,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
}

# ── Team City Coordinates (lat, lon) ───────────────────────────────────────
# Used for calculating travel distance between teams

TEAM_COORDINATES = {
    "ATL": (33.757, -84.396),    # Atlanta
    "BOS": (42.366, -71.062),    # Boston
    "BKN": (40.683, -73.976),    # Brooklyn
    "NJN": (40.734, -74.171),    # New Jersey (old Nets)
    "CHA": (35.225, -80.839),    # Charlotte
    "CHH": (35.225, -80.839),    # Charlotte Hornets (old)
    "CHI": (41.881, -87.674),    # Chicago
    "CLE": (41.496, -81.688),    # Cleveland
    "DAL": (32.790, -96.810),    # Dallas
    "DEN": (39.749, -105.010),   # Denver
    "DET": (42.341, -83.055),    # Detroit
    "GSW": (37.768, -122.388),   # San Francisco
    "GS":  (37.768, -122.388),   # Golden State alias
    "HOU": (29.751, -95.362),    # Houston
    "IND": (39.764, -86.156),    # Indianapolis
    "LAC": (34.043, -118.267),   # Los Angeles (Clippers)
    "LAL": (34.043, -118.267),   # Los Angeles (Lakers)
    "MEM": (35.138, -90.051),    # Memphis
    "VAN": (49.278, -123.109),   # Vancouver (old Grizzlies)
    "MIA": (25.781, -80.187),    # Miami
    "MIL": (43.045, -87.917),    # Milwaukee
    "MIN": (44.980, -93.276),    # Minneapolis
    "NOP": (29.949, -90.082),    # New Orleans
    "NO":  (29.949, -90.082),    # New Orleans alias
    "NOH": (29.949, -90.082),    # New Orleans Hornets
    "NOK": (35.463, -97.515),    # NO/OKC temp (played in OKC)
    "NYK": (40.751, -73.994),    # New York
    "NY":  (40.751, -73.994),    # New York alias
    "OKC": (35.463, -97.515),    # Oklahoma City
    "SEA": (47.622, -122.354),   # Seattle (old Sonics)
    "ORL": (28.539, -81.384),    # Orlando
    "PHI": (39.901, -75.172),    # Philadelphia
    "PHX": (33.446, -112.071),   # Phoenix
    "POR": (45.532, -122.667),   # Portland
    "SAC": (38.580, -121.500),   # Sacramento
    "SAS": (29.427, -98.438),    # San Antonio
    "SA":  (29.427, -98.438),    # San Antonio alias
    "TOR": (43.643, -79.379),    # Toronto
    "UTA": (40.768, -111.901),   # Salt Lake City
    "UTAH":(40.768, -111.901),   # Utah alias
    "WAS": (38.898, -77.021),    # Washington DC
    "WSH": (38.898, -77.021),    # Washington alias
}
