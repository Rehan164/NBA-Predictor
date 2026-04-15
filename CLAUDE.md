# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

**Web dashboard:**
```bash
python -m web.app
# http://localhost:5000
```

**Data collection (one-time setup or refresh):**
```bash
python -m nba_ml.collect_data                  # Fetch historical games (2000–present)
python -m nba_ml.collect_players               # Fetch player game logs (full history)
python -m nba_ml.collect_players --update      # Incremental update for current season
```

**Model training** happens inside `nba_ml/advanced_model.py` — models are saved to `models/advanced/` and loaded at pick time. No separate training command; the module handles both training and inference.

**Hyperparameter tuning** (find params that close the train/test accuracy gap):
```bash
# Fast search — architecture + regularization + split (1 model, 15 epochs)
python -m nba_ml.tune --trials 30 --ensemble-size 1 --max-epochs 15

# Balanced — all fast params (no Phase 1 re-run)
python -m nba_ml.tune --trials 60 --ensemble-size 2 --max-epochs 20

# Deep — also tune tracker alphas / ELO (re-runs Phase 1 each trial, slow)
python -m nba_ml.tune --trials 40 --tune-features --max-epochs 20

# Resume / extend a previous study
python -m nba_ml.tune --trials 50 --study-name nba_v1 --storage sqlite:///tuning.db
```
Best params are saved to `models/advanced/best_params.json`.

## Architecture Overview

This is a daily NBA betting dashboard combining ensemble neural networks with real-time game, odds, and injury data.

### Data Flow (Web Dashboard)

The dashboard fetches games, odds, injuries, and ML predictions from the `picks/` modules and serves them via Flask API routes.

### Key Modules

| Module | Role |
|--------|------|
| `nba_ml/advanced_model.py` | Ensemble of 5 PyTorch neural networks (spread, total, moneyline). Uses Optuna tuning (25 QMC + 40 TPE + 15 CMA-ES trials) and 7-fold CV with sample decay weighting. |
| `nba_ml/player_props_nn.py` | Separate deep NN ensemble for PTS/REB/AST player prop predictions. |
| `nba_ml/collect_data.py` | Fetches 25 years of NBA game results + advanced stats via `nba_api`. |
| `nba_ml/collect_players.py` | Fetches player-level game logs. Outputs `data/nba_player_game_logs.csv` (~72 MB) and `data/nba_player_features.csv` (~588 MB). |
| `nba_ml/config.py` | Central config: file paths, ELO parameters (K=20, home=50pts, reversion=0.33), rolling windows [3,5,10,15,20], training hyperparameters. |
| `nba_ml/injuries.py` | Manual impact multipliers for star players (LeBron, Jokić, etc.). |
| `picks/patterns.py` | Builds/queries the injury pattern cache: "when Player X is OUT, Player Y averages +N pts vs baseline." |
| `picks/model_predictions.py` | Loads trained models, engineers matchup features at pick time, returns structured predictions. |
| `picks/odds_api.py` | The-Odds-API client for spreads, totals, and player prop lines. |
| `picks/injury_report.py` | Enhanced injury fetcher with severity classification. |
| `web/app.py` | Flask dashboard — run with `python -m web.app`. |

### Data Storage

All data is CSV/pickle-based (no SQL database):

- `data/nba_historical_games.csv` — 25 years of game results + advanced stats (~7 MB)
- `data/nba_training_features.csv` — engineered features for model training (~185 MB)
- `data/nba_player_game_logs.csv` — raw player logs (~72 MB)
- `data/nba_player_features.csv` — derived player features (~588 MB)
- `data/injury_patterns_cache.json` — pre-computed injury impact patterns
- `models/advanced/model.pt` — ensemble NN weights
- `models/advanced/meta.pkl` — feature names + scaler
- `models/advanced/elo.pkl` — team ELO ratings
- `models/advanced/rest.pkl` — team rest-day tracking
- `models/player_props_nn/model.pt` — player prop NN weights

### Environment Variables

Required in `.env`:
```
ODDS_API_KEY=...        # Optional; ESPN odds used as fallback if missing
```

## Dependencies

Core: `torch`, `flask`, `nba_api`, `pandas`, `numpy`, `optuna`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `python-dotenv`

Install: `pip install -r requirements.txt`
