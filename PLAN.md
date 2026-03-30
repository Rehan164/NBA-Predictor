# NBA Predictor Upgrade Plan: 58% → 62-65% Accuracy

## Phase 1: Feature Engineering Upgrades (build_features.py)
**Impact: +2-3% accuracy | Effort: Medium**

### 1A. ELO Power Ratings
Add dynamic team strength ratings that update after every game.

**New function: `compute_elo_ratings(df)`**
- Initialize all teams at ELO 1500
- After each game, update both teams' ELO based on result + margin of victory
- K-factor: 20, with MOV multiplier = ln(abs(margin) + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))
- Season reset: regress 1/3 toward 1500 at start of each season
- Output: dict mapping (team, date) → elo rating

**Features added to each game:**
- `home_elo`, `away_elo` - Raw ratings
- `elo_diff` - home_elo - away_elo
- `elo_expected` - Expected win probability from ELO

### 1B. Four Factors of Basketball
Compute the "Four Factors" from raw box score data (already in CSV).

**Computed per game in `build_team_game_history()`:**
- `efg_pct` = (FGM + 0.5 * FG3M) / FGA
- `tov_pct` = TOV / (FGA + 0.44 * FTA + TOV)
- `oreb_pct` = OREB / (OREB + Opponent_DREB)
- `ft_rate` = FTM / FGA
- `pace` = FGA + 0.44 * FTA - OREB + TOV (estimated possessions)
- `off_rating` = PTS / pace * 100 (points per 100 possessions)
- `def_rating` = OPP_PTS / pace * 100
- `net_rating` = off_rating - def_rating

**Rolling averages added via `calculate_rolling_stats()`:**
- All four factors + pace/ratings at windows [3, 5, 10, 15, 20]
- Differential features: `efg_pct_diff_l10`, `net_rating_diff_l10`, etc.

### 1C. After Phase 1
- Run: `python -m nba_ml.build_features` (rebuild features with ELO + Four Factors)
- Run: `python -m nba_ml.train_models` (retrain with new features)
- Verify accuracy improvement

---

## Phase 2: Ensemble Stacking (train_models.py)
**Impact: +2-3% accuracy | Effort: Medium**

### 2A. Add LightGBM + CatBoost Base Models
**Install:** `pip install lightgbm catboost`

For each target (home_win, home_margin, total_score):
1. Train XGBoost (existing)
2. Train LightGBM (same Optuna framework, different hyperparameter space)
3. Train CatBoost (same framework, different hyperparameter space)

Each uses same TimeSeriesSplit CV and sample weights.

### 2B. Meta-Learner Stacking
**New function: `train_stacked_ensemble()`**

1. Generate out-of-fold predictions from each base model using TimeSeriesSplit
2. Stack predictions as meta-features: [xgb_pred, lgb_pred, cat_pred]
3. Train a simple meta-learner:
   - Classification (ML): Logistic Regression
   - Regression (spread/totals): Ridge Regression
4. Final prediction = meta_learner.predict([xgb_pred, lgb_pred, cat_pred])

### 2C. Model Save Format Update
Save ensemble as:
```python
{
    "models": {"xgb": model, "lgb": model, "cat": model},
    "meta_learner": meta_model,
    "feature_columns": [...],
    "metrics": {...},
    "model_type": "ensemble_classifier" or "ensemble_regressor",
}
```

### 2D. Update predict.py
- Load ensemble model
- Run all 3 base models, combine via meta-learner
- Backward compatible: if old single-model format, use as before

### 2E. Reduce Optuna Trials Per Model
With 3 base models, reduce from 150 to 80 trials each (25 QMC + 40 TPE + 15 CMA-ES) to keep training time manageable.

---

## Phase 3: LSTM Neural Network (neural_model.py - NEW)
**Impact: +1-2% accuracy | Effort: High**

### 3A. Install Dependencies
`pip install torch` (PyTorch)

### 3B. New File: `nba_ml/neural_model.py`

**Sequence Builder:**
- For each game, build sequences: last 10 games for home team + last 10 games for away team
- Features per game in sequence: margin, efg_pct, tov_pct, oreb_pct, ft_rate, pace, off_rating, def_rating, elo, rest_days, is_home, won (12 features × 10 games × 2 teams)

**Model Architecture:**
```
Home Sequence (10 × 12) → LSTM(64) → h_home
Away Sequence (10 × 12) → LSTM(64) → h_away  (shared weights)
[h_home, h_away, matchup_features] → Dense(128) → Dense(64) → Output
```

**Three output heads:**
1. Win probability (sigmoid)
2. Predicted margin (linear)
3. Predicted total (linear)

**Training:**
- Adam optimizer, lr=0.001 with cosine annealing
- Batch size 64, 50 epochs with early stopping (patience 10)
- Same chronological split as XGBoost
- Dropout 0.3 for regularization

### 3C. Integration with Ensemble
- LSTM predictions become additional meta-features in the stacking layer
- Meta-learner now takes [xgb_pred, lgb_pred, cat_pred, lstm_pred]

---

## Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `nba_ml/build_features.py` | 1 | Add ELO computation, Four Factors, pace/ratings |
| `nba_ml/config.py` | 1,2 | Add ELO config, reduced Optuna trials |
| `nba_ml/train_models.py` | 2 | Add LightGBM, CatBoost, stacking meta-learner |
| `nba_ml/predict.py` | 2,3 | Load ensemble, combine predictions |
| `nba_ml/neural_model.py` | 3 | NEW - LSTM model + training |

## Dependencies to Install
```
pip install lightgbm catboost torch
```

## Execution Order
1. `pip install lightgbm catboost torch`
2. Phase 1: Edit build_features.py → rebuild features → retrain → verify
3. Phase 2: Edit train_models.py → retrain ensemble → verify
4. Phase 3: Create neural_model.py → train LSTM → integrate → verify
