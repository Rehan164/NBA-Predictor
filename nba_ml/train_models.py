"""
Model Training Pipeline for NBA Betting ML

Ensemble Stacking Strategy:
  Base models: XGBoost, LightGBM, CatBoost (each tuned with 3-phase Optuna)
  Meta-learner: Logistic Regression (classifier) / Ridge (regressor)
  Stacking: Out-of-fold predictions from TimeSeriesSplit

3-Phase Optuna per base model:
  Phase 1 - QMC Exploration:  Wide ranges, space-filling search
  Phase 2 - TPE Exploitation: Bayesian optimization around best regions
  Phase 3 - CMA-ES Polish:    Evolution strategy for final fine-tuning

Models:
  - Moneyline: Ensemble Classifier (home win probability)
  - Spread:    Ensemble Regressor  (predict home margin -> compare to spread)
  - Totals:    Ensemble Regressor  (predict total score -> compare to O/U)

Usage:
    python -m nba_ml.train_models
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import optuna
from optuna.samplers import TPESampler, QMCSampler, CmaEsSampler

from .config import (
    TRAINING_FEATURES_CSV,
    MODELS_DIR,
    TRAIN_END_DATE,
    TEST_START_DATE,
    CV_FOLDS,
    RANDOM_STATE,
    OPTUNA_PHASE1_TRIALS,
    OPTUNA_PHASE2_TRIALS,
    OPTUNA_PHASE3_TRIALS,
    EARLY_STOPPING_ROUNDS,
    USE_SAMPLE_WEIGHTS,
    WEIGHT_DECAY,
    BREAK_EVEN_PCT,
    USE_FEATURE_SELECTION,
    FEATURE_IMPORTANCE_THRESHOLD,
)


EXCLUDE_COLS = [
    "game_id", "date", "season", "home_team", "away_team",
    "home_score", "away_score", "home_win", "total_score", "home_margin",
    "spread_line", "total_line", "home_cover", "total_over",
    "home_ml", "away_ml",
    "home_cover_proxy", "total_over_proxy",
    "total_deviation",
]


# ===================================================================
# DATA LOADING
# ===================================================================

def load_and_prepare_data():
    print(f"Loading features from {TRAINING_FEATURES_CSV}...")

    if not TRAINING_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Features file not found: {TRAINING_FEATURES_CSV}\n"
            "Run 'python -m nba_ml.build_features' first."
        )

    df = pd.read_csv(TRAINING_FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Loaded {len(df):,} games")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    train_df = df[df["date"] < TRAIN_END_DATE].copy()
    test_df = df[df["date"] >= TEST_START_DATE].copy()

    print(f"\nTrain set: {len(train_df):,} games (before {TRAIN_END_DATE})")
    print(f"Test set:  {len(test_df):,} games (after {TEST_START_DATE})")

    return train_df, test_df


def get_feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and not df[c].isna().all()]


def prepare_xy(df, target, feature_cols):
    valid_df = df.dropna(subset=[target])
    X = valid_df[feature_cols].copy().fillna(0)
    y = valid_df[target].copy()
    return X, y


def compute_sample_weights(X_train):
    """Exponential decay weights: recent games matter more."""
    if not USE_SAMPLE_WEIGHTS:
        return None
    n = len(X_train)
    weights = np.array([WEIGHT_DECAY ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.mean()
    return weights


def select_features(X_train, y_train, feature_cols, task="classification"):
    print(f"\n  Feature selection on {len(feature_cols)} features...")

    if task == "classification":
        quick = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, tree_method="hist",
        )
    else:
        quick = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, tree_method="hist",
        )

    weights = compute_sample_weights(X_train)
    quick.fit(X_train, y_train, sample_weight=weights, verbose=False)

    importances = pd.Series(quick.feature_importances_, index=feature_cols)
    selected = importances[importances >= FEATURE_IMPORTANCE_THRESHOLD].index.tolist()

    dropped = len(feature_cols) - len(selected)
    print(f"    Dropped {dropped} low-importance features, keeping {len(selected)}")

    if len(selected) < 10:
        print("    Too few - keeping all")
        return feature_cols

    return selected


def get_feature_importance(model, feature_names):
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


def save_model(model_data, model_name):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to: {model_path}")


# ===================================================================
# MULTI-PHASE OPTUNA
# ===================================================================

def run_multiphase_optuna(objective_fn, direction="maximize"):
    """
    3-phase Optuna optimization:
      Phase 1 - QMC:   Broad space-filling exploration
      Phase 2 - TPE:   Bayesian optimization (exploit promising regions)
      Phase 3 - CMA-ES: Fine-tune continuous params around the best
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n  Phase 1: QMC Exploration ({OPTUNA_PHASE1_TRIALS} trials)...")
    study = optuna.create_study(
        direction=direction,
        sampler=QMCSampler(seed=RANDOM_STATE),
    )
    study.optimize(objective_fn, n_trials=OPTUNA_PHASE1_TRIALS, show_progress_bar=True)
    best_after_p1 = study.best_value
    print(f"    Best after Phase 1: {best_after_p1:.4f}")

    print(f"\n  Phase 2: TPE Exploitation ({OPTUNA_PHASE2_TRIALS} trials)...")
    study.sampler = TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=10,
        multivariate=True,
    )
    study.optimize(objective_fn, n_trials=OPTUNA_PHASE2_TRIALS, show_progress_bar=True)
    best_after_p2 = study.best_value
    improvement = best_after_p2 - best_after_p1
    print(f"    Best after Phase 2: {best_after_p2:.4f} ({improvement:+.4f} vs Phase 1)")

    print(f"\n  Phase 3: CMA-ES Polish ({OPTUNA_PHASE3_TRIALS} trials)...")
    study.sampler = CmaEsSampler(
        seed=RANDOM_STATE,
        n_startup_trials=0,
    )
    study.optimize(objective_fn, n_trials=OPTUNA_PHASE3_TRIALS, show_progress_bar=True)
    best_final = study.best_value
    total_improvement = best_final - best_after_p1
    print(f"    Best final: {best_final:.4f} ({total_improvement:+.4f} total improvement)")

    total_trials = len(study.trials)
    print(f"\n  Total trials: {total_trials}, Best: {best_final:.4f}")

    return study


# ===================================================================
# OBJECTIVE FUNCTIONS FOR EACH BASE MODEL
# ===================================================================

def make_xgb_classifier_objective(X_train, y_train, cv_folds, sample_weights):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        }
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
            scores.append(accuracy_score(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective


def make_lgb_classifier_objective(X_train, y_train, cv_folds, sample_weights):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 5),
        }
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(-1), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            scores.append(accuracy_score(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective


def make_cat_classifier_objective(X_train, y_train, cv_folds, sample_weights):
    def objective(trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "bootstrap_type": "Bernoulli",
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "random_strength": trial.suggest_float("random_strength", 0, 10),
        }
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            pool_tr = cb.Pool(X_tr, y_tr, weight=w_tr)
            pool_val = cb.Pool(X_val, y_val)
            model = cb.CatBoostClassifier(**params)
            model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            scores.append(accuracy_score(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective


def make_xgb_regressor_objective(X_train, y_train, cv_folds, sample_weights):
    def objective(trial):
        loss = trial.suggest_categorical("loss_fn", ["reg:squarederror", "reg:pseudohubererror"])
        params = {
            "objective": loss,
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        }
        if loss == "reg:pseudohubererror":
            params["huber_slope"] = trial.suggest_float("huber_slope", 0.5, 5.0)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
            scores.append(-mean_absolute_error(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective


def make_lgb_regressor_objective(X_train, y_train, cv_folds, sample_weights):
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(-1), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            scores.append(-mean_absolute_error(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective


def make_cat_regressor_objective(X_train, y_train, cv_folds, sample_weights):
    def objective(trial):
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "bootstrap_type": "Bernoulli",
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "random_strength": trial.suggest_float("random_strength", 0, 10),
        }
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            pool_tr = cb.Pool(X_tr, y_tr, weight=w_tr)
            pool_val = cb.Pool(X_val, y_val)
            model = cb.CatBoostRegressor(**params)
            model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            scores.append(-mean_absolute_error(y_val, model.predict(X_val)))
        return np.mean(scores)
    return objective


# ===================================================================
# TRAIN INDIVIDUAL BASE MODELS
# ===================================================================

def train_xgb_final(study, X_train, y_train, weights, task="classifier"):
    """Train final XGBoost model with best Optuna params + early stopping."""
    best_p = study.best_params.copy()
    loss = best_p.pop("loss_fn", None)

    if task == "classifier":
        base_params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        base_params = {"objective": loss or "reg:squarederror", "eval_metric": "mae"}

    params = {**base_params, "tree_method": "hist", "random_state": RANDOM_STATE, **best_p}
    params["n_estimators"] = max(params.get("n_estimators", 500), 1500)

    split_idx = int(len(X_train) * 0.85)
    X_fit, X_hold = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_hold = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    w_fit = weights[:split_idx] if weights is not None else None

    ModelClass = xgb.XGBClassifier if task == "classifier" else xgb.XGBRegressor
    model = ModelClass(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_hold, y_hold)], verbose=False)

    trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else params["n_estimators"]
    print(f"    XGBoost: {trees} trees")
    return model


def train_lgb_final(study, X_train, y_train, weights, task="classifier"):
    """Train final LightGBM model with best Optuna params + early stopping."""
    params = {**study.best_params, "random_state": RANDOM_STATE, "verbose": -1}

    if task == "classifier":
        params.update({"objective": "binary", "metric": "binary_logloss"})
    else:
        params.update({"objective": "regression", "metric": "mae"})

    params["n_estimators"] = max(params.get("n_estimators", 500), 1500)

    split_idx = int(len(X_train) * 0.85)
    X_fit, X_hold = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_hold = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    w_fit = weights[:split_idx] if weights is not None else None

    ModelClass = lgb.LGBMClassifier if task == "classifier" else lgb.LGBMRegressor
    model = ModelClass(**params)
    model.fit(X_fit, y_fit, sample_weight=w_fit,
              eval_set=[(X_hold, y_hold)], callbacks=[lgb.log_evaluation(-1), lgb.early_stopping(EARLY_STOPPING_ROUNDS)])

    trees = model.best_iteration_ if hasattr(model, 'best_iteration_') else params["n_estimators"]
    print(f"    LightGBM: {trees} trees")
    return model


def train_cat_final(study, X_train, y_train, weights, task="classifier"):
    """Train final CatBoost model with best Optuna params + early stopping."""
    params = {**study.best_params, "random_seed": RANDOM_STATE, "verbose": 0,
              "bootstrap_type": "Bernoulli"}

    if task == "classifier":
        params.update({"loss_function": "Logloss", "eval_metric": "Accuracy"})
    else:
        params.update({"loss_function": "MAE", "eval_metric": "MAE"})

    params["iterations"] = max(params.get("iterations", 500), 1500)

    split_idx = int(len(X_train) * 0.85)
    X_fit, X_hold = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_hold = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    w_fit = weights[:split_idx] if weights is not None else None

    pool_fit = cb.Pool(X_fit, y_fit, weight=w_fit)
    pool_hold = cb.Pool(X_hold, y_hold)

    ModelClass = cb.CatBoostClassifier if task == "classifier" else cb.CatBoostRegressor
    model = ModelClass(**params)
    model.fit(pool_fit, eval_set=pool_hold, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    trees = model.best_iteration_ if hasattr(model, 'best_iteration_') else params["iterations"]
    print(f"    CatBoost: {trees} iterations")
    return model


# ===================================================================
# ENSEMBLE STACKING
# ===================================================================

def generate_oof_predictions(models_and_names, X_train, y_train, cv_folds, weights, task="classifier"):
    """
    Generate out-of-fold predictions for stacking.
    Each base model produces predictions on held-out folds.
    """
    n = len(X_train)
    oof = np.zeros((n, len(models_and_names)))
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    print(f"\n  Generating OOF predictions ({cv_folds} folds)...")

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = weights[train_idx] if weights is not None else None

        for model_idx, (model_class, model_params, name) in enumerate(models_and_names):
            m = model_class(**model_params)

            if "xgb" in name.lower():
                m.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], verbose=False)
            elif "lgb" in name.lower():
                m.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(-1)])
            elif "cat" in name.lower():
                pool_tr = cb.Pool(X_tr, y_tr, weight=w_tr)
                pool_val = cb.Pool(X_val, y_val)
                m.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=30)

            if task == "classifier":
                oof[val_idx, model_idx] = m.predict_proba(X_val)[:, 1]
            else:
                oof[val_idx, model_idx] = m.predict(X_val)

    # Only return rows that were in at least one validation fold
    # First fold's training set is never in validation, so we mask those
    first_val_start = 0
    for train_idx, val_idx in TimeSeriesSplit(n_splits=cv_folds).split(X_train):
        first_val_start = val_idx[0]
        break

    return oof, first_val_start


def train_meta_learner(oof_preds, y_train, first_val_start, task="classifier"):
    """Train a simple meta-learner on out-of-fold predictions."""
    # Use only rows that have valid OOF predictions
    X_meta = oof_preds[first_val_start:]
    y_meta = y_train.iloc[first_val_start:]

    if task == "classifier":
        meta = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    else:
        meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)

    meta.fit(X_meta, y_meta)

    # Show meta-learner weights
    if hasattr(meta, 'coef_'):
        coefs = meta.coef_.flatten()
        names = ["XGBoost", "LightGBM", "CatBoost"]
        print(f"  Meta-learner weights: {', '.join(f'{n}: {c:.3f}' for n, c in zip(names, coefs))}")

    return meta


# ===================================================================
# ENSEMBLE TRAINING ORCHESTRATION
# ===================================================================

def train_ensemble_classifier(X_train, y_train, feature_cols, model_name):
    """Train a stacked ensemble classifier (XGB + LGB + CAT -> LogReg)."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Ensemble Classification)")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")

    weights = compute_sample_weights(X_train)

    # --- XGBoost ---
    print(f"\n  --- XGBoost ---")
    xgb_obj = make_xgb_classifier_objective(X_train, y_train, CV_FOLDS, weights)
    xgb_study = run_multiphase_optuna(xgb_obj, direction="maximize")
    xgb_model = train_xgb_final(xgb_study, X_train, y_train, weights, "classifier")

    # --- LightGBM ---
    print(f"\n  --- LightGBM ---")
    lgb_obj = make_lgb_classifier_objective(X_train, y_train, CV_FOLDS, weights)
    lgb_study = run_multiphase_optuna(lgb_obj, direction="maximize")
    lgb_model = train_lgb_final(lgb_study, X_train, y_train, weights, "classifier")

    # --- CatBoost ---
    print(f"\n  --- CatBoost ---")
    cat_obj = make_cat_classifier_objective(X_train, y_train, CV_FOLDS, weights)
    cat_study = run_multiphase_optuna(cat_obj, direction="maximize")
    cat_model = train_cat_final(cat_study, X_train, y_train, weights, "classifier")

    # --- Stacking ---
    print(f"\n  --- Meta-Learner Stacking ---")

    # Build OOF models config (lighter versions for OOF generation)
    oof_models = [
        (xgb.XGBClassifier, {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "random_state": RANDOM_STATE,
            **{k: v for k, v in xgb_study.best_params.items()},
        }, "xgb"),
        (lgb.LGBMClassifier, {
            "objective": "binary", "metric": "binary_logloss",
            "random_state": RANDOM_STATE, "verbose": -1,
            **lgb_study.best_params,
        }, "lgb"),
        (cb.CatBoostClassifier, {
            "loss_function": "Logloss", "eval_metric": "Accuracy",
            "random_seed": RANDOM_STATE, "verbose": 0,
            "bootstrap_type": "Bernoulli",
            **cat_study.best_params,
        }, "cat"),
    ]

    oof, first_val = generate_oof_predictions(oof_models, X_train, y_train, CV_FOLDS, weights, "classifier")
    meta = train_meta_learner(oof, y_train, first_val, "classifier")

    # --- Isotonic Probability Calibration ---
    print(f"\n  --- Probability Calibration (Isotonic) ---")
    # Use OOF meta-learner predictions as calibration input
    X_meta = oof[first_val:]
    y_meta = y_train.iloc[first_val:].values

    raw_probs = meta.predict_proba(X_meta)[:, 1]
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(raw_probs, y_meta)

    cal_probs = calibrator.predict(raw_probs)
    raw_acc = accuracy_score(y_meta, (raw_probs > 0.5).astype(int))
    cal_acc = accuracy_score(y_meta, (cal_probs > 0.5).astype(int))
    print(f"  Raw accuracy: {raw_acc:.4f}, Calibrated accuracy: {cal_acc:.4f}")
    print(f"  Raw mean prob: {raw_probs.mean():.4f}, Calibrated mean: {cal_probs.mean():.4f}")

    return {
        "models": {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model},
        "meta_learner": meta,
        "calibrator": calibrator,
        "feature_columns": feature_cols,
        "model_type": "ensemble_classifier",
        "params": {
            "xgb": xgb_study.best_params,
            "lgb": lgb_study.best_params,
            "cat": cat_study.best_params,
        },
    }


def train_ensemble_regressor(X_train, y_train, feature_cols, model_name):
    """Train a stacked ensemble regressor (XGB + LGB + CAT -> Ridge)."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Ensemble Regression)")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")
    print(f"  Target - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    weights = compute_sample_weights(X_train)

    # --- XGBoost ---
    print(f"\n  --- XGBoost ---")
    xgb_obj = make_xgb_regressor_objective(X_train, y_train, CV_FOLDS, weights)
    xgb_study = run_multiphase_optuna(xgb_obj, direction="maximize")
    xgb_model = train_xgb_final(xgb_study, X_train, y_train, weights, "regressor")

    # --- LightGBM ---
    print(f"\n  --- LightGBM ---")
    lgb_obj = make_lgb_regressor_objective(X_train, y_train, CV_FOLDS, weights)
    lgb_study = run_multiphase_optuna(lgb_obj, direction="maximize")
    lgb_model = train_lgb_final(lgb_study, X_train, y_train, weights, "regressor")

    # --- CatBoost ---
    print(f"\n  --- CatBoost ---")
    cat_obj = make_cat_regressor_objective(X_train, y_train, CV_FOLDS, weights)
    cat_study = run_multiphase_optuna(cat_obj, direction="maximize")
    cat_model = train_cat_final(cat_study, X_train, y_train, weights, "regressor")

    # --- Stacking ---
    print(f"\n  --- Meta-Learner Stacking ---")

    xgb_best = xgb_study.best_params.copy()
    xgb_loss = xgb_best.pop("loss_fn", "reg:squarederror")

    oof_models = [
        (xgb.XGBRegressor, {
            "objective": xgb_loss, "eval_metric": "mae",
            "tree_method": "hist", "random_state": RANDOM_STATE,
            **xgb_best,
        }, "xgb"),
        (lgb.LGBMRegressor, {
            "objective": "regression", "metric": "mae",
            "random_state": RANDOM_STATE, "verbose": -1,
            **lgb_study.best_params,
        }, "lgb"),
        (cb.CatBoostRegressor, {
            "loss_function": "MAE", "eval_metric": "MAE",
            "random_seed": RANDOM_STATE, "verbose": 0,
            "bootstrap_type": "Bernoulli",
            **cat_study.best_params,
        }, "cat"),
    ]

    oof, first_val = generate_oof_predictions(oof_models, X_train, y_train, CV_FOLDS, weights, "regressor")
    meta = train_meta_learner(oof, y_train, first_val, "regressor")

    return {
        "models": {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model},
        "meta_learner": meta,
        "feature_columns": feature_cols,
        "model_type": "ensemble_regressor",
        "params": {
            "xgb": xgb_study.best_params,
            "lgb": lgb_study.best_params,
            "cat": cat_study.best_params,
        },
    }


# ===================================================================
# ENSEMBLE PREDICTION
# ===================================================================

def ensemble_predict_proba(ensemble_data, X):
    """Get probability predictions from ensemble classifier."""
    models = ensemble_data["models"]
    meta = ensemble_data["meta_learner"]

    base_preds = np.column_stack([
        models["xgb"].predict_proba(X)[:, 1],
        models["lgb"].predict_proba(X)[:, 1],
        models["cat"].predict_proba(X)[:, 1],
    ])

    return meta.predict_proba(base_preds)[:, 1]


def ensemble_predict(ensemble_data, X):
    """Get predictions from ensemble regressor."""
    models = ensemble_data["models"]
    meta = ensemble_data["meta_learner"]

    base_preds = np.column_stack([
        models["xgb"].predict(X),
        models["lgb"].predict(X),
        models["cat"].predict(X),
    ])

    return meta.predict(base_preds)


# ===================================================================
# EVALUATION
# ===================================================================

def evaluate_ensemble_classifier(ensemble_data, X_test, y_test, model_name):
    y_prob = ensemble_predict_proba(ensemble_data, X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # Also show individual model accuracies
    models = ensemble_data["models"]
    print(f"\n  {model_name} - Individual Model Accuracies:")
    for name, m in models.items():
        ind_pred = m.predict(X_test)
        ind_acc = accuracy_score(y_test, ind_pred)
        print(f"    {name:10}: {ind_acc:.4f}")

    print(f"\n  {model_name} ENSEMBLE Test Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")

    if metrics["accuracy"] > BREAK_EVEN_PCT:
        roi = (metrics["accuracy"] - BREAK_EVEN_PCT) / BREAK_EVEN_PCT * 100
        print(f"    ROI estimate: +{roi:.1f}%")
    else:
        print(f"    Below break-even ({BREAK_EVEN_PCT:.2%} needed at -110)")

    return metrics


def evaluate_ensemble_regressor(ensemble_data, X_test, y_test, model_name, target_name):
    y_pred = ensemble_predict(ensemble_data, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    # Individual model MAEs
    models = ensemble_data["models"]
    print(f"\n  {model_name} - Individual Model MAEs:")
    for name, m in models.items():
        ind_pred = m.predict(X_test)
        ind_mae = mean_absolute_error(y_test, ind_pred)
        print(f"    {name:10}: {ind_mae:.2f}")

    print(f"\n  {model_name} ENSEMBLE Test Results:")
    print(f"    MAE:   {mae:.2f} points")
    print(f"    RMSE:  {rmse:.2f} points")
    print(f"    R2:    {r2:.4f}")

    if target_name == "home_margin":
        dir_acc = ((y_pred > 0) == (y_test > 0)).mean()
        metrics["direction_accuracy"] = dir_acc
        print(f"    Winner prediction: {dir_acc:.2%}")

        for edge in [2, 3, 5, 7]:
            mask = np.abs(y_pred) >= edge
            if mask.sum() > 10:
                acc = ((y_pred[mask] > 0) == (y_test.values[mask] > 0)).mean()
                print(f"    |margin| >= {edge}: {acc:.2%} ({mask.sum()} games)")

    elif target_name == "total_score":
        median = y_test.median()
        dir_acc = ((y_pred > median) == (y_test > median)).mean()
        metrics["direction_accuracy"] = dir_acc
        print(f"    O/U median ({median:.0f}): {dir_acc:.2%}")

        for edge in [3, 5, 8, 10]:
            mask = np.abs(y_pred - median) >= edge
            if mask.sum() > 10:
                acc = ((y_pred[mask] > median) == (y_test.values[mask] > median)).mean()
                print(f"    |edge| >= {edge}: {acc:.2%} ({mask.sum()} games)")

    return metrics


def simulate_ensemble_betting(ensemble_data, X_test, y_test, threshold, task, target_name):
    if task == "classifier":
        y_prob = ensemble_predict_proba(ensemble_data, X_test)
        mask = (y_prob >= threshold) | (y_prob <= (1 - threshold))
        if not mask.any():
            return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_units": 0.0, "roi": 0.0}
        y_pred_conf = (y_prob[mask] >= 0.5).astype(int)
        y_true_conf = y_test.iloc[mask].values
    else:
        y_pred = ensemble_predict(ensemble_data, X_test)
        if target_name == "home_margin":
            mask = np.abs(y_pred) >= threshold
            if not mask.any():
                return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_units": 0.0, "roi": 0.0}
            y_pred_conf = (y_pred[mask] > 0).astype(int)
            y_true_conf = (y_test.values[mask] > 0).astype(int)
        else:
            median = y_test.median()
            mask = np.abs(y_pred - median) >= threshold
            if not mask.any():
                return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_units": 0.0, "roi": 0.0}
            y_pred_conf = (y_pred[mask] > median).astype(int)
            y_true_conf = (y_test.values[mask] > median).astype(int)

    wins = (y_pred_conf == y_true_conf).sum()
    total = len(y_pred_conf)
    win_rate = wins / total if total > 0 else 0
    profit = wins * 100 - (total - wins) * 110
    roi = profit / (total * 110) * 100 if total > 0 else 0

    return {
        "bets": total, "wins": wins, "losses": total - wins,
        "win_rate": win_rate, "profit_units": profit / 110, "roi": roi,
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("NBA ML Ensemble Training Pipeline")
    print("  Base: XGBoost + LightGBM + CatBoost")
    print("  Meta: Stacked Logistic/Ridge Regression")
    print("  Tuning: 3-Phase Optuna (QMC -> TPE -> CMA-ES)")
    print("=" * 60)

    train_df, test_df = load_and_prepare_data()
    feature_cols = get_feature_columns(train_df)
    print(f"\nTotal available features: {len(feature_cols)}")

    results = {}

    # ----------------------------------------------------------------
    # 1. MONEYLINE MODEL (Ensemble Classification: home_win)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 1: MONEYLINE (Ensemble Classification)")
    print("=" * 60)

    X_train, y_train = prepare_xy(train_df, "home_win", feature_cols)
    X_test, y_test = prepare_xy(test_df, "home_win", feature_cols)

    if USE_FEATURE_SELECTION:
        ml_cols = select_features(X_train, y_train, feature_cols, "classification")
        X_train_ml, X_test_ml = X_train[ml_cols], X_test[ml_cols]
    else:
        ml_cols = feature_cols
        X_train_ml, X_test_ml = X_train, X_test

    ensemble_ml = train_ensemble_classifier(X_train_ml, y_train, ml_cols, "moneyline_model")
    metrics = evaluate_ensemble_classifier(ensemble_ml, X_test_ml, y_test, "moneyline_model")

    # Show XGBoost feature importance (most interpretable)
    print(f"\n  Top 10 Features (XGBoost):")
    print(get_feature_importance(ensemble_ml["models"]["xgb"], ml_cols).head(10).to_string(index=False))

    sim = simulate_ensemble_betting(ensemble_ml, X_test_ml, y_test, 0.55, "classifier", "home_win")
    print(f"\n  Betting Sim (55%+ conf): {sim['bets']} bets, "
          f"{sim['win_rate']:.2%} win rate, {sim['roi']:+.1f}% ROI")

    ensemble_ml["metrics"] = metrics
    ensemble_ml["trained_at"] = datetime.now().isoformat()
    save_model(ensemble_ml, "moneyline_model")
    results["moneyline_model"] = metrics

    # ----------------------------------------------------------------
    # 2. SPREAD MODEL (Ensemble Regression: home_margin)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 2: SPREAD (Ensemble Regression)")
    print("=" * 60)

    X_train_s, y_train_s = prepare_xy(train_df, "home_margin", feature_cols)
    X_test_s, y_test_s = prepare_xy(test_df, "home_margin", feature_cols)

    if USE_FEATURE_SELECTION:
        spread_cols = select_features(X_train_s, y_train_s, feature_cols, "regression")
        X_train_s, X_test_s = X_train_s[spread_cols], X_test_s[spread_cols]
    else:
        spread_cols = feature_cols

    ensemble_s = train_ensemble_regressor(X_train_s, y_train_s, spread_cols, "spread_model")
    metrics_s = evaluate_ensemble_regressor(ensemble_s, X_test_s, y_test_s, "spread_model", "home_margin")

    print(f"\n  Top 10 Features (XGBoost):")
    print(get_feature_importance(ensemble_s["models"]["xgb"], spread_cols).head(10).to_string(index=False))

    sim_s = simulate_ensemble_betting(ensemble_s, X_test_s, y_test_s, 3.0, "regressor", "home_margin")
    print(f"\n  Betting Sim (3+ pt edge): {sim_s['bets']} bets, "
          f"{sim_s['win_rate']:.2%} win rate, {sim_s['roi']:+.1f}% ROI")

    ensemble_s["metrics"] = metrics_s
    ensemble_s["trained_at"] = datetime.now().isoformat()
    save_model(ensemble_s, "spread_model")
    results["spread_model"] = metrics_s

    # ----------------------------------------------------------------
    # 3. TOTALS MODEL (Ensemble Regression: total deviation from projected)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 3: TOTALS (Ensemble Regression - Deviation Target)")
    print("=" * 60)

    # Train on deviation from projected total instead of absolute total
    # This normalizes across eras (2000 avg ~195, 2025 avg ~228)
    train_df["total_deviation"] = train_df["total_score"] - train_df["projected_total"].fillna(train_df["total_score"])
    test_df["total_deviation"] = test_df["total_score"] - test_df["projected_total"].fillna(test_df["total_score"])
    print(f"  Target: total_score - projected_total (deviation)")
    print(f"  Train deviation mean: {train_df['total_deviation'].mean():.2f}, std: {train_df['total_deviation'].std():.2f}")

    X_train_t, y_train_t = prepare_xy(train_df, "total_deviation", feature_cols)
    X_test_t, y_test_t = prepare_xy(test_df, "total_deviation", feature_cols)
    # Keep original total_score for evaluation
    y_test_t_original = test_df.loc[y_test_t.index, "total_score"]
    projected_test = test_df.loc[y_test_t.index, "projected_total"].fillna(test_df.loc[y_test_t.index, "total_score"])

    if USE_FEATURE_SELECTION:
        totals_cols = select_features(X_train_t, y_train_t, feature_cols, "regression")
        X_train_t, X_test_t = X_train_t[totals_cols], X_test_t[totals_cols]
    else:
        totals_cols = feature_cols

    ensemble_t = train_ensemble_regressor(X_train_t, y_train_t, totals_cols, "totals_model")

    # Evaluate on deviation first
    metrics_t = evaluate_ensemble_regressor(ensemble_t, X_test_t, y_test_t, "totals_model", "total_deviation")

    # Also evaluate in real total-score space
    y_pred_dev = ensemble_predict(ensemble_t, X_test_t)
    y_pred_total = projected_test.values + y_pred_dev
    total_mae = mean_absolute_error(y_test_t_original, y_pred_total)
    total_rmse = np.sqrt(mean_squared_error(y_test_t_original, y_pred_total))
    total_r2 = r2_score(y_test_t_original, y_pred_total)

    print(f"\n  In real total-score space:")
    print(f"    MAE:  {total_mae:.2f} pts")
    print(f"    RMSE: {total_rmse:.2f} pts")
    print(f"    R2:   {total_r2:.4f}")

    # O/U direction accuracy vs median
    median_total = y_test_t_original.median()
    ou_dir = ((y_pred_total > median_total) == (y_test_t_original.values > median_total)).mean()
    print(f"    O/U vs median ({median_total:.0f}): {ou_dir:.2%}")
    metrics_t["total_mae_real"] = total_mae
    metrics_t["total_r2_real"] = total_r2
    metrics_t["direction_accuracy"] = ou_dir

    print(f"\n  Top 10 Features (XGBoost):")
    print(get_feature_importance(ensemble_t["models"]["xgb"], totals_cols).head(10).to_string(index=False))

    sim_t = simulate_ensemble_betting(ensemble_t, X_test_t, y_test_t, 5.0, "regressor", "total_deviation")
    print(f"\n  Betting Sim (5+ pt edge): {sim_t['bets']} bets, "
          f"{sim_t['win_rate']:.2%} win rate, {sim_t['roi']:+.1f}% ROI")

    ensemble_t["metrics"] = metrics_t
    ensemble_t["target_type"] = "deviation"  # Flag for predict.py
    ensemble_t["trained_at"] = datetime.now().isoformat()
    save_model(ensemble_t, "totals_model")
    results["totals_model"] = metrics_t

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    trials_per_model = OPTUNA_PHASE1_TRIALS + OPTUNA_PHASE2_TRIALS + OPTUNA_PHASE3_TRIALS

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Ensemble: XGBoost + LightGBM + CatBoost -> Meta-Learner")
    print(f"  Optuna: {trials_per_model} trials/base model x 3 base models x 3 targets")
    print(f"  Models saved to: {MODELS_DIR}\n")

    for name, m in results.items():
        if "accuracy" in m:
            status = "PROFITABLE" if m["accuracy"] > BREAK_EVEN_PCT else "needs work"
            print(f"  {name:20}: Accuracy {m['accuracy']:.2%} ({status})")
        elif "mae" in m:
            dir_acc = m.get("direction_accuracy", 0)
            print(f"  {name:20}: MAE {m['mae']:.2f} pts, Direction {dir_acc:.2%}")

    print(f"\nNext: python -m nba_ml.predict")


if __name__ == "__main__":
    main()
