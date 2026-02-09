"""
Model Training Pipeline for NBA Betting ML

3-Phase Optuna Strategy:
  Phase 1 — QMC Exploration:  Wide ranges, space-filling search
  Phase 2 — TPE Exploitation: Bayesian optimization around best regions
  Phase 3 — CMA-ES Polish:    Evolution strategy for final fine-tuning

Models:
  - Moneyline: XGBoost Classifier (home win probability)
  - Spread:    XGBoost Regressor  (predict home margin → compare to spread)
  - Totals:    XGBoost Regressor  (predict total score → compare to O/U)

Accuracy boosters:
  - Early stopping (prevents overfitting, allows 2000+ trees)
  - Sample weights (recent games weighted more via exponential decay)
  - Huber loss for regression (robust to blowout outliers)
  - Multi-phase Optuna (explore then exploit then polish)

Usage:
    python -m nba_ml.train_models
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
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
]


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

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
    # Normalize so mean weight = 1
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
        print("    Too few — keeping all")
        return feature_cols

    return selected


def get_feature_importance(model, feature_names):
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


def save_model(model, model_name, feature_cols, metrics, params, model_type="classifier"):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": feature_cols,
            "metrics": metrics,
            "params": params,
            "model_type": model_type,
            "trained_at": datetime.now().isoformat(),
        }, f)
    print(f"  Saved to: {model_path}")


# ═══════════════════════════════════════════════════════════════════
# MULTI-PHASE OPTUNA
# ═══════════════════════════════════════════════════════════════════

def run_multiphase_optuna(objective_fn, direction="maximize"):
    """
    3-phase Optuna optimization:
      Phase 1 — QMC:   Broad space-filling exploration
      Phase 2 — TPE:   Bayesian optimization (exploit promising regions)
      Phase 3 — CMA-ES: Fine-tune continuous params around the best
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Phase 1: QMC exploration (wide ranges, find promising regions fast)
    print(f"\n  Phase 1: QMC Exploration ({OPTUNA_PHASE1_TRIALS} trials)...")
    study = optuna.create_study(
        direction=direction,
        sampler=QMCSampler(seed=RANDOM_STATE),
    )
    study.optimize(objective_fn, n_trials=OPTUNA_PHASE1_TRIALS, show_progress_bar=True)
    best_after_p1 = study.best_value
    print(f"    Best after Phase 1: {best_after_p1:.4f}")

    # Phase 2: TPE exploitation (Bayesian, focuses on best regions)
    print(f"\n  Phase 2: TPE Exploitation ({OPTUNA_PHASE2_TRIALS} trials)...")
    study.sampler = TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=10,  # Use QMC results to warm-start
        multivariate=True,    # Model parameter correlations
    )
    study.optimize(objective_fn, n_trials=OPTUNA_PHASE2_TRIALS, show_progress_bar=True)
    best_after_p2 = study.best_value
    improvement = best_after_p2 - best_after_p1
    print(f"    Best after Phase 2: {best_after_p2:.4f} ({improvement:+.4f} vs Phase 1)")

    # Phase 3: CMA-ES polish (evolution strategy, fine-tunes continuous params)
    print(f"\n  Phase 3: CMA-ES Polish ({OPTUNA_PHASE3_TRIALS} trials)...")
    study.sampler = CmaEsSampler(
        seed=RANDOM_STATE,
        n_startup_trials=0,  # Use all prior trials
    )
    study.optimize(objective_fn, n_trials=OPTUNA_PHASE3_TRIALS, show_progress_bar=True)
    best_final = study.best_value
    total_improvement = best_final - best_after_p1
    print(f"    Best final: {best_final:.4f} ({total_improvement:+.4f} total improvement)")

    total_trials = len(study.trials)
    print(f"\n  Total trials: {total_trials}, Best: {best_final:.4f}")

    return study


# ═══════════════════════════════════════════════════════════════════
# CLASSIFIER (Moneyline)
# ═══════════════════════════════════════════════════════════════════

def make_classifier_objective(X_train, y_train, cv_folds, sample_weights):
    """Create objective function for classification with early stopping."""

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            # Wide ranges for Phase 1, TPE/CMA-ES will narrow down
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 200, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 200, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.7, 1.3),
            "max_delta_step": trial.suggest_float("max_delta_step", 0, 5),
        }

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Sample weights for this fold
            w_tr = sample_weights[train_idx] if sample_weights is not None else None

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        return np.mean(scores)

    return objective


def train_classifier(X_train, y_train, model_name):
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Classification)")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")

    weights = compute_sample_weights(X_train)
    objective_fn = make_classifier_objective(X_train, y_train, CV_FOLDS, weights)

    study = run_multiphase_optuna(objective_fn, direction="maximize")

    # Train final model with best params + early stopping on holdout
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        **study.best_params,
    }
    # Increase n_estimators and rely on early stopping for the final model
    best_params["n_estimators"] = max(best_params.get("n_estimators", 500), 1500)

    # Split last 15% of training data as early-stopping holdout
    split_idx = int(len(X_train) * 0.85)
    X_fit, X_hold = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_hold = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    w_fit = weights[:split_idx] if weights is not None else None

    final_model = xgb.XGBClassifier(**best_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    final_model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_hold, y_hold)],
        verbose=False,
    )

    actual_trees = final_model.best_iteration + 1 if hasattr(final_model, 'best_iteration') else best_params["n_estimators"]
    print(f"\n  Final model: {actual_trees} trees (early stopped from {best_params['n_estimators']})")

    return final_model, study.best_params


def evaluate_classifier(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n  {model_name} Test Results:")
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


def simulate_ml_betting(model, X_test, y_test, threshold=0.55):
    y_prob = model.predict_proba(X_test)[:, 1]
    mask = (y_prob >= threshold) | (y_prob <= (1 - threshold))

    if not mask.any():
        return {"bets": 0, "wins": 0, "roi": 0}

    y_pred_conf = (y_prob[mask] >= 0.5).astype(int)
    y_true_conf = y_test.iloc[mask].values
    wins = (y_pred_conf == y_true_conf).sum()
    total = len(y_pred_conf)
    win_rate = wins / total if total > 0 else 0
    profit = wins * 100 - (total - wins) * 110
    roi = profit / (total * 110) * 100 if total > 0 else 0

    return {
        "bets": total, "wins": wins, "losses": total - wins,
        "win_rate": win_rate, "profit_units": profit / 110, "roi": roi,
    }


# ═══════════════════════════════════════════════════════════════════
# REGRESSOR (Spread / Totals)
# ═══════════════════════════════════════════════════════════════════

def make_regressor_objective(X_train, y_train, cv_folds, sample_weights):
    """Create objective function for regression with Huber loss + early stopping."""

    def objective(trial):
        # Choose loss function: Huber is robust to blowout outliers
        loss = trial.suggest_categorical("loss_fn", ["reg:squarederror", "reg:pseudohubererror"])

        params = {
            "objective": loss,
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 200, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 200, log=True),
            "max_delta_step": trial.suggest_float("max_delta_step", 0, 5),
        }

        # Huber delta (only used when loss is pseudohubererror)
        if loss == "reg:pseudohubererror":
            params["huber_slope"] = trial.suggest_float("huber_slope", 0.5, 5.0)

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            w_tr = sample_weights[train_idx] if sample_weights is not None else None

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            scores.append(-mean_absolute_error(y_val, y_pred))

        return np.mean(scores)

    return objective


def train_regressor(X_train, y_train, model_name):
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Regression)")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")
    print(f"  Target — mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    weights = compute_sample_weights(X_train)
    objective_fn = make_regressor_objective(X_train, y_train, CV_FOLDS, weights)

    study = run_multiphase_optuna(objective_fn, direction="maximize")

    # Build final params
    best_p = study.best_params.copy()
    loss = best_p.pop("loss_fn", "reg:squarederror")

    best_params = {
        "objective": loss,
        "eval_metric": "mae",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        **best_p,
    }
    best_params["n_estimators"] = max(best_params.get("n_estimators", 500), 1500)

    # Early stopping holdout
    split_idx = int(len(X_train) * 0.85)
    X_fit, X_hold = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_hold = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    w_fit = weights[:split_idx] if weights is not None else None

    final_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    final_model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_hold, y_hold)],
        verbose=False,
    )

    actual_trees = final_model.best_iteration + 1 if hasattr(final_model, 'best_iteration') else best_params["n_estimators"]
    print(f"\n  Final model: {actual_trees} trees, loss: {loss}")

    return final_model, study.best_params


def evaluate_regressor(model, X_test, y_test, model_name, target_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print(f"\n  {model_name} Test Results:")
    print(f"    MAE:   {mae:.2f} points")
    print(f"    RMSE:  {rmse:.2f} points")
    print(f"    R²:    {r2:.4f}")

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


def simulate_regression_betting(model, X_test, y_test, edge_threshold, target_name):
    y_pred = model.predict(X_test)

    if target_name == "home_margin":
        mask = np.abs(y_pred) >= edge_threshold
        if not mask.any():
            return {"bets": 0, "wins": 0, "roi": 0}
        correct = (y_pred[mask] > 0) == (y_test.values[mask] > 0)
    else:
        median = y_test.median()
        mask = np.abs(y_pred - median) >= edge_threshold
        if not mask.any():
            return {"bets": 0, "wins": 0, "roi": 0}
        correct = (y_pred[mask] > median) == (y_test.values[mask] > median)

    wins = correct.sum()
    total = len(correct)
    win_rate = wins / total if total > 0 else 0
    profit = wins * 100 - (total - wins) * 110
    roi = profit / (total * 110) * 100 if total > 0 else 0

    return {
        "bets": total, "wins": wins, "losses": total - wins,
        "win_rate": win_rate, "profit_units": profit / 110, "roi": roi,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("NBA ML Model Training Pipeline")
    print("  3-Phase Optuna: QMC → TPE → CMA-ES")
    print("  + Early stopping, sample weights, Huber loss")
    print("=" * 60)

    train_df, test_df = load_and_prepare_data()
    feature_cols = get_feature_columns(train_df)
    print(f"\nTotal available features: {len(feature_cols)}")

    results = {}

    # ────────────────────────────────────────────────────────────
    # 1. MONEYLINE MODEL (Classification: home_win)
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 1: MONEYLINE (Classification)")
    print("=" * 60)

    X_train, y_train = prepare_xy(train_df, "home_win", feature_cols)
    X_test, y_test = prepare_xy(test_df, "home_win", feature_cols)

    if USE_FEATURE_SELECTION:
        ml_cols = select_features(X_train, y_train, feature_cols, "classification")
        X_train_ml, X_test_ml = X_train[ml_cols], X_test[ml_cols]
    else:
        ml_cols = feature_cols
        X_train_ml, X_test_ml = X_train, X_test

    model, params = train_classifier(X_train_ml, y_train, "moneyline_model")
    metrics = evaluate_classifier(model, X_test_ml, y_test, "moneyline_model")

    print(f"\n  Top 10 Features:")
    print(get_feature_importance(model, ml_cols).head(10).to_string(index=False))

    sim = simulate_ml_betting(model, X_test_ml, y_test, 0.55)
    print(f"\n  Betting Sim (55%+ conf): {sim['bets']} bets, "
          f"{sim['win_rate']:.2%} win rate, {sim['roi']:+.1f}% ROI")

    save_model(model, "moneyline_model", ml_cols, metrics, params, "classifier")
    results["moneyline_model"] = metrics

    # ────────────────────────────────────────────────────────────
    # 2. SPREAD MODEL (Regression: home_margin)
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: SPREAD (Regression on home margin)")
    print("=" * 60)

    X_train_s, y_train_s = prepare_xy(train_df, "home_margin", feature_cols)
    X_test_s, y_test_s = prepare_xy(test_df, "home_margin", feature_cols)

    if USE_FEATURE_SELECTION:
        spread_cols = select_features(X_train_s, y_train_s, feature_cols, "regression")
        X_train_s, X_test_s = X_train_s[spread_cols], X_test_s[spread_cols]
    else:
        spread_cols = feature_cols

    model_s, params_s = train_regressor(X_train_s, y_train_s, "spread_model")
    metrics_s = evaluate_regressor(model_s, X_test_s, y_test_s, "spread_model", "home_margin")

    print(f"\n  Top 10 Features:")
    print(get_feature_importance(model_s, spread_cols).head(10).to_string(index=False))

    sim_s = simulate_regression_betting(model_s, X_test_s, y_test_s, 3.0, "home_margin")
    print(f"\n  Betting Sim (3+ pt edge): {sim_s['bets']} bets, "
          f"{sim_s['win_rate']:.2%} win rate, {sim_s['roi']:+.1f}% ROI")

    save_model(model_s, "spread_model", spread_cols, metrics_s, params_s, "regressor")
    results["spread_model"] = metrics_s

    # ────────────────────────────────────────────────────────────
    # 3. TOTALS MODEL (Regression: total_score)
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 3: TOTALS (Regression on total score)")
    print("=" * 60)

    X_train_t, y_train_t = prepare_xy(train_df, "total_score", feature_cols)
    X_test_t, y_test_t = prepare_xy(test_df, "total_score", feature_cols)

    if USE_FEATURE_SELECTION:
        totals_cols = select_features(X_train_t, y_train_t, feature_cols, "regression")
        X_train_t, X_test_t = X_train_t[totals_cols], X_test_t[totals_cols]
    else:
        totals_cols = feature_cols

    model_t, params_t = train_regressor(X_train_t, y_train_t, "totals_model")
    metrics_t = evaluate_regressor(model_t, X_test_t, y_test_t, "totals_model", "total_score")

    print(f"\n  Top 10 Features:")
    print(get_feature_importance(model_t, totals_cols).head(10).to_string(index=False))

    sim_t = simulate_regression_betting(model_t, X_test_t, y_test_t, 5.0, "total_score")
    print(f"\n  Betting Sim (5+ pt edge): {sim_t['bets']} bets, "
          f"{sim_t['win_rate']:.2%} win rate, {sim_t['roi']:+.1f}% ROI")

    save_model(model_t, "totals_model", totals_cols, metrics_t, params_t, "regressor")
    results["totals_model"] = metrics_t

    # ────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────
    total_trials = OPTUNA_PHASE1_TRIALS + OPTUNA_PHASE2_TRIALS + OPTUNA_PHASE3_TRIALS

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Optuna: {total_trials} trials/model (QMC {OPTUNA_PHASE1_TRIALS} → TPE {OPTUNA_PHASE2_TRIALS} → CMA-ES {OPTUNA_PHASE3_TRIALS})")
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
