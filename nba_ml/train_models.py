"""
Model Training Pipeline for NBA Betting ML

Trains models for:
- Moneyline: XGBoost Classifier (home win probability)
- Spread: XGBoost Regressor (predict home margin, compare to live spread)
- Totals: XGBoost Regressor (predict total score, compare to live O/U)

Usage:
    python -m nba_ml.train_models
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import optuna
from optuna.samplers import TPESampler

from .config import (
    TRAINING_FEATURES_CSV,
    MODELS_DIR,
    TRAIN_END_DATE,
    TEST_START_DATE,
    CV_FOLDS,
    RANDOM_STATE,
    OPTUNA_TRIALS,
    BREAK_EVEN_PCT,
    USE_FEATURE_SELECTION,
    FEATURE_IMPORTANCE_THRESHOLD,
)


# Features to exclude from training (identifiers + targets)
EXCLUDE_COLS = [
    "game_id", "date", "season", "home_team", "away_team",
    "home_score", "away_score", "home_win", "total_score", "home_margin",
    "spread_line", "total_line", "home_cover", "total_over",
    "home_ml", "away_ml",
    "home_cover_proxy", "total_over_proxy",
]


def load_and_prepare_data():
    """Load feature data and prepare train/test splits."""
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

    # Chronological split
    train_df = df[df["date"] < TRAIN_END_DATE].copy()
    test_df = df[df["date"] >= TEST_START_DATE].copy()

    print(f"\nTrain set: {len(train_df):,} games (before {TRAIN_END_DATE})")
    print(f"Test set: {len(test_df):,} games (after {TEST_START_DATE})")

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns for training."""
    return [c for c in df.columns if c not in EXCLUDE_COLS and not df[c].isna().all()]


def prepare_xy(df: pd.DataFrame, target: str, feature_cols: list):
    """Prepare X and y arrays for training."""
    valid_df = df.dropna(subset=[target])
    X = valid_df[feature_cols].copy()
    y = valid_df[target].copy()
    X = X.fillna(X.median())
    return X, y


def select_features(X_train, y_train, feature_cols: list, task="classification") -> list:
    """Feature selection using a quick model to drop near-zero importance features."""
    print(f"\nRunning feature selection on {len(feature_cols)} features...")

    if task == "classification":
        quick_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, tree_method="hist",
        )
    else:
        quick_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, tree_method="hist",
        )

    quick_model.fit(X_train, y_train, verbose=False)

    importances = pd.Series(quick_model.feature_importances_, index=feature_cols)
    selected = importances[importances >= FEATURE_IMPORTANCE_THRESHOLD].index.tolist()

    dropped = len(feature_cols) - len(selected)
    print(f"  Dropped {dropped} low-importance features")
    print(f"  Keeping {len(selected)} features")

    if len(selected) < 10:
        print("  Too few features after selection, keeping all")
        return feature_cols

    return selected


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    return importance.sort_values("importance", ascending=False)


def save_model(model, model_name: str, feature_cols: list, metrics: dict,
               params: dict, model_type: str = "classifier"):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    model_data = {
        "model": model,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "params": params,
        "model_type": model_type,
        "trained_at": datetime.now().isoformat(),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved to: {model_path}")


# ═══════════════════════════════════════════════════════════════════
# CLASSIFICATION (Moneyline)
# ═══════════════════════════════════════════════════════════════════

def objective_classifier(trial, X_train, y_train, cv_folds):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2),
    }

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))

    return np.mean(scores)


def train_classifier(X_train, y_train, model_name: str, n_trials: int):
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Classification)")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")
    print(f"  Target distribution: {y_train.value_counts(normalize=True).to_dict()}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective_classifier(trial, X_train, y_train, CV_FOLDS),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n  Best CV accuracy: {study.best_value:.4f}")

    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        **study.best_params,
    }

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, verbose=False)

    return final_model, study.best_params


def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
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
        print(f"    ROI estimate: +{roi:.1f}% (above {BREAK_EVEN_PCT:.2%} break-even)")
    else:
        print(f"    Below break-even ({BREAK_EVEN_PCT:.2%} needed at -110)")

    return metrics


def simulate_ml_betting(model, X_test, y_test, threshold=0.55):
    """Simulate moneyline flat betting with confidence threshold."""
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
# REGRESSION (Spread / Totals)
# ═══════════════════════════════════════════════════════════════════

def objective_regressor(trial, X_train, y_train, cv_folds):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
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
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_val)
        scores.append(-mean_absolute_error(y_val, y_pred))

    return np.mean(scores)


def train_regressor(X_train, y_train, model_name: str, n_trials: int):
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Regression)")
    print(f"{'='*60}")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")
    print(f"  Target — mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective_regressor(trial, X_train, y_train, CV_FOLDS),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n  Best CV MAE: {-study.best_value:.2f} points")

    best_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        **study.best_params,
    }

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train, verbose=False)

    return final_model, study.best_params


def evaluate_regressor(model, X_test, y_test, model_name: str, target_name: str) -> dict:
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
        # Direction accuracy: did we predict the right winner?
        dir_acc = ((y_pred > 0) == (y_test > 0)).mean()
        metrics["direction_accuracy"] = dir_acc
        print(f"    Winner prediction accuracy: {dir_acc:.2%}")

        for edge in [2, 3, 5]:
            mask = np.abs(y_pred) >= edge
            if mask.sum() > 10:
                acc = ((y_pred[mask] > 0) == (y_test.values[mask] > 0)).mean()
                print(f"    Accuracy when |predicted margin| >= {edge}: {acc:.2%} ({mask.sum()} games)")

    elif target_name == "total_score":
        median_total = y_test.median()
        dir_acc = ((y_pred > median_total) == (y_test > median_total)).mean()
        metrics["direction_accuracy"] = dir_acc
        print(f"    Over/Under median ({median_total:.0f}) accuracy: {dir_acc:.2%}")

        for edge in [3, 5, 8]:
            mask = np.abs(y_pred - median_total) >= edge
            if mask.sum() > 10:
                acc = ((y_pred[mask] > median_total) == (y_test.values[mask] > median_total)).mean()
                print(f"    Accuracy when |edge| >= {edge}: {acc:.2%} ({mask.sum()} games)")

    return metrics


def simulate_regression_betting(model, X_test, y_test, edge_threshold, target_name):
    """Simulate flat betting using regression edge threshold."""
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

    model, params = train_classifier(X_train_ml, y_train, "moneyline_model", OPTUNA_TRIALS)
    metrics = evaluate_classifier(model, X_test_ml, y_test, "moneyline_model")

    print(f"\n  Top 10 Features:")
    print(get_feature_importance(model, ml_cols).head(10).to_string(index=False))

    sim = simulate_ml_betting(model, X_test_ml, y_test, 0.55)
    print(f"\n  Betting Sim (55%+ confidence): {sim['bets']} bets, "
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

    model_s, params_s = train_regressor(X_train_s, y_train_s, "spread_model", OPTUNA_TRIALS)
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

    model_t, params_t = train_regressor(X_train_t, y_train_t, "totals_model", OPTUNA_TRIALS)
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
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: {MODELS_DIR}\n")

    for name, m in results.items():
        if "accuracy" in m:
            status = "PROFITABLE" if m["accuracy"] > BREAK_EVEN_PCT else "needs work"
            print(f"  {name:20}: Accuracy {m['accuracy']:.2%} ({status})")
        elif "mae" in m:
            dir_acc = m.get("direction_accuracy", 0)
            print(f"  {name:20}: MAE {m['mae']:.2f} pts, Direction accuracy {dir_acc:.2%}")

    print(f"\nNext: python -m nba_ml.predict")


if __name__ == "__main__":
    main()
