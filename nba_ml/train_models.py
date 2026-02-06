"""
Model Training Pipeline for NBA Betting ML

Trains XGBoost classifiers for:
- Spread (ATS): Predict if home team covers the spread
- Totals (O/U): Predict if game goes over the total
- Moneyline: Predict if home team wins outright

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
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


# Features to use for training (exclude identifiers and targets)
EXCLUDE_COLS = [
    "game_id", "date", "season", "home_team", "away_team",
    "home_score", "away_score", "home_win", "total_score", "home_margin",
    "spread_line", "total_line", "home_cover", "total_over",
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
    # Drop rows where target is missing
    valid_df = df.dropna(subset=[target])

    X = valid_df[feature_cols].copy()
    y = valid_df[target].copy()

    # Fill remaining NaN features with median
    X = X.fillna(X.median())

    return X, y


def select_features(X_train, y_train, feature_cols: list) -> list:
    """
    Two-phase feature selection:
    1. Train a quick XGBoost model
    2. Remove features with near-zero importance
    """
    print(f"\nRunning feature selection on {len(feature_cols)} features...")

    quick_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        tree_method="hist",
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


def objective_xgb(trial, X_train, y_train, cv_folds=5):
    """Optuna objective function for XGBoost hyperparameter tuning."""
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

    # Time series cross-validation
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


def train_model_with_optuna(X_train, y_train, model_name: str, n_trials: int = 50):
    """Train XGBoost model with Optuna hyperparameter tuning."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target distribution: {y_train.value_counts(normalize=True).to_dict()}")

    # Optuna study
    print(f"\nRunning Optuna optimization ({n_trials} trials)...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, CV_FOLDS),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest CV accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best params
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


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{model_name} Test Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    # Profitability check
    if metrics["accuracy"] > BREAK_EVEN_PCT:
        roi = (metrics["accuracy"] - BREAK_EVEN_PCT) / BREAK_EVEN_PCT * 100
        print(f"  Estimated ROI: +{roi:.1f}% (above break-even at -110)")
    else:
        print(f"  Below break-even ({BREAK_EVEN_PCT:.2%} needed at -110)")

    return metrics


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Get feature importance from trained model."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    importance = importance.sort_values("importance", ascending=False)
    return importance


def simulate_betting(model, X_test, y_test, confidence_threshold: float = 0.54) -> dict:
    """
    Simulate flat betting strategy with confidence threshold.

    Returns betting simulation results.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    # Only bet when confidence exceeds threshold (either direction)
    high_conf_mask = (y_prob >= confidence_threshold) | (y_prob <= (1 - confidence_threshold))

    if not high_conf_mask.any():
        return {"bets": 0, "wins": 0, "roi": 0}

    # Predictions for high confidence bets
    y_pred_conf = (y_prob[high_conf_mask] >= 0.5).astype(int)
    y_true_conf = y_test.iloc[high_conf_mask].values

    wins = (y_pred_conf == y_true_conf).sum()
    total = len(y_pred_conf)
    win_rate = wins / total if total > 0 else 0

    # ROI calculation at -110 odds
    # Win: +100, Loss: -110
    profit = wins * 100 - (total - wins) * 110
    roi = profit / (total * 110) * 100 if total > 0 else 0

    return {
        "bets": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": win_rate,
        "profit_units": profit / 110,
        "roi": roi,
    }


def save_model(model, model_name: str, feature_cols: list, metrics: dict, params: dict):
    """Save trained model and metadata."""
    model_path = MODELS_DIR / f"{model_name}.pkl"

    model_data = {
        "model": model,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "params": params,
        "trained_at": datetime.now().isoformat(),
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Saved model to: {model_path}")


def train_single_model(train_df, test_df, target: str, model_name: str,
                        feature_cols: list, results: dict):
    """Train, evaluate, and save a single model with feature selection."""
    X_train, y_train = prepare_xy(train_df, target, feature_cols)
    X_test, y_test = prepare_xy(test_df, target, feature_cols)

    # Per-model feature selection
    if USE_FEATURE_SELECTION:
        selected_cols = select_features(X_train, y_train, feature_cols)
        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]
    else:
        selected_cols = feature_cols

    model, params = train_model_with_optuna(
        X_train, y_train, model_name, OPTUNA_TRIALS
    )
    metrics = evaluate_model(model, X_test, y_test, model_name)

    print(f"\nTop 10 Features ({model_name}):")
    importance = get_feature_importance(model, selected_cols)
    print(importance.head(10).to_string(index=False))

    sim = simulate_betting(model, X_test, y_test, 0.55)
    print("\nBetting Simulation (55%+ confidence):")
    print(f"  Bets: {sim['bets']}, Wins: {sim['wins']}, Win Rate: {sim['win_rate']:.2%}")
    print(f"  ROI: {sim['roi']:.1f}%, Units: {sim['profit_units']:.1f}")

    save_model(model, model_name, selected_cols, metrics, params)
    results[model_name] = metrics


def main():
    """Main training routine."""
    print("=" * 60)
    print("NBA ML Model Training Pipeline")
    print("=" * 60)

    # Load data
    train_df, test_df = load_and_prepare_data()
    feature_cols = get_feature_columns(train_df)
    print(f"\nTotal available features: {len(feature_cols)}")

    results = {}

    # ────────────────────────────────────────────────────────────
    # 1. MONEYLINE MODEL (home_win)
    # ────────────────────────────────────────────────────────────
    train_single_model(
        train_df, test_df, "home_win", "moneyline_model",
        feature_cols, results
    )

    # ────────────────────────────────────────────────────────────
    # 2. SPREAD MODEL (home covers positive margin)
    # ────────────────────────────────────────────────────────────
    train_df["home_cover_proxy"] = (train_df["home_margin"] > 0).astype(int)
    test_df["home_cover_proxy"] = (test_df["home_margin"] > 0).astype(int)

    train_single_model(
        train_df, test_df, "home_cover_proxy", "spread_model",
        feature_cols, results
    )

    # ────────────────────────────────────────────────────────────
    # 3. TOTALS MODEL (over median total)
    # ────────────────────────────────────────────────────────────
    median_total = train_df["total_score"].median()
    print(f"\nUsing median total ({median_total:.1f}) as proxy O/U line")

    train_df["total_over_proxy"] = (train_df["total_score"] > median_total).astype(int)
    test_df["total_over_proxy"] = (test_df["total_score"] > median_total).astype(int)

    train_single_model(
        train_df, test_df, "total_over_proxy", "totals_model",
        feature_cols, results
    )

    # ────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print("\nTest Set Accuracy:")
    for name, metrics in results.items():
        status = "PROFITABLE" if metrics["accuracy"] > BREAK_EVEN_PCT else "needs work"
        print(f"  {name:20}: {metrics['accuracy']:.2%} ({status})")

    print("\nNote: Spread and Totals models use proxy targets.")
    print("For actual ATS/O/U predictions, add historical betting lines to the dataset.")


if __name__ == "__main__":
    main()
