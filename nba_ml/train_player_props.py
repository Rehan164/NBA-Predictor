"""
Player Prop Model Training

Trains XGBoost regressors for player stat predictions:
  - Points (PTS)
  - Rebounds (REB)
  - Assists (AST)

Usage:
    python -m nba_ml.train_player_props
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler

from .config import (
    PLAYER_FEATURES_CSV,
    MODELS_DIR,
    RANDOM_STATE,
)

# Player prop models saved here
PLAYER_PROPS_DIR = MODELS_DIR / "player_props"

# Training config
TRAIN_CUTOFF = "2022-07-01"
TEST_CUTOFF = "2022-10-01"
CV_FOLDS = 5
N_TRIALS = 60  # Fewer trials than team models since there's more data

EXCLUDE_COLS = [
    "player_id", "player_name", "team", "opponent",
    "game_id", "game_date",
    "target_pts", "target_reb", "target_ast", "actual_min",
]

TARGETS = {
    "pts": {"col": "target_pts", "name": "Points"},
    "reb": {"col": "target_reb", "name": "Rebounds"},
    "ast": {"col": "target_ast", "name": "Assists"},
}


def load_data():
    print(f"Loading player features from {PLAYER_FEATURES_CSV}...")

    if not PLAYER_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Player features not found: {PLAYER_FEATURES_CSV}\n"
            "Run 'python -m nba_ml.build_player_features' first."
        )

    df = pd.read_csv(PLAYER_FEATURES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])

    print(f"  {len(df):,} player-game rows")
    print(f"  {df['player_id'].nunique():,} unique players")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    train_df = df[df["game_date"] < TRAIN_CUTOFF].copy()
    test_df = df[df["game_date"] >= TEST_CUTOFF].copy()

    print(f"\n  Train: {len(train_df):,} rows (before {TRAIN_CUTOFF})")
    print(f"  Test:  {len(test_df):,} rows (after {TEST_CUTOFF})")

    return train_df, test_df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS and not df[c].isna().all()]


def prepare_xy(df: pd.DataFrame, target_col: str, feature_cols: list):
    valid = df.dropna(subset=[target_col])
    x = valid[feature_cols].fillna(0)
    y = valid[target_col]
    return x, y


def objective(trial, x_train, y_train, cv_folds):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = []
    for tr_idx, val_idx in tscv.split(x_train):
        x_tr, x_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(x_tr, y_tr, verbose=False)
        pred = model.predict(x_val)
        scores.append(-mean_absolute_error(y_val, pred))

    return np.mean(scores)


def train_prop_model(x_train, y_train, stat_name: str):
    print(f"\n  Training {stat_name} model...")
    print(f"    Samples: {len(x_train):,}, Features: {x_train.shape[1]}")
    print(f"    Target — mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, CV_FOLDS),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"    Best CV MAE: {-study.best_value:.2f}")

    best_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        **study.best_params,
    }

    model = xgb.XGBRegressor(**best_params)
    model.fit(x_train, y_train, verbose=False)

    return model, study.best_params


def evaluate(model, x_test, y_test, stat_name: str) -> dict:
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n    {stat_name} Test Results:")
    print(f"      MAE:  {mae:.2f}")
    print(f"      RMSE: {rmse:.2f}")
    print(f"      R²:   {r2:.4f}")

    # Within-X accuracy (how often is prediction within X points of actual)
    errors = np.abs(pred - y_test.values)
    for threshold in [2, 3, 5]:
        pct = (errors <= threshold).mean()
        print(f"      Within {threshold} pts: {pct:.1%}")

    # Top feature importances
    importance = pd.DataFrame({
        "feature": x_test.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n    Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"      {row['feature']:<30} {row['importance']:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    print("=" * 60)
    print("Player Prop Model Training")
    print("=" * 60)

    PLAYER_PROPS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()
    feature_cols = get_feature_cols(train_df)
    print(f"\n  Available features: {len(feature_cols)}")

    results = {}

    for stat_key, stat_info in TARGETS.items():
        target_col = stat_info["col"]
        stat_name = stat_info["name"]

        print(f"\n{'='*60}")
        print(f"  {stat_name.upper()} MODEL")
        print(f"{'='*60}")

        x_train, y_train = prepare_xy(train_df, target_col, feature_cols)
        x_test, y_test = prepare_xy(test_df, target_col, feature_cols)

        model, params = train_prop_model(x_train, y_train, stat_name)
        metrics = evaluate(model, x_test, y_test, stat_name)

        # Save model
        model_path = PLAYER_PROPS_DIR / f"player_{stat_key}_model.pkl"
        model_data = {
            "model": model,
            "feature_columns": feature_cols,
            "metrics": metrics,
            "params": params,
            "model_type": "regressor",
            "stat": stat_key,
            "trained_at": datetime.now().isoformat(),
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"\n    Saved to: {model_path}")

        results[stat_name] = metrics

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: {PLAYER_PROPS_DIR}\n")

    for stat_name, m in results.items():
        print(f"  {stat_name:<12}: MAE {m['mae']:.2f}, RMSE {m['rmse']:.2f}, R² {m['r2']:.4f}")

    print(f"\nNext: python -m nba_ml.predict")


if __name__ == "__main__":
    main()
