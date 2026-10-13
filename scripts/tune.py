"""Optuna hyperparameter tuning for LightGBM walk-forward CV.

Run from project root:
    python scripts/tune.py
"""

import json

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from src.features import build_dataset
from src.model import WeightedPipeline, compute_sample_weights
from src.paths import DataFile

OUTPUT_PATH = DataFile.BEST_PARAMS

VALIDATION_YEARS = list(range(2010, 2020, 2))  # 2010, 2012, 2014, 2016, 2018


def load_data():
    ds = build_dataset()
    return ds.X, ds.y, ds.years


def objective(trial, X, y, years):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 8, 31),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "verbose": -1,
    }

    accuracies = []
    for test_year in VALIDATION_YEARS:
        train_idx = years[years < test_year].index
        test_idx = years[years == test_year].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        pipeline = WeightedPipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(**params)),
        ])
        w = compute_sample_weights(years.loc[train_idx])
        pipeline.fit(X.loc[train_idx], y.loc[train_idx], sample_weight=w)
        preds = pipeline.predict(X.loc[test_idx])
        acc = float((preds == y.loc[test_idx].values).mean())
        accuracies.append(acc)

    return float(np.mean(accuracies)) if accuracies else 0.0


def main() -> None:
    print("Loading data...")
    X, y, years = load_data()
    print(f"  {len(X)} rows, {X.shape[1]} features, years {years.min()}-{years.max()}")
    print(f"  Validation window: {VALIDATION_YEARS}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, years), n_trials=50)

    best_params = study.best_params
    best_score = study.best_value

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest score (mean district accuracy): {best_score:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
