"""Walk-forward and bootstrap evaluation for the House forecast model.

All functions here build on walk_forward_probas as their shared foundation:
train the base LR pipeline on data up to year T, predict on T, repeat for each T.
The stacking ensemble is not used for walk-forward eval (too slow to refit 13×);
LR is fast, interpretable, and representative enough for these diagnostics.

Run via predict.py: python -m src.models.predict
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score

from src.models.model import build_lr_pipeline, compute_sample_weights, feature_weights
from src.types import BootstrapResult

logger = logging.getLogger(__name__)

__all__ = [
    "cross_validate",
    "walk_forward_cv",
    "walk_forward_feature_importance",
    "walk_forward_probas",
    "walk_forward_calibration",
    "walk_forward_tossup_skew",
    "bootstrap_ci",
]


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """Return mean stratified CV accuracy for the base LR pipeline."""
    logger.info("Running %d-fold cross-validation", cv)
    return cross_val_score(build_lr_pipeline(), X, y, cv=cv).mean()


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    start_test_year: int = 2000,
) -> pd.DataFrame:
    """Train on all years before test_year, evaluate on test_year. Repeat for each year >= start."""
    test_years = sorted(yr for yr in years.unique() if yr >= start_test_year)
    logger.info("Walk-forward CV: %d years to evaluate", len(test_years))
    results = []
    for test_year in test_years:
        logger.debug("  → %d", test_year)
        train_idx = years[years < test_year].index
        test_idx = years[years == test_year].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        pipeline = build_lr_pipeline()
        w = compute_sample_weights(years.loc[train_idx])
        pipeline.fit(X.loc[train_idx], y.loc[train_idx], sample_weight=w)
        preds = pipeline.predict(X.loc[test_idx])
        y_test = y.loc[test_idx]
        results.append(
            {
                "year": test_year,
                "actual_dem": int(y_test.sum()),
                "predicted_dem": int(preds.sum()),
                "seat_error": int(preds.sum()) - int(y_test.sum()),
                "district_accuracy": round(float((preds == y_test.values).mean()), 4),
            }
        )
    return pd.DataFrame(results)


def walk_forward_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    start_test_year: int = 2000,
) -> pd.DataFrame:
    """Feature weight (%) per walk-forward year using LR (fast, interpretable)."""
    test_years = sorted(yr for yr in years.unique() if yr >= start_test_year)
    rows = []
    for test_year in test_years:
        train_idx = years[years < test_year].index
        if len(train_idx) == 0:
            continue
        pipeline = build_lr_pipeline()
        pipeline.fit(X.loc[train_idx], y.loc[train_idx])
        rows.append({"year": test_year, **feature_weights(pipeline, X.columns)})
    return pd.DataFrame(rows).set_index("year")


def walk_forward_probas(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    start_test_year: int = 2000,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Train the base LR pipeline on all years before test_year; predict probabilities
    for test_year. Returns {year: (proba_dem, actual)} — shared building block for
    walk_forward_calibration (Brier/log-loss) and calibrate_national_sigma.
    """
    test_years = sorted(yr for yr in years.unique() if yr >= start_test_year)
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for test_year in test_years:
        train_idx = years[years < test_year].index
        test_idx = years[years == test_year].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        pipeline = build_lr_pipeline()
        w = compute_sample_weights(years.loc[train_idx])
        pipeline.fit(X.loc[train_idx], y.loc[train_idx], sample_weight=w)
        proba = pipeline.predict_proba(X.loc[test_idx])[:, 1]
        folds[test_year] = (proba, y.loc[test_idx].values)
    return folds


def walk_forward_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    start_test_year: int = 2000,
) -> pd.DataFrame:
    """Per-year Brier score and log loss from walk-forward probabilities.

    walk_forward_cv only checks the hard class prediction (district_accuracy); it says
    nothing about whether the predicted *probabilities* are well-calibrated. This does.
    """
    folds = walk_forward_probas(X, y, years, start_test_year)
    rows = []
    for year, (proba, actual) in folds.items():
        clipped = np.clip(proba, 1e-6, 1 - 1e-6)
        rows.append(
            {
                "year": year,
                "brier_score": float(brier_score_loss(actual, proba)),
                "log_loss": float(log_loss(actual, clipped, labels=[0, 1])),
                "n_districts": len(actual),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_tossup_skew(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    tossup_threshold: float = 0.1,
    start_test_year: int = 2000,
) -> pd.DataFrame:
    """Per-year toss-up district win rates from walk-forward predictions.

    A toss-up is any district where |proba_dem - 0.5| <= tossup_threshold (default ±10%).
    dem_pct near 50% = normal year; skewed 70-80% = national wave. Quantifies how
    correlated toss-up outcomes were — a direct read on national vs. district forces.
    """
    folds = walk_forward_probas(X, y, years, start_test_year)
    rows = []
    for year, (proba, actual) in folds.items():
        mask = np.abs(proba - 0.5) <= tossup_threshold
        n_tossup = int(mask.sum())
        if n_tossup == 0:
            continue
        dem_wins = int(actual[mask].sum())
        dem_pct = dem_wins / n_tossup * 100
        rows.append({
            "year": year,
            "n_tossup": n_tossup,
            "dem_wins": dem_wins,
            "dem_pct": round(dem_pct, 1),
            "skew": round(dem_pct - 50.0, 1),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for walk-forward district accuracy and mean seat error.

    Re-samples districts (with replacement) within each test year,
    re-evaluates predictions on the bootstrap sample.
    Returns dict with keys: accuracy_mean, accuracy_low, accuracy_high,
                            seat_error_mean, seat_error_low, seat_error_high
    """
    rng = np.random.default_rng(seed)
    test_years = sorted(yr for yr in years.unique() if yr >= 2000)

    # First fit one pipeline per year (reuse across bootstrap draws for speed)
    pipelines = {}
    for test_year in test_years:
        train_idx = years[years < test_year].index
        if len(train_idx) == 0:
            continue
        pipe = build_lr_pipeline()
        w = compute_sample_weights(years.loc[train_idx])
        pipe.fit(X.loc[train_idx], y.loc[train_idx], sample_weight=w)
        pipelines[test_year] = (pipe, years[years == test_year].index)

    boot_accuracies = []
    boot_seat_errors = []

    for _ in range(n_bootstrap):
        accs, errors = [], []
        for test_year, (pipe, test_idx) in pipelines.items():
            bs_idx = rng.choice(test_idx, size=len(test_idx), replace=True)
            X_bs = X.loc[bs_idx]
            y_bs = y.loc[bs_idx]
            preds = pipe.predict(X_bs)
            accs.append((preds == y_bs.values).mean())
            errors.append(abs(preds.sum() - y_bs.sum()))
        boot_accuracies.append(np.mean(accs))
        boot_seat_errors.append(np.mean(errors))

    alpha = (1 - ci) / 2
    return {
        "accuracy_mean": float(np.mean(boot_accuracies)),
        "accuracy_low": float(np.percentile(boot_accuracies, alpha * 100)),
        "accuracy_high": float(np.percentile(boot_accuracies, (1 - alpha) * 100)),
        "seat_error_mean": float(np.mean(boot_seat_errors)),
        "seat_error_low": float(np.percentile(boot_seat_errors, alpha * 100)),
        "seat_error_high": float(np.percentile(boot_seat_errors, (1 - alpha) * 100)),
    }
