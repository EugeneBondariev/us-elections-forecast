import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    """Isotonic calibration on top of the base pipeline — better probability estimates."""
    return CalibratedClassifierCV(build_pipeline(), method="isotonic", cv=5)


def train(X: pd.DataFrame, y: pd.Series, calibrated: bool = False) -> Pipeline:
    logger.info("Training on %d samples", len(X))
    pipeline = build_calibrated_pipeline() if calibrated else build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    logger.info("Running %d-fold cross-validation", cv)
    return cross_val_score(build_pipeline(), X, y, cv=cv).mean()


def feature_weights(pipeline: Pipeline, feature_names) -> dict:
    model = pipeline.named_steps["model"]
    coef_abs = np.abs(model.coef_[0])
    weights = coef_abs / coef_abs.sum() * 100
    return dict(sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True))


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    start_test_year: int = 2000,
) -> pd.DataFrame:
    """
    For each election year >= start_test_year: train on all prior years, predict that year.
    Returns per-year seat count accuracy — more honest than random CV for time-series data.
    """
    test_years = sorted(yr for yr in years.unique() if yr >= start_test_year)
    logger.info("Walk-forward CV: %d years to evaluate", len(test_years))
    results = []
    for test_year in test_years:
        logger.debug("  → %d", test_year)
        train_idx = years[years < test_year].index
        test_idx = years[years == test_year].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        pipeline = build_pipeline()
        pipeline.fit(X.loc[train_idx], y.loc[train_idx])
        preds = pipeline.predict(X.loc[test_idx])
        y_test = y.loc[test_idx]
        results.append({
            "year": test_year,
            "actual_dem": int(y_test.sum()),
            "predicted_dem": int(preds.sum()),
            "seat_error": int(preds.sum()) - int(y_test.sum()),
            "district_accuracy": round(float((preds == y_test.values).mean()), 4),
        })
    return pd.DataFrame(results)


def simulate_election(
    proba: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo simulation of total Dem seats.
    Each district wins independently with its predicted probability.
    Returns array of length n with total Dem seat counts per simulation.
    """
    logger.info("Simulating %d elections (%d districts)", n, len(proba))
    rng = np.random.default_rng(seed)
    return rng.binomial(1, proba, (n, len(proba))).sum(axis=1)
