import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def build_gbm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=20,
        )),
    ])


def build_ensemble() -> VotingClassifier:
    """Soft-voting ensemble: average probabilities from LR and GBM."""
    return VotingClassifier(
        estimators=[("lr", build_pipeline()), ("gbm", build_gbm_pipeline())],
        voting="soft",
    )


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(build_pipeline(), method="isotonic", cv=5)


def train(X: pd.DataFrame, y: pd.Series) -> VotingClassifier:
    logger.info("Training ensemble on %d samples, %d features", len(X), X.shape[1])
    model = build_ensemble()
    model.fit(X, y)
    return model


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    logger.info("Running %d-fold cross-validation", cv)
    return cross_val_score(build_pipeline(), X, y, cv=cv).mean()


def feature_weights(model, feature_names) -> dict:
    """Extract normalised feature importance (%) from LR pipeline, GBM pipeline, or ensemble."""
    if isinstance(model, VotingClassifier):
        importances = model.named_estimators_["gbm"].named_steps["model"].feature_importances_
    elif hasattr(model, "named_steps"):
        m = model.named_steps["model"]
        importances = np.abs(m.coef_[0]) if hasattr(m, "coef_") else m.feature_importances_
    else:
        return {}
    weights = importances / importances.sum() * 100
    return dict(sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True))


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
        pipeline = build_pipeline()
        pipeline.fit(X.loc[train_idx], y.loc[train_idx])
        rows.append({"year": test_year, **feature_weights(pipeline, X.columns)})
    return pd.DataFrame(rows).set_index("year")


def simulate_election(
    proba: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
    sigma_national: float = 0.25,
    sigma_district: float = 0.15,
) -> np.ndarray:
    """Monte Carlo simulation with correlated national + independent district shocks (logit space)."""
    logger.info("Simulating %d elections (%d districts)", n, len(proba))
    rng = np.random.default_rng(seed)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    logits = np.log(proba / (1 - proba))
    national_shocks = rng.normal(0, sigma_national, (n, 1))
    district_shocks = rng.normal(0, sigma_district, (n, len(proba)))
    adjusted_proba = 1 / (1 + np.exp(-(logits + national_shocks + district_shocks)))
    return rng.binomial(1, adjusted_proba).sum(axis=1)
