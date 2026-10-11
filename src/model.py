from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import DEM_MAJORITY_SEATS

logger = logging.getLogger(__name__)

__all__ = [
    "WeightedPipeline",
    "build_pipeline", "build_gbm_pipeline", "build_rf_pipeline",
    "build_ensemble", "build_stacking_ensemble", "build_calibrated_pipeline",
    "compute_sample_weights", "train", "cross_validate",
    "feature_weights", "walk_forward_cv", "walk_forward_feature_importance",
    "explain_districts", "simulate_election", "bayesian_simulate", "bootstrap_ci",
]

class WeightedPipeline(Pipeline):
    """Pipeline that accepts sample_weight and routes it to the model step."""
    def fit(self, X, y, sample_weight=None, **params):
        if sample_weight is not None:
            params["model__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


def build_pipeline() -> WeightedPipeline:
    return WeightedPipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def build_gbm_pipeline() -> WeightedPipeline:
    return WeightedPipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )),
    ])


def build_rf_pipeline() -> WeightedPipeline:
    return WeightedPipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
        )),
    ])


def build_ensemble() -> VotingClassifier:
    """Soft-voting ensemble: average probabilities from LR, GBM, and RF."""
    return VotingClassifier(
        estimators=[
            ("lr", build_pipeline()),
            ("gbm", build_gbm_pipeline()),
            ("rf", build_rf_pipeline()),
        ],
        voting="soft",
    )


def build_stacking_ensemble() -> StackingClassifier:
    """Stacking ensemble: LR meta-learner trained on OOF predictions from LR, GBM, RF."""
    return StackingClassifier(
        estimators=[
            ("lr", build_pipeline()),
            ("gbm", build_gbm_pipeline()),
            ("rf", build_rf_pipeline()),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=False,
    )


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(build_pipeline(), method="isotonic", cv=5)


def compute_sample_weights(years: pd.Series, half_life: int = 4) -> np.ndarray:
    """Exponential recency weighting; elections `half_life` cycles ago get half the weight."""
    elections_ago = (years.max() - years) / 2
    return np.exp(-np.log(2) / half_life * elections_ago).values


def train(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> StackingClassifier:
    logger.info("Training stacking ensemble on %d samples, %d features", len(X), X.shape[1])
    model = build_stacking_ensemble()
    model.fit(X, y, sample_weight=sample_weight)
    return model


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    logger.info("Running %d-fold cross-validation", cv)
    return cross_val_score(build_pipeline(), X, y, cv=cv).mean()


def feature_weights(model: object, feature_names: pd.Index | list[str]) -> dict[str, float]:
    """Extract normalised feature importance (%) from LR pipeline, GBM pipeline, or ensemble."""
    if isinstance(model, (VotingClassifier, StackingClassifier)):
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
        w = compute_sample_weights(years.loc[train_idx])
        pipeline.fit(X.loc[train_idx], y.loc[train_idx], sample_weight=w)
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


def explain_districts(model: StackingClassifier, X_pred: pd.DataFrame) -> pd.DataFrame:
    """Return SHAP values per feature for the prediction set.
    Uses the LightGBM sub-model (faster and more meaningful than LR for SHAP).
    Returns DataFrame: rows=districts (same index as X_pred), columns=features.
    """
    import shap
    lgbm_pipe = model.named_estimators_["gbm"]
    lgbm_model = lgbm_pipe.named_steps["model"]
    scaler = lgbm_pipe.named_steps["scaler"]
    X_scaled = scaler.transform(X_pred)
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_scaled)
    # For binary classification, shap_values may be a list [class0, class1] — take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return pd.DataFrame(shap_values, index=X_pred.index, columns=X_pred.columns)


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


def bayesian_simulate(
    proba: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
    concentration: float = 15.0,
) -> np.ndarray:
    """
    Bayesian simulation: treat each district probability as the mean of a
    Beta distribution. concentration controls uncertainty (higher = tighter).

    For each district, sample a 'true' probability from Beta(alpha, beta)
    where alpha = p * concentration, beta = (1-p) * concentration.
    Then draw a win/loss for each district from those sampled probabilities.
    """
    logger.info("Bayesian simulation: %d districts, %d draws", len(proba), n)
    rng = np.random.default_rng(seed)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    alpha = proba * concentration
    beta_param = (1 - proba) * concentration
    sampled_p = rng.beta(alpha, beta_param, size=(n, len(proba)))
    wins = rng.binomial(1, sampled_p)
    result = wins.sum(axis=1)
    median_seats = int(np.median(result))
    p_majority = float((result >= DEM_MAJORITY_SEATS).mean())
    logger.info("Bayesian simulation results: median=%d seats, P(majority>=%d)=%.1f%%",
                median_seats, DEM_MAJORITY_SEATS, p_majority * 100)
    return result


def bootstrap_ci(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap CI for walk-forward district accuracy and mean seat error.
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
        pipe = build_pipeline()
        w = compute_sample_weights(years.loc[train_idx])
        pipe.fit(X.loc[train_idx], y.loc[train_idx], sample_weight=w)
        pipelines[test_year] = (pipe, years[years == test_year].index)

    boot_accuracies = []
    boot_seat_errors = []

    for _ in range(n_bootstrap):
        accs, errors = [], []
        for test_year, (pipe, test_idx) in pipelines.items():
            # bootstrap resample test districts
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
