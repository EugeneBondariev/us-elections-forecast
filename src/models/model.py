from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.types import BootstrapResult, CalibrationResult, SimParams

logger = logging.getLogger(__name__)

__all__ = [
    "WeightedPipeline",
    "build_pipeline",
    "build_gbm_pipeline",
    "build_rf_pipeline",
    "build_ensemble",
    "build_stacking_ensemble",
    "build_calibrated_pipeline",
    "compute_sample_weights",
    "train",
    "cross_validate",
    "feature_weights",
    "walk_forward_cv",
    "walk_forward_feature_importance",
    "walk_forward_probas",
    "walk_forward_calibration",
    "calibrate_national_sigma",
    "load_simulation_params",
    "explain_districts",
    "simulate_election",
    "bayesian_simulate",
    "simulate_blended",
    "calibrate_simulation_params",
    "bootstrap_ci",
]


class WeightedPipeline(Pipeline):
    """Pipeline that accepts sample_weight and routes it to the model step."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None, **params
    ) -> WeightedPipeline:
        if sample_weight is not None:
            params["model__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


def build_pipeline() -> WeightedPipeline:
    """Logistic Regression pipeline with StandardScaler."""
    return WeightedPipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


_GBM_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


def _gbm_params() -> dict[str, Any]:
    """Return tuned params from disk if available, else hardcoded defaults."""
    from src.paths import DataFile
    if DataFile.BEST_PARAMS.exists():
        with open(DataFile.BEST_PARAMS) as f:
            params = json.load(f)
        params.setdefault("verbose", -1)
        logger.info("LightGBM: loaded tuned params from %s", DataFile.BEST_PARAMS)
        return params
    return _GBM_DEFAULTS


def build_gbm_pipeline() -> WeightedPipeline:
    """LightGBM pipeline with StandardScaler. Uses tuned params if available."""
    return WeightedPipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(**_gbm_params())),
        ]
    )


def build_rf_pipeline() -> WeightedPipeline:
    """Random Forest pipeline with StandardScaler."""
    return WeightedPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=15,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ]
    )


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
    """Isotonic-calibrated LR pipeline (5-fold)."""
    return CalibratedClassifierCV(build_pipeline(), method="isotonic", cv=5)


def compute_sample_weights(years: pd.Series, half_life: int = 4) -> np.ndarray:
    """Exponential recency weighting; elections `half_life` cycles ago get half the weight."""
    elections_ago = (years.max() - years) / 2
    return np.exp(-np.log(2) / half_life * elections_ago).values


def train(
    X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
) -> StackingClassifier:
    """Fit stacking ensemble on X, y and return the trained model."""
    logger.info("Training stacking ensemble on %d samples, %d features", len(X), X.shape[1])
    model = build_stacking_ensemble()
    model.fit(X, y, sample_weight=sample_weight)
    return model


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """Return mean stratified CV accuracy for the base LR pipeline."""
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
    total = importances.sum()
    weights = importances / total * 100 if total > 0 else np.zeros_like(importances, dtype=float)
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
        pipeline = build_pipeline()
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
        pipeline = build_pipeline()
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


def _robust_std(values: np.ndarray) -> float:
    """Median absolute deviation, scaled by 1.4826 to be std-equivalent under
    normality - unlike plain std, a single extreme value can't dominate it, since
    it's built from the median rather than squared deviations from the mean."""
    values = np.asarray(values, dtype=float)
    mad = np.median(np.abs(values - np.median(values)))
    return float(1.4826 * mad)


def calibrate_national_sigma(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    sigma_district: float = 0.15,
    start_test_year: int = 2000,
    n: int = 2000,
    seed: int = 42,
    min_years: int = 3,
) -> CalibrationResult:
    """Pick sigma_national so the Monte Carlo seat-count spread matches the
    *empirical* spread of realized walk-forward national seat errors, instead of
    using a hand-picked constant.

    For each walk-forward year, the realized error is actual_dem_seats minus the
    probability-weighted expected seats (sum of proba). Binary search finds the
    sigma_national whose simulated seat-count std (averaged across walk-forward
    years) matches that spread. sigma_district is held fixed: with only a dozen or
    so walk-forward years, there isn't enough held-out signal to identify both
    parameters jointly, so district-level noise stays at its prior default and only
    the aggregate/national component is data-driven.

    The spread itself is a robust measure (median absolute deviation, scaled by
    1.4826 to be std-equivalent under normality), not a plain standard deviation.
    With only ~13 walk-forward years, a single historically unusual miss can swing
    a plain std enormously - e.g. 2002's -51.7 seat error (the model, trained only
    on 1984-2000, badly over-predicted right after that cycle's redistricting) is
    roughly 1.5x any other year's miss and alone pushed a plain-std target high
    enough to pin sigma_national at its search ceiling (confirmed empirically: hit
    exactly 1.0 every time against real data). A robust spread still reflects that
    2002 was a bad year without letting one outlier set the uncertainty band every
    other, more typical year gets simulated with.
    """
    folds = walk_forward_probas(X, y, years, start_test_year)
    if len(folds) < min_years:
        logger.warning(
            "Only %d walk-forward years available (<%d) — keeping default sigma_national",
            len(folds),
            min_years,
        )
        return {
            "sigma_national": 0.25,
            "sigma_district": sigma_district,
            "target_std": float("nan"),
            "n_years": len(folds),
        }

    realized_errors = np.array([actual.sum() - proba.sum() for proba, actual in folds.values()])
    target_std = _robust_std(realized_errors)

    def _mean_simulated_std(sigma_national: float) -> float:
        stds = [
            simulate_election(
                proba, n=n, seed=seed, sigma_national=sigma_national, sigma_district=sigma_district
            ).std()
            for proba, _ in folds.values()
        ]
        return float(np.mean(stds))

    lo, hi = 0.01, 1.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if _mean_simulated_std(mid) < target_std:
            lo = mid
        else:
            hi = mid
    sigma_national = (lo + hi) / 2

    logger.info(
        "Calibrated sigma_national=%.4f (target seat-count std=%.2f from %d walk-forward years)",
        sigma_national,
        target_std,
        len(folds),
    )
    return {
        "sigma_national": round(sigma_national, 4),
        "sigma_district": sigma_district,
        "target_std": round(target_std, 4),
        "n_years": len(folds),
    }


def calibrate_simulation_params(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    sigma_district: float = 0.15,
    start_test_year: int = 2000,
    n: int = 2000,
    seed: int = 42,
) -> CalibrationResult:
    """Calibrate all four Monte Carlo parameters from walk-forward residuals.

    1. sigma_national: binary search so simulated seat-count spread matches
       the robust std of realized walk-forward errors (same as calibrate_national_sigma).
    2. concentration: binary search so bayesian_simulate spread matches the same
       target_std, making both components equally wide on their own.
    3. blend_weight: grid search [0, 0.05, ..., 1.0] — pick the weight whose
       simulated 68% CI covers actual seat counts in closest to 68% of walk-forward
       years (the correct frequentist interval calibration criterion).
    """
    base = calibrate_national_sigma(
        X, y, years,
        sigma_district=sigma_district, start_test_year=start_test_year, n=n, seed=seed,
    )
    sigma_national = base["sigma_national"]
    target_std = base["target_std"]

    folds = walk_forward_probas(X, y, years, start_test_year)

    # Step 2: concentration — higher → tighter Beta → lower std
    def _mean_bayesian_std(conc: float) -> float:
        return float(np.mean([
            bayesian_simulate(proba, n=n, seed=seed, concentration=conc).std()
            for proba, _ in folds.values()
        ]))

    lo_c, hi_c = 1.0, 500.0
    for _ in range(25):
        mid = (lo_c + hi_c) / 2
        if _mean_bayesian_std(mid) < target_std:
            hi_c = mid  # too tight, need lower concentration
        else:
            lo_c = mid  # too loose, need higher concentration
    concentration = (lo_c + hi_c) / 2
    logger.info(
        "Calibrated concentration=%.1f (target seat-count std=%.2f from %d walk-forward years)",
        concentration, target_std, len(folds),
    )

    # Step 3: blend_weight — 68% CI coverage criterion
    realized_seats = {year: int(actual.sum()) for year, (_, actual) in folds.items()}
    best_weight, best_gap, best_coverage = 0.5, float("inf"), 0.0

    for w_int in range(0, 21):
        w = round(w_int / 20, 2)
        covered = 0
        for year, (proba, _) in folds.items():
            sim = simulate_blended(
                proba, n=n, seed=seed, blend_weight=w,
                sigma_national=sigma_national, sigma_district=sigma_district,
                concentration=concentration,
            )
            if int(np.percentile(sim, 16)) <= realized_seats[year] <= int(np.percentile(sim, 84)):
                covered += 1
        coverage = covered / len(folds)
        gap = abs(coverage - 0.68)
        if gap < best_gap:
            best_gap, best_weight, best_coverage = gap, w, coverage

    logger.info(
        "Calibrated blend_weight=%.2f (68%% CI coverage=%.1f%% across %d walk-forward years)",
        best_weight, best_coverage * 100, len(folds),
    )
    return {
        "sigma_national": sigma_national,
        "sigma_district": sigma_district,
        "target_std": round(target_std, 4),
        "n_years": len(folds),
        "blend_weight": round(best_weight, 2),
        "concentration": round(concentration, 1),
    }


def load_simulation_params() -> SimParams:
    """Load calibrated Monte Carlo sigmas from disk if available, else fall back
    to the uncalibrated defaults.

    Run `python -m src.models.calibrate_sigma` to (re)generate data/calibrated_sigma.json
    from walk-forward residuals. Mirrors the _gbm_params() pattern used for LightGBM.
    """
    from src.paths import DataFile

    if DataFile.CALIBRATED_SIGMA.exists():
        with open(DataFile.CALIBRATED_SIGMA) as f:
            params = json.load(f)
        logger.info("Loaded calibrated simulation sigmas from %s", DataFile.CALIBRATED_SIGMA)
        return SimParams(
            sigma_national=float(params.get("sigma_national", 0.25)),
            sigma_district=float(params.get("sigma_district", 0.15)),
            blend_weight=float(params.get("blend_weight", 0.5)),
            concentration=float(params.get("concentration", 15.0)),
        )
    return SimParams(sigma_national=0.25, sigma_district=0.15, blend_weight=0.5, concentration=15.0)


def explain_districts(model: StackingClassifier, X_pred: pd.DataFrame) -> pd.DataFrame:
    """Return SHAP values per feature for the prediction set.
    Uses the LightGBM sub-model (faster and more meaningful than LR for SHAP).
    Returns DataFrame: rows=districts (same index as X_pred), columns=features.
    """
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
    rng = np.random.default_rng(seed)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    alpha = proba * concentration
    beta_param = (1 - proba) * concentration
    sampled_p = rng.beta(alpha, beta_param, size=(n, len(proba)))
    return rng.binomial(1, sampled_p).sum(axis=1)


def simulate_blended(
    proba: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
    blend_weight: float = 0.5,
    sigma_national: float = 0.25,
    sigma_district: float = 0.15,
    concentration: float = 15.0,
) -> np.ndarray:
    """Blend simulate_election and bayesian_simulate.

    blend_weight=1.0 is pure simulate_election (correlated national shock);
    blend_weight=0.0 is pure bayesian_simulate (independent Beta draws).
    The two halves sum to exactly n samples.
    """
    n_corr = round(blend_weight * n)
    n_bayes = n - n_corr
    logger.info(
        "Blended simulation: %d draws (%d districts, %.0f%% correlated / %.0f%% Bayesian)",
        n, len(proba), blend_weight * 100, (1 - blend_weight) * 100,
    )
    parts: list[np.ndarray] = []
    if n_corr > 0:
        parts.append(simulate_election(
            proba, n=n_corr, seed=seed,
            sigma_national=sigma_national, sigma_district=sigma_district,
        ))
    if n_bayes > 0:
        parts.append(bayesian_simulate(
            proba, n=n_bayes, seed=seed + 1, concentration=concentration,
        ))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def bootstrap_ci(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
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
