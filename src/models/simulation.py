"""Monte Carlo simulation and parameter calibration for the House forecast model.

simulate_election: correlated national shock in logit space (primary component).
bayesian_simulate: independent Beta distribution per district (secondary component).
simulate_blended: weighted combination of both, with calibrated parameters.
calibrate_simulation_params: data-driven calibration of all four parameters
    from walk-forward residuals.

Run calibration: python -m src.models.calibrate_sigma
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.models.evaluation import walk_forward_probas
from src.types import CalibrationResult, SimParams

logger = logging.getLogger(__name__)

__all__ = [
    "simulate_election",
    "bayesian_simulate",
    "simulate_blended",
    "calibrate_national_sigma",
    "calibrate_simulation_params",
    "load_simulation_params",
]


def _robust_std(values: np.ndarray) -> float:
    """Median absolute deviation, scaled by 1.4826 to be std-equivalent under
    normality - unlike plain std, a single extreme value can't dominate it, since
    it's built from the median rather than squared deviations from the mean."""
    values = np.asarray(values, dtype=float)
    mad = np.median(np.abs(values - np.median(values)))
    return float(1.4826 * mad)


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
    """Bayesian simulation: treat each district probability as the mean of a
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

    lo_c, hi_c = 2.0, 500.0
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

    if abs(concentration - 2.0) < 0.1:
        logger.warning(
            "Concentration hit search floor (%.1f) — Bayesian Beta draws cannot match "
            "walk-forward variance regardless of per-district uncertainty. "
            "Setting blend_weight=1.0 (pure correlated national-shock simulation).",
            concentration,
        )
        return {
            "sigma_national": sigma_national,
            "sigma_district": sigma_district,
            "target_std": round(target_std, 4),
            "n_years": len(folds),
            "blend_weight": 1.0,
            "concentration": round(concentration, 1),
        }

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
    """Load calibrated Monte Carlo params from disk if available, else fall back
    to the uncalibrated defaults.

    Run `python -m src.models.calibrate_sigma` to (re)generate models/calibrated_sigma.json
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
