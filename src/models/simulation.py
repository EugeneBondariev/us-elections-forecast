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
    _MAD_TO_STD = 0.6745  # MAD / sigma for a normal distribution
    values = np.asarray(values, dtype=float)
    mad = np.median(np.abs(values - float(np.median(values))))
    return float(mad / _MAD_TO_STD)


def simulate_election(
    proba: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
    sigma_national: float = 0.25,
    sigma_district: float = 0.15,
    hetero_k: float = 1.0,
) -> np.ndarray:
    """Monte Carlo simulation with correlated national + independent district shocks (logit space).

    hetero_k scales per-district noise by competitiveness: sigma_i = sigma_district * (1 + k * c)
    where c = 1 - |2p - 1| (1 at p=0.5, 0 at safe seats). k=0 is homoskedastic.
    """
    rng = np.random.default_rng(seed)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    logits = np.log(proba / (1 - proba))
    national_shocks = rng.normal(loc=0, scale=sigma_national, size=(n, 1))
    competitiveness = 1.0 - np.abs(2.0 * proba - 1.0)  # 1 at p=0.5, 0 at extremes
    sigma_d = sigma_district * (1.0 + hetero_k * competitiveness)  # shape (D,)
    district_shocks = rng.normal(loc=0, scale=sigma_d, size=(n, len(proba)))
    adjusted_proba = 1 / (1 + np.exp(-(logits + national_shocks + district_shocks)))
    return rng.binomial(n=1, p=adjusted_proba).sum(axis=1)


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
    beta = (1 - proba) * concentration
    sampled_p = rng.beta(a=alpha, b=beta, size=(n, len(proba)))
    return rng.binomial(n=1, p=sampled_p).sum(axis=1)


def simulate_blended(
    proba: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
    blend_weight: float = 0.5,
    sigma_national: float = 0.25,
    sigma_district: float = 0.15,
    concentration: float = 15.0,
    hetero_k: float = 1.0,
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
        n,
        len(proba),
        blend_weight * 100,
        (1 - blend_weight) * 100,
    )
    parts: list[np.ndarray] = []
    if n_corr > 0:
        parts.append(
            simulate_election(
                proba,
                n=n_corr,
                seed=seed,
                sigma_national=sigma_national,
                sigma_district=sigma_district,
                hetero_k=hetero_k,
            )
        )
    if n_bayes > 0:
        parts.append(
            bayesian_simulate(
                proba,
                n=n_bayes,
                seed=seed + 1,
                concentration=concentration,
            )
        )
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def calibrate_national_sigma(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    sigma_district: float = 0.15,
    hetero_k: float = 1.0,
    start_test_year: int = 2000,
    exclude_years: frozenset[int] = frozenset({2002}),
    recency_decay: float = 0.85,
    n: int = 2000,
    seed: int = 42,
    min_years: int = 3,
    compute_ci: bool = False,
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

    exclude_years removes structurally broken years from the error distribution —
    by default 2002, where the model was trained only on pre-redistricting data and
    missed by -51.7 seats, a model coverage failure rather than genuine electoral
    variance. recency_decay exponentially down-weights older walk-forward years so
    the calibration better reflects modern partisan conditions.
    """
    folds = walk_forward_probas(X, y, years, start_test_year)
    if exclude_years:
        dropped = sorted(exclude_years & set(folds))
        if dropped:
            logger.info("Excluding years from sigma calibration: %s", dropped)
        folds = {yr: v for yr, v in folds.items() if yr not in exclude_years}
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

    fold_years = list(folds.keys())
    max_year = max(fold_years)
    weights = np.array([recency_decay ** (max_year - yr) for yr in fold_years])
    weights /= weights.sum()

    realized_errors = np.array([actual.sum() - proba.sum() for proba, actual in folds.values()])
    w_mean = float(np.average(realized_errors, weights=weights))
    target_std = float(np.sqrt(np.average((realized_errors - w_mean) ** 2, weights=weights)))

    def _weighted_simulated_std(sigma_national: float) -> float:
        stds = np.array(
            [
                simulate_election(
                    proba,
                    n=n,
                    seed=seed,
                    sigma_national=sigma_national,
                    sigma_district=sigma_district,
                    hetero_k=hetero_k,
                ).std()
                for proba, _ in folds.values()
            ]
        )
        return float(np.average(stds, weights=weights))

    lo, hi = 0.01, 1.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if _weighted_simulated_std(mid) < target_std:
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

    result: CalibrationResult = {
        "sigma_national": round(sigma_national, 4),
        "sigma_district": sigma_district,
        "target_std": round(target_std, 4),
        "n_years": len(folds),
    }

    if compute_ci and len(folds) >= 4:
        # Leave-one-year-out jackknife: rerun binary search with each year dropped once.
        # Uses n=500 (lighter) since jackknife runs len(folds) full calibrations.
        jk_sigmas: list[float] = []
        fold_items = list(folds.items())
        for leave_out in range(len(fold_items)):
            jk_folds = {yr: v for i, (yr, v) in enumerate(fold_items) if i != leave_out}
            jk_years = list(jk_folds.keys())
            max_yr_jk = max(jk_years)
            jk_weights = np.array([recency_decay ** (max_yr_jk - yr) for yr in jk_years])
            jk_weights /= jk_weights.sum()
            jk_errors = np.array(
                [actual.sum() - proba.sum() for proba, actual in jk_folds.values()]
            )
            jk_mean = float(np.average(jk_errors, weights=jk_weights))
            jk_target = float(np.sqrt(np.average((jk_errors - jk_mean) ** 2, weights=jk_weights)))

            def _jk_std(s: float, _jf: dict = jk_folds, _jw: np.ndarray = jk_weights) -> float:
                stds = np.array(
                    [
                        simulate_election(
                            proba,
                            n=500,
                            seed=seed,
                            sigma_national=s,
                            sigma_district=sigma_district,
                            hetero_k=hetero_k,
                        ).std()
                        for proba, _ in _jf.values()
                    ]
                )
                return float(np.average(stds, weights=_jw))

            lo_jk, hi_jk = 0.01, 1.0
            for _ in range(15):
                mid_jk = (lo_jk + hi_jk) / 2
                if _jk_std(mid_jk) < jk_target:
                    lo_jk = mid_jk
                else:
                    hi_jk = mid_jk
            jk_sigmas.append((lo_jk + hi_jk) / 2)

        n_jk = len(jk_sigmas)
        jk_arr = np.array(jk_sigmas)
        jk_var = (n_jk - 1) / n_jk * float(np.sum((jk_arr - jk_arr.mean()) ** 2))
        jk_se = float(np.sqrt(jk_var))
        result["sigma_low"] = round(max(0.01, sigma_national - 1.96 * jk_se), 4)
        result["sigma_high"] = round(sigma_national + 1.96 * jk_se, 4)
        logger.info(
            "  Jackknife 95%% CI for sigma_national: [%.4f, %.4f]",
            result["sigma_low"],
            result["sigma_high"],
        )

    return result


def calibrate_simulation_params(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    sigma_district: float = 0.15,
    hetero_k: float = 1.0,
    start_test_year: int = 2000,
    exclude_years: frozenset[int] = frozenset({2002}),
    recency_decay: float = 0.85,
    n: int = 2000,
    seed: int = 42,
    compute_ci: bool = False,
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
        X,
        y,
        years,
        sigma_district=sigma_district,
        hetero_k=hetero_k,
        start_test_year=start_test_year,
        exclude_years=exclude_years,
        recency_decay=recency_decay,
        n=n,
        seed=seed,
        compute_ci=compute_ci,
    )
    sigma_national = base["sigma_national"]
    target_std = base["target_std"]

    folds = walk_forward_probas(X, y, years, start_test_year)
    if exclude_years:
        folds = {yr: v for yr, v in folds.items() if yr not in exclude_years}

    fold_years = list(folds.keys())
    max_year = max(fold_years)
    weights = np.array([recency_decay ** (max_year - yr) for yr in fold_years])
    weights /= weights.sum()

    # Step 2: concentration — higher → tighter Beta → lower std
    def _weighted_bayesian_std(conc: float) -> float:
        stds = np.array(
            [
                bayesian_simulate(proba, n=n, seed=seed, concentration=conc).std()
                for proba, _ in folds.values()
            ]
        )
        return float(np.average(stds, weights=weights))

    lo_c, hi_c = 2.0, 500.0
    for _ in range(25):
        mid = (lo_c + hi_c) / 2
        if _weighted_bayesian_std(mid) < target_std:
            hi_c = mid  # too tight, need lower concentration
        else:
            lo_c = mid  # too loose, need higher concentration
    concentration = (lo_c + hi_c) / 2
    logger.info(
        "Calibrated concentration=%.1f (target seat-count std=%.2f from %d walk-forward years)",
        concentration,
        target_std,
        len(folds),
    )

    if abs(concentration - 2.0) < 0.1:
        logger.warning(
            "Concentration hit search floor (%.1f) — Bayesian Beta draws cannot match "
            "walk-forward variance regardless of per-district uncertainty. "
            "Setting blend_weight=1.0 (pure correlated national-shock simulation).",
            concentration,
        )
        floor_result: CalibrationResult = {
            "sigma_national": sigma_national,
            "sigma_district": sigma_district,
            "target_std": round(target_std, 4),
            "n_years": len(folds),
            "blend_weight": 1.0,
            "concentration": round(concentration, 1),
            "hetero_k": hetero_k,
        }
        if "sigma_low" in base:
            floor_result["sigma_low"] = base.get("sigma_low", float("nan"))
            floor_result["sigma_high"] = base.get("sigma_high", float("nan"))
        return floor_result

    # Step 3: blend_weight — weighted 68% CI coverage criterion (recent years count more)
    realized_seats = {year: int(actual.sum()) for year, (_, actual) in folds.items()}
    best_weight, best_gap, best_coverage = 0.5, float("inf"), 0.0

    for w_int in range(0, 21):
        w = round(w_int / 20, 2)
        weighted_covered = 0.0
        for (year, (proba, _)), wt in zip(folds.items(), weights):
            sim = simulate_blended(
                proba,
                n=n,
                seed=seed,
                blend_weight=w,
                sigma_national=sigma_national,
                sigma_district=sigma_district,
                concentration=concentration,
                hetero_k=hetero_k,
            )
            if int(np.percentile(sim, 16)) <= realized_seats[year] <= int(np.percentile(sim, 84)):
                weighted_covered += float(wt)
        coverage = weighted_covered  # weights already sum to 1
        gap = abs(coverage - 0.68)
        if gap < best_gap:
            best_gap, best_weight, best_coverage = gap, w, coverage

    logger.info(
        "Calibrated blend_weight=%.2f (68%% CI coverage=%.1f%% across %d walk-forward years)",
        best_weight,
        best_coverage * 100,
        len(folds),
    )
    full_result: CalibrationResult = {
        "sigma_national": sigma_national,
        "sigma_district": sigma_district,
        "target_std": round(target_std, 4),
        "n_years": len(folds),
        "blend_weight": round(best_weight, 2),
        "concentration": round(concentration, 1),
        "hetero_k": hetero_k,
    }
    if "sigma_low" in base:
        full_result["sigma_low"] = base.get("sigma_low", float("nan"))
        full_result["sigma_high"] = base.get("sigma_high", float("nan"))
    return full_result


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
            hetero_k=float(params.get("hetero_k", 1.0)),
        )
    return SimParams(
        sigma_national=0.25,
        sigma_district=0.15,
        blend_weight=0.5,
        concentration=15.0,
        hetero_k=1.0,
    )
