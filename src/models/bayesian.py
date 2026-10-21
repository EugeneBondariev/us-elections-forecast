"""Bayesian hierarchical election model with district-level partial pooling.

Teaches: PyMC model specification, NUTS MCMC, InferenceData/trace objects,
ArviZ convergence diagnostics (R-hat, ESS), posterior predictive distributions,
credible intervals, and the bias-variance tradeoff expressed as a prior.

Model
-----
    sigma_alpha   ~ HalfNormal(0.5)              # between-district heterogeneity
    alpha_national ~ Normal(0, 1)                # national logit intercept
    alpha_district[d] ~ Normal(0, sigma_alpha)   # district random effects
    beta[k]        ~ Normal(0, 0.5)              # feature coefficients

    logit(p) = alpha_national + alpha_district[d] + X @ beta
    y ~ Bernoulli(p)

Partial pooling shrinks data-sparse districts toward the national mean, trading a
small amount of bias for a large reduction in variance.

Usage
-----
    from src.features.features import build_dataset
    from src.models.bayesian import BayesianElectionModel, make_district_labels

    ds = build_dataset()
    district_ids = make_district_labels(ds.winners, ds.X.index)
    model = BayesianElectionModel(n_draws=1000, n_tune=1000)
    model.fit(ds.X, ds.y, district_ids)
    proba = model.predict_proba(ds.X, district_ids)
    lo, hi = model.credible_interval(ds.X, district_ids)
    print(model.diagnostics())
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.constants import Col

logger = logging.getLogger(__name__)

__all__ = ["BayesianElectionModel", "make_district_labels"]


def make_district_labels(winners: pd.DataFrame, X_index: pd.Index) -> pd.DataFrame:
    """Select rows from winners by X_index and return state/district columns.

    winners may carry StrEnum column keys, so we coerce via str() before indexing.
    """
    state_col = str(Col.STATE)
    district_col = str(Col.DISTRICT)
    subset = winners.loc[X_index, [state_col, district_col]].reset_index(drop=True)
    return subset.astype({state_col: str, district_col: int})


class BayesianElectionModel:
    """Hierarchical Bernoulli model with district random effects via NUTS MCMC."""

    def __init__(
        self,
        n_draws: int = 1000,
        n_tune: int = 1000,
        n_chains: int = 2,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> None:
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed

        self._trace: object | None = None  # az.InferenceData
        self._scaler: StandardScaler | None = None
        self._district_map: dict[tuple[str, int], int] = {}
        self._n_districts: int = 0
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        district_ids: pd.DataFrame,
    ) -> BayesianElectionModel:
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError as exc:
            raise ImportError(
                "PyMC is required for BayesianElectionModel. Install it with: pip install pymc"
            ) from exc

        n_obs, n_features = X.shape
        self._feature_names = list(X.columns)

        # Encode (state, district) pairs as integer indices
        state_col = str(Col.STATE)
        district_col = str(Col.DISTRICT)
        pairs = list(
            zip(district_ids[state_col].astype(str), district_ids[district_col].astype(int))
        )
        unique_pairs = sorted(set(pairs))
        self._district_map = {pair: idx for idx, pair in enumerate(unique_pairs)}
        self._n_districts = len(unique_pairs)
        district_idx = np.array([self._district_map[p] for p in pairs], dtype=int)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X).astype(float)

        logger.info(
            "Fitting Bayesian model: %d obs, %d districts, %d features",
            n_obs,
            self._n_districts,
            n_features,
        )

        y_vals = np.asarray(y, dtype=int)
        X_const = pt.as_tensor_variable(X_scaled)

        with pm.Model():
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
            alpha_national = pm.Normal("alpha_national", mu=0, sigma=1)
            alpha_district = pm.Normal(
                "alpha_district", mu=0, sigma=sigma_alpha, shape=self._n_districts
            )
            beta = pm.Normal("beta", mu=0, sigma=0.5, shape=n_features)

            logit_p = alpha_national + alpha_district[district_idx] + pt.dot(X_const, beta)
            p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
            pm.Bernoulli("y", p=p, observed=y_vals)

            trace = pm.sample(
                draws=self.n_draws,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=False,
                return_inferencedata=True,
            )

        self._trace = trace
        logger.info("Sampling complete")
        return self

    def predict_proba(self, X: pd.DataFrame, district_ids: pd.DataFrame) -> np.ndarray:
        """Mean posterior win probability, shape (n_obs,)."""
        X_scaled, district_idx = self._prepare_inputs(X=X, district_ids=district_ids)
        samples = self._posterior_samples(X_scaled=X_scaled, district_idx=district_idx)
        return samples.mean(axis=1)

    def credible_interval(
        self,
        X: pd.DataFrame,
        district_ids: pd.DataFrame,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) credible interval at the given alpha level, shapes (n_obs,)."""
        X_scaled, district_idx = self._prepare_inputs(X=X, district_ids=district_ids)
        samples = self._posterior_samples(X_scaled=X_scaled, district_idx=district_idx)
        lo = np.quantile(samples, q=alpha / 2, axis=1)
        hi = np.quantile(samples, q=1 - alpha / 2, axis=1)
        return lo, hi

    def diagnostics(self) -> dict[str, float]:
        """ArviZ convergence diagnostics: R-hat, ESS, and divergence count."""
        import arviz as az

        trace = self._trace
        summary = az.summary(
            trace,
            var_names=["alpha_national", "sigma_alpha", "alpha_district", "beta"],
        )
        n_divergences = int(trace.sample_stats["diverging"].values.sum())  # type: ignore[union-attr]
        return {
            "r_hat_max": float(summary["r_hat"].max()),
            "r_hat_mean": float(summary["r_hat"].mean()),
            "ess_bulk_min": float(summary["ess_bulk"].min()),
            "ess_tail_min": float(summary["ess_tail"].min()),
            "n_divergences": n_divergences,
        }

    def summary(self) -> pd.DataFrame:
        """ArviZ parameter summary for national-level parameters and feature coefficients."""
        import arviz as az

        return cast(
            pd.DataFrame,
            az.summary(
                self._trace,
                var_names=["alpha_national", "sigma_alpha", "beta"],
                round_to=4,
            ),
        )

    def save(self, path: Path) -> None:
        """Persist trace (.nc) and metadata (.json) alongside it."""
        import arviz as az

        az.to_netcdf(self._trace, str(path.with_suffix(".nc")))

        scaler: StandardScaler = self._scaler
        meta = {
            "district_map": {
                f"{state}-{district}": idx for (state, district), idx in self._district_map.items()
            },
            "n_districts": self._n_districts,
            "feature_names": self._feature_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "n_draws": self.n_draws,
            "n_chains": self.n_chains,
        }
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: Path) -> BayesianElectionModel:
        """Reconstruct model from saved .nc trace and .json metadata."""
        import arviz as az

        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)

        obj = cls(n_draws=meta["n_draws"], n_chains=meta["n_chains"])
        obj._trace = az.from_netcdf(str(path.with_suffix(".nc")))
        obj._feature_names = meta["feature_names"]
        obj._n_districts = meta["n_districts"]

        obj._district_map = {}
        for key, idx in meta["district_map"].items():
            state, district_str = key.rsplit("-", maxsplit=1)
            obj._district_map[(state, int(district_str))] = idx

        scaler = StandardScaler()
        mean_ = np.array(meta["scaler_mean"], dtype=float)
        scale_ = np.array(meta["scaler_scale"], dtype=float)
        scaler.mean_ = mean_
        scaler.scale_ = scale_
        scaler.var_ = scale_**2
        scaler.n_features_in_ = len(mean_)
        obj._scaler = scaler

        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _posterior_samples(
        self,
        X_scaled: np.ndarray,
        district_idx: np.ndarray,
    ) -> np.ndarray:
        """Return posterior probability matrix of shape (n_obs, n_samples).

        district_idx=-1 means an unseen district: its random effect is set to zero
        so inference falls back to the national intercept only.
        """
        posterior = self._trace.posterior  # type: ignore[union-attr]

        # Flatten chain × draw into one samples dimension
        alpha_national = posterior["alpha_national"].values.reshape(-1)  # (S,)
        alpha_district = posterior["alpha_district"].values.reshape(-1, self._n_districts)  # (S, D)
        beta = posterior["beta"].values.reshape(-1, len(self._feature_names))  # (S, F)

        n_samples = alpha_national.shape[0]

        # District effect: zero for unseen (-1) districts
        district_effect = np.where(
            district_idx[:, None] == -1,
            0.0,
            alpha_district[np.newaxis, :, :]
            .reshape(n_samples, self._n_districts)[:, np.maximum(district_idx, 0)]
            .T,
        )  # shape (n_obs, n_samples)

        feature_effect = X_scaled @ beta.T  # (n_obs, n_samples)
        logit_p = alpha_national[np.newaxis, :] + district_effect + feature_effect
        return 1.0 / (1.0 + np.exp(-logit_p))

    def _prepare_inputs(
        self,
        X: pd.DataFrame,
        district_ids: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scale X and map district pairs to integer indices (-1 for unseen)."""
        X_scaled = self._scaler.transform(X).astype(float)  # type: ignore[union-attr]
        state_col = str(Col.STATE)
        district_col = str(Col.DISTRICT)
        pairs = list(
            zip(district_ids[state_col].astype(str), district_ids[district_col].astype(int))
        )
        district_idx = np.array([self._district_map.get(p, -1) for p in pairs], dtype=int)
        return X_scaled, district_idx
