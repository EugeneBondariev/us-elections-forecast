"""Pipeline definitions, training, and feature utilities for the House forecast model.

Defines the sklearn pipelines (LR, GBM, RF) and the stacking ensemble used throughout
the project. Evaluation lives in evaluation.py; simulation in simulation.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = [
    "WeightedPipeline",
    "VoteShareRegressor",
    "build_lr_pipeline",
    "build_gbm_pipeline",
    "build_rf_pipeline",
    "build_ensemble",
    "build_stacking_ensemble",
    "build_calibrated_pipeline",
    "compute_sample_weights",
    "train",
    "feature_weights",
    "explain_districts",
]


class WeightedPipeline(Pipeline):
    """Pipeline that accepts sample_weight and routes it to the model step."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None, **params
    ) -> WeightedPipeline:
        if sample_weight is not None:
            params["model__sample_weight"] = sample_weight
        return super().fit(X, y, **params)

    def predict(self, X: object, **params: object) -> np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().predict(X, **params)  # pyright: ignore[reportArgumentType, reportReturnType]

    def predict_proba(self, X: object, **params: object) -> np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().predict_proba(X, **params)  # pyright: ignore[reportArgumentType]


def build_lr_pipeline() -> WeightedPipeline:
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
            ("lr", build_lr_pipeline()),
            ("gbm", build_gbm_pipeline()),
            ("rf", build_rf_pipeline()),
        ],
        voting="soft",
    )


def build_stacking_ensemble() -> StackingClassifier:
    """Stacking ensemble: LR meta-learner trained on OOF predictions from LR, GBM, RF."""
    return StackingClassifier(
        estimators=[
            ("lr", build_lr_pipeline()),
            ("gbm", build_gbm_pipeline()),
            ("rf", build_rf_pipeline()),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=False,
    )


def build_calibrated_pipeline() -> CalibratedClassifierCV:
    """Isotonic-calibrated LR pipeline (5-fold)."""
    return CalibratedClassifierCV(build_lr_pipeline(), method="isotonic", cv=5)


class VoteShareRegressor:
    """LightGBM regressor trained on continuous Dem vote share (0–1), exposing a
    predict_proba interface so it's drop-in compatible with the rest of the pipeline.

    Training on vote share preserves the magnitude of wins and losses — a 51% Dem
    result and a 70% result both look like "1" to a classifier, but carry very
    different signal for how safe or competitive a district is. The predicted vote
    share is clipped to [0, 1] and returned directly as P(Dem win).
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._lgbm: LGBMRegressor | None = None
        self.feature_importances_: np.ndarray | None = None
        self._bootstrap: list[tuple[StandardScaler, LGBMRegressor]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y_vote_share: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> VoteShareRegressor:
        params = {k: v for k, v in _gbm_params().items()}
        params["verbose"] = -1
        y_arr = np.asarray(y_vote_share, dtype=float)
        mask = ~np.isnan(y_arr)
        X_fit = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
        y_fit = y_arr[mask]
        sw_fit = sample_weight[mask] if sample_weight is not None else None
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_fit)
        self._lgbm = LGBMRegressor(**params)
        self._lgbm.fit(X_scaled, y_fit, sample_weight=sw_fit)
        self.feature_importances_ = self._lgbm.feature_importances_
        return self

    def fit_bootstrap(
        self,
        X: pd.DataFrame,
        y_vote_share: pd.Series,
        years: pd.Series,
        n_bootstrap: int = 50,
        sample_weight: np.ndarray | None = None,
    ) -> VoteShareRegressor:
        """Fit year-block bootstrap models for predict_ci(). Resamples election years
        (not rows) with replacement so within-year correlation is preserved."""
        unique_years = np.array(sorted(years.unique()))
        rng = np.random.default_rng(seed=42)
        params = {k: v for k, v in _gbm_params().items()}
        params["verbose"] = -1
        self._bootstrap = []
        for _ in range(n_bootstrap):
            boot_years = rng.choice(unique_years, size=len(unique_years), replace=True)
            row_idx = np.where(years.isin(boot_years))[0]
            X_b = X.iloc[row_idx]
            y_b = np.asarray(y_vote_share.iloc[row_idx]).astype(float)
            sw_b = sample_weight[row_idx] if sample_weight is not None else None
            nan_idx = np.where(~np.isnan(y_b))[0]
            X_fit = X_b.iloc[nan_idx]
            y_fit = y_b[nan_idx]
            sw_fit = sw_b[nan_idx] if sw_b is not None else None
            if len(y_fit) == 0:
                continue
            scaler = StandardScaler()
            lgbm = LGBMRegressor(**params)
            lgbm.fit(scaler.fit_transform(X_fit), y_fit, sample_weight=sw_fit)
            self._bootstrap.append((scaler, lgbm))
        return self

    def predict_ci(self, X: pd.DataFrame, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) vote-share CI at (1-alpha) coverage using bootstrap models."""
        if not self._bootstrap:
            raise RuntimeError("Call fit_bootstrap() before predict_ci()")
        preds = np.column_stack(
            [
                np.clip(np.asarray(lgbm.predict(scaler.transform(X)), dtype=float), 0.0, 1.0)
                for scaler, lgbm in self._bootstrap
            ]
        )
        lo = np.percentile(preds, 100 * alpha / 2, axis=1)
        hi = np.percentile(preds, 100 * (1 - alpha / 2), axis=1)
        return lo, hi

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._scaler is not None and self._lgbm is not None
        p = np.clip(
            np.asarray(self._lgbm.predict(self._scaler.transform(X)), dtype=float), 0.0, 1.0
        )
        return np.column_stack([1.0 - p, p])


def compute_sample_weights(years: pd.Series, half_life: int = 4) -> np.ndarray:
    """Exponential recency weighting; elections `half_life` cycles ago get half the weight."""
    elections_ago = (years.max() - years) / 2
    return np.exp(-np.log(2) / half_life * elections_ago).values


def train(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
    y_vote_share: pd.Series | None = None,
) -> StackingClassifier | VoteShareRegressor:
    """Fit and return the best available model.

    If y_vote_share is provided, trains a VoteShareRegressor (LightGBM regression on
    continuous Dem vote share). Otherwise falls back to the stacking classifier on
    binary labels. Both expose predict_proba so downstream code is unchanged.
    """
    if y_vote_share is not None:
        logger.info("Training VoteShareRegressor on %d samples, %d features", len(X), X.shape[1])
        return VoteShareRegressor().fit(X, y_vote_share, sample_weight=sample_weight)
    logger.info("Training stacking ensemble on %d samples, %d features", len(X), X.shape[1])
    model = build_stacking_ensemble()
    model.fit(X, y, sample_weight=sample_weight)
    return model


def feature_weights(model: object, feature_names: pd.Index | list[str]) -> dict[str, float]:
    """Extract normalised feature importance (%) from LR pipeline, GBM pipeline, or ensemble."""
    if isinstance(model, VoteShareRegressor):
        importances = model.feature_importances_
        if importances is None:
            return {}
    elif isinstance(model, (VotingClassifier, StackingClassifier)):
        importances = model.named_estimators_["gbm"].named_steps["model"].feature_importances_
    elif hasattr(model, "named_steps"):
        m = cast(Pipeline, model).named_steps["model"]
        importances = np.abs(m.coef_[0]) if hasattr(m, "coef_") else m.feature_importances_
    else:
        return {}
    total = importances.sum()
    weights = importances / total * 100 if total > 0 else np.zeros_like(importances, dtype=float)
    return dict(sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True))


def explain_districts(
    model: StackingClassifier | VoteShareRegressor, X_pred: pd.DataFrame
) -> pd.DataFrame:
    """Return SHAP values per feature for the prediction set.
    Uses the LightGBM sub-model (faster and more meaningful than LR for SHAP).
    Returns DataFrame: rows=districts (same index as X_pred), columns=features.
    """
    if isinstance(model, VoteShareRegressor):
        assert model._lgbm is not None and model._scaler is not None
        lgbm_model = model._lgbm
        scaler = model._scaler
    else:
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
