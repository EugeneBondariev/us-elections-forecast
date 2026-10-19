"""Pipeline definitions, training, and feature utilities for the House forecast model.

Defines the sklearn pipelines (LR, GBM, RF) and the stacking ensemble used throughout
the project. Evaluation lives in evaluation.py; simulation in simulation.py.
"""
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = [
    "WeightedPipeline",
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
