import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])


def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def cross_validate(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    return cross_val_score(build_pipeline(), X, y, cv=cv).mean()


def feature_weights(pipeline: Pipeline, feature_names) -> dict:
    coef_abs = np.abs(pipeline.named_steps["model"].coef_[0])
    weights = coef_abs / coef_abs.sum() * 100
    return dict(sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True))
