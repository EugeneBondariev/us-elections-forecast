from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd

__all__ = ["ForecastResult", "SimParams", "CalibrationResult", "BootstrapResult"]


class ForecastResult(TypedDict):
    seats_sim: np.ndarray
    n_districts: int
    districts: pd.DataFrame
    shap_values: pd.DataFrame


class SimParams(TypedDict):
    sigma_national: float
    sigma_district: float
    blend_weight: float
    concentration: float


class CalibrationResult(TypedDict):
    sigma_national: float
    sigma_district: float
    target_std: float
    n_years: int
    blend_weight: NotRequired[float]
    concentration: NotRequired[float]


class BootstrapResult(TypedDict):
    accuracy_mean: float
    accuracy_low: float
    accuracy_high: float
    seat_error_mean: float
    seat_error_low: float
    seat_error_high: float
