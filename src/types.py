from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd

__all__ = ["ForecastResult", "SimParams", "CalibrationResult", "BootstrapResult", "FECCandidateRecord"]


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
    hetero_k: NotRequired[float]  # competitive-district noise scaling; defaults to 1.0


class CalibrationResult(TypedDict):
    sigma_national: float
    sigma_district: float
    target_std: float
    n_years: int
    blend_weight: NotRequired[float]
    concentration: NotRequired[float]
    hetero_k: NotRequired[float]
    sigma_low: NotRequired[float]
    sigma_high: NotRequired[float]


class BootstrapResult(TypedDict):
    accuracy_mean: float
    accuracy_low: float
    accuracy_high: float
    seat_error_mean: float
    seat_error_low: float
    seat_error_high: float


class FECCandidateRecord(TypedDict):
    year: int
    state_po: str | None
    district: int
    party: str | None
    candidate_id: str | None
    candidate_name: str | None
    receipts: float
    individual_receipts: float
    individual_itemized_contributions: float
    individual_unitemized_contributions: float
    other_political_committee_contributions: float
    candidate_contribution: float
    disbursements: float
