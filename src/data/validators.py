"""Pandera schemas for pipeline DataFrame boundaries.

All validators here RAISE on failure (pandera.errors.SchemaErrors, or ValueError for
the manual plausibility checks below). This is a deliberate choice for data that can
be hand-assembled from external sources (e.g. presidential approval ratings, often
copied cell-by-cell from a spreadsheet) — a bad cell copy should break the pipeline
loudly, not slip through as a logged warning nobody reads until the forecast looks
wrong for reasons nobody can trace back.
"""

import logging

import pandera.pandas as pa
from pandera.typing import Series

from src.constants import Col

__all__ = [
    "HouseDataSchema",
    "FeatureMatrixSchema",
    "ApprovalSchema",
    "validate_house_data",
    "validate_feature_matrix",
    "validate_approval_data",
    "validate_approval_coverage",
]

logger = logging.getLogger(__name__)


class HouseDataSchema(pa.DataFrameModel):
    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    party: Series[str] = pa.Field(alias=Col.PARTY)
    candidatevotes: Series[float] = pa.Field(ge=0, alias=Col.CANDIDATE_VOTES)
    totalvotes: Series[float] = pa.Field(gt=0, alias=Col.TOTAL_VOTES)

    class Config:
        coerce = True
        strict = False


class FeatureMatrixSchema(pa.DataFrameModel):
    """Schema for the feature matrix X before model training."""

    dem_lean: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN)
    prev_dem_vote_share: Series[float] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PREV_DEM_SHARE
    )
    pres_approval: Series[float] = pa.Field(alias=Col.PRES_APPROVAL)

    class Config:
        coerce = True
        strict = False


class ApprovalSchema(pa.DataFrameModel):
    """Schema for raw approval ratings — 0-100 scale, validated BEFORE the
    Dem/Rep sign-flip in loaders.load_approval, while the numbers still mean
    exactly what they say on the source spreadsheet."""

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    pres_approval: Series[float] = pa.Field(ge=0, le=100, alias=Col.PRES_APPROVAL)
    pres_disapproval: Series[float] = pa.Field(ge=0, le=100, alias=Col.PRES_DISAPPROVAL)
    unsure: Series[float] = pa.Field(ge=0, le=100, alias=Col.UNSURE)

    class Config:
        coerce = True
        strict = False


def validate_house_data(df):
    """Validate raw house data. Raises pandera.errors.SchemaErrors on failure."""
    try:
        HouseDataSchema.validate(df, lazy=True)
        logger.info("House data validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "House data validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df


def validate_feature_matrix(X):
    """Validate feature matrix before training. Raises pandera.errors.SchemaErrors on failure."""
    try:
        FeatureMatrixSchema.validate(X, lazy=True)
        logger.info("Feature matrix validation passed (%d rows, %d cols)", *X.shape)
    except pa.errors.SchemaErrors as e:
        logger.error("Feature matrix validation FAILED: %d issue(s)", len(e.failure_cases))
        raise
    return X


def validate_approval_data(df, tolerance: float = 5.0):
    """Validate raw approval ratings before the sign-flip. Raises on failure.

    Catches the kind of mistake that's easy to make when hand-copying cells from a
    spreadsheet: out-of-range percentages (schema bounds), and rows where
    approve + disapprove + unsure is far from 100% — a strong signal that a cell
    was copied from the wrong row or column. `tolerance` allows for the couple of
    points of rounding slack real polling data normally has.
    """
    try:
        ApprovalSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Approval data validation FAILED: %d issue(s) out of range\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise

    totals = df[Col.PRES_APPROVAL] + df[Col.PRES_DISAPPROVAL] + df[Col.UNSURE]
    bad = df[(totals < 100 - tolerance) | (totals > 100 + tolerance)]
    if not bad.empty:
        raise ValueError(
            f"Approval data validation FAILED: approve+disapprove+unsure is not ~100% "
            f"(tolerance={tolerance}) for {len(bad)} row(s) — likely a cell copied from "
            f"the wrong row/column:\n{bad.to_string()}"
        )

    logger.info("Approval data validation passed (%d rows)", len(df))
    return df


def validate_approval_coverage(df, required_years: set[int]):
    """Validate that approval data has exactly one row per required year. Raises on failure.

    Separate from validate_approval_data because this needs to know which years the
    pipeline actually requires (build_dataset's call site knows; a unit test loading
    a 1-row fixture does not) — missing years silently become NaN after the merge in
    build_winners and get dropped by build_feature_matrix's dropna(), and duplicate
    years silently multiply rows in that same merge. Both are easy to miss without
    this check and easy to cause with a skipped or double-copied cell.
    """
    years = df[Col.YEAR]

    dupes = sorted(years[years.duplicated()].unique())
    if dupes:
        raise ValueError(
            f"Approval data validation FAILED: duplicate entries for year(s) {dupes} — "
            "expected exactly one reading per cycle."
        )

    missing = sorted(required_years - set(years))
    if missing:
        raise ValueError(
            f"Approval data validation FAILED: missing year(s) {missing} — check you "
            "copied a cell for every administration/cycle."
        )

    logger.info("Approval data coverage validated (%d required years present)", len(required_years))
    return df
