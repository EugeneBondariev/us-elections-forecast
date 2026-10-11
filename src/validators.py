"""Pandera schemas for pipeline DataFrame boundaries."""

import pandera.pandas as pa
from pandera.typing import Series

from src.constants import Col


class HouseRowSchema(pa.DataFrameModel):
    """Schema for a single row after load_clean_house_data."""
    class Config:
        coerce = True
        strict = False  # allow extra columns

# Raw house data
class HouseDataSchema(pa.DataFrameModel):
    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    party: Series[str] = pa.Field(alias=Col.PARTY)
    candidatevotes: Series[float] = pa.Field(ge=0, alias=Col.CANDIDATE_VOTES)
    totalvotes: Series[float] = pa.Field(ge=0, alias=Col.TOTAL_VOTES)

    class Config:
        coerce = True
        strict = False


class FeatureMatrixSchema(pa.DataFrameModel):
    """Schema for the feature matrix X before model training."""
    dem_lean: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN)
    prev_dem_vote_share: Series[float] = pa.Field(ge=0, le=1, nullable=True, alias=Col.PREV_DEM_SHARE)
    pres_approval: Series[float] = pa.Field(alias=Col.PRES_APPROVAL)

    class Config:
        coerce = True
        strict = False


def validate_house_data(df):
    """Validate raw house data. Logs warnings on failure, does not raise."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        HouseDataSchema.validate(df, lazy=True)
        logger.info("House data validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.warning("House data validation warnings: %d issues\n%s",
                       len(e.failure_cases), e.failure_cases.head(5).to_string())
    return df


def validate_feature_matrix(X):
    """Validate feature matrix before training. Logs warnings, does not raise."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        FeatureMatrixSchema.validate(X, lazy=True)
        logger.info("Feature matrix validation passed (%d rows, %d cols)", *X.shape)
    except pa.errors.SchemaErrors as e:
        logger.warning("Feature matrix validation warnings: %d issues", len(e.failure_cases))
    return X
