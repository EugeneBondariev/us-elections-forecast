"""Pandera schemas for pipeline DataFrame boundaries.

All validators here RAISE on failure (pandera.errors.SchemaErrors, or ValueError for
the manual plausibility checks below). This is a deliberate choice for data that can
be hand-assembled from external sources (e.g. presidential approval ratings, often
copied cell-by-cell from a spreadsheet) — a bad cell copy should break the pipeline
loudly, not slip through as a logged warning nobody reads until the forecast looks
wrong for reasons nobody can trace back.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from src.constants import Col

__all__ = [
    "HouseDataSchema",
    "TopCandidatesSchema",
    "DemsRepsSchema",
    "ApprovalSchema",
    "ApprovalDataSchema",
    "WinnersSchema",
    "FeatureMatrixSchema",
    "PresLeanSchema",
    "CookCsvSchema",
    "FundraisingSchema",
    "CensusSchema",
    "CandidateQualitySchema",
    "PollAggregatesSchema",
    "validate_house_data",
    "validate_feature_matrix",
    "validate_approval_data",
    "validate_approval_coverage",
    "validate_pres_lean",
    "validate_cook_csv",
    "validate_fundraising",
    "validate_census",
    "validate_candidate_quality",
    "validate_poll_aggregates",
    "GerrymanderingSchema",
    "validate_gerrymandering",
    "SpeechSentimentSchema",
    "validate_speech_sentiment",
    "SpeechSentimentHistSchema",
    "validate_speech_sentiment_hist",
    "DistrictAnalyticsSchema",
    "validate_district_analytics",
    "FipsStatesSchema",
    "validate_fips_states",
    "BillEmbeddingsSchema",
    "validate_bill_embeddings",
    "PolicyTopicsSchema",
    "validate_policy_topics",
    "RatingsEnsembleSchema",
    "validate_ratings_ensemble",
    "CosponsorshipSchema",
    "validate_cosponsorship",
]

logger = logging.getLogger(__name__)


class HouseDataSchema(pa.DataFrameModel):
    """Raw MIT Election Lab House returns after filtering to general-election rows.

    `year` — election year [1976, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `party` — candidate party label (DEMOCRAT, REPUBLICAN, or other)
    `candidatevotes` — votes received by this candidate [0, ∞)
    `totalvotes` — total votes cast in the district (> 0)
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    party: Series[str] = pa.Field(str_length={"min_value": 1}, alias=Col.PARTY)
    candidatevotes: Series[float] = pa.Field(ge=0, alias=Col.CANDIDATE_VOTES)
    totalvotes: Series[float] = pa.Field(gt=0, alias=Col.TOTAL_VOTES)

    class Config:
        coerce = True
        strict = False

    @pa.dataframe_check
    @classmethod
    def candidate_votes_le_total(cls, df: pd.DataFrame) -> pd.Series:
        return df[Col.CANDIDATE_VOTES] <= df[Col.TOTAL_VOTES]


class TopCandidatesSchema(pa.DataFrameModel):
    """One row per (year, state_po, district, party) — highest-vote candidate only.

    `year` — election year [1976, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `party` — candidate party label
    `candidatevotes` — votes received by the top candidate [0, ∞)
    `totalvotes` — total votes cast in the district (> 0)
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    party: Series[str] = pa.Field(str_length={"min_value": 1}, alias=Col.PARTY)
    candidatevotes: Series[float] = pa.Field(ge=0, alias=Col.CANDIDATE_VOTES)
    totalvotes: Series[float] = pa.Field(gt=0, alias=Col.TOTAL_VOTES)

    class Config:
        coerce = True
        strict = False


class DemsRepsSchema(pa.DataFrameModel):
    """Dem/Rep top-candidate merge with rolling vote-share features.

    `year` — election year [1976, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `dem_votes` — Democratic candidate vote total [0, ∞)
    `rep_votes` — Republican candidate vote total [0, ∞) (optional)
    `dem_vote_share` — Dem votes / total votes [0, 1] (optional)
    `dem_lean` — district-level Democratic lean relative to national (optional)
    `dem_lean_trend` — cycle-over-cycle change in dem_lean (optional)
    `prev_dem_vote_share` — prior-cycle dem_vote_share [0, 1] (optional)
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    dem_votes: Series[float] = pa.Field(ge=0, alias=Col.DEM_VOTES)
    rep_votes: Series[float] = pa.Field(ge=0, nullable=True, alias=Col.REP_VOTES)
    dem_vote_share: Series[float] = pa.Field(ge=0, le=1, nullable=True, alias=Col.DEM_VOTE_SHARE)
    dem_lean: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN)
    dem_lean_trend: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN_TREND)
    prev_dem_vote_share: Series[float] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PREV_DEM_SHARE
    )

    class Config:
        coerce = True
        strict = False


class ApprovalDataSchema(pa.DataFrameModel):
    """Presidential approval ratings after sign-adjustment — positive values = Dem-favorable.

    `year` — election year [1976, 2030]
    `pres_approval` — sign-adjusted approval rating (positive when Dem president)
    `pres_disapproval` — sign-adjusted disapproval rating (negative when Dem president)
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    pres_approval: Series[float] = pa.Field(alias=Col.PRES_APPROVAL)
    pres_disapproval: Series[float] = pa.Field(alias=Col.PRES_DISAPPROVAL)

    class Config:
        coerce = True
        strict = False


class WinnersSchema(pa.DataFrameModel):
    """District-level election winners merged with structural, economic, and demographic features.

    `year` — election year [1976, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `party` — winning party label
    `candidatevotes` — winner's vote total [0, ∞)
    `totalvotes` — total votes cast in the district (> 0)
    `vote_share` — winner's share of total votes [0, 1]
    `dem_lean` — district-level Democratic lean relative to national (optional)
    `dem_lean_trend` — cycle-over-cycle change in dem_lean (optional)
    `dem_vote_share` — Dem share of total votes [0, 1] (optional)
    `prev_dem_vote_share` — prior-cycle dem_vote_share [0, 1] (optional)
    `pres_approval` — sign-adjusted presidential approval
    `pres_disapproval` — sign-adjusted presidential disapproval
    `generic_ballot` — Dem minus Rep generic ballot margin (pp)
    `econ_growth` — real GDP growth rate in election year
    `unemployment` — unemployment rate in election year
    `is_incumbent_on_ballot` — 1.0 if incumbent ran, 0.0 otherwise (optional)
    `redistricting` — 1 if district redrawn since prior cycle, else 0 (optional)
    `pres_lean` — district presidential lean from prior cycle [0, 1] (optional)
    `fundraising_dem_share` — Dem share of total campaign receipts [0, 1] (optional)
    `large_donor_dem_share` — Dem share of itemized contributions [0, 1] (optional)
    `pac_dem_share` — Dem share of PAC contributions [0, 1] (optional)
    `spending_dem_share` — Dem share of total disbursements [0, 1] (optional)
    `surplus_dem_share` — Dem share of receipts-minus-disbursements surplus (optional)
    `pct_bachelor_plus` — share of district adults with a bachelor's degree or higher [0, 1] (optional)
    `pct_non_white` — share of district population identifying as non-white [0, 1] (optional)
    `median_income` — median household income in dollars [0, ∞) (optional)
    `median_age` — median age of district residents [0, 120] (optional)
    `candidate_quality_dem_share` — Dem share of combined candidate quality score [0, 1] (optional)
    `experience_cycles_dem` — Democratic candidate's prior election cycles [0, ∞) (optional)
    `experience_cycles_rep` — Republican candidate's prior election cycles [0, ∞) (optional)
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    party: Series[str] = pa.Field(str_length={"min_value": 1}, alias=Col.PARTY)
    candidatevotes: Series[float] = pa.Field(ge=0, alias=Col.CANDIDATE_VOTES)
    totalvotes: Series[float] = pa.Field(gt=0, alias=Col.TOTAL_VOTES)
    vote_share: Series[float] = pa.Field(ge=0, le=1, alias=Col.VOTE_SHARE)
    dem_lean: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN)
    dem_lean_trend: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN_TREND)
    dem_vote_share: Optional[Series[float]] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.DEM_VOTE_SHARE
    )
    prev_dem_vote_share: Series[float] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PREV_DEM_SHARE
    )
    pres_approval: Series[float] = pa.Field(alias=Col.PRES_APPROVAL)
    pres_disapproval: Series[float] = pa.Field(alias=Col.PRES_DISAPPROVAL)
    generic_ballot: Series[float] = pa.Field(alias=Col.GENERIC_BALLOT)
    econ_growth: Series[float] = pa.Field(alias=Col.ECON_GROWTH)
    unemployment: Series[float] = pa.Field(alias=Col.UNEMPLOYMENT)
    # Added by add_incumbency_features — absent in build_winners output
    is_incumbent_on_ballot: Optional[Series[float]] = pa.Field(
        nullable=True, alias=Col.IS_INCUMBENT
    )
    redistricting: Optional[Series[int]] = pa.Field(alias=Col.REDISTRICTING)
    # Present only when source data available
    pres_lean: Optional[Series[float]] = pa.Field(nullable=True, alias=Col.PRES_LEAN)
    fundraising_dem_share: Optional[Series[float]] = pa.Field(
        nullable=True, alias=Col.FUNDRAISING_DEM_SHARE
    )
    large_donor_dem_share: Optional[Series[float]] = pa.Field(
        nullable=True, alias=Col.LARGE_DONOR_DEM_SHARE
    )
    pac_dem_share: Optional[Series[float]] = pa.Field(nullable=True, alias=Col.PAC_DEM_SHARE)
    spending_dem_share: Optional[Series[float]] = pa.Field(
        nullable=True, alias=Col.SPENDING_DEM_SHARE
    )
    surplus_dem_share: Optional[Series[float]] = pa.Field(alias=Col.SURPLUS_DEM_SHARE)
    # Census ACS demographics — present when census_acs.csv is available
    pct_bachelor_plus: Optional[Series[float]] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PCT_BACHELOR_PLUS
    )
    pct_non_white: Optional[Series[float]] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PCT_NON_WHITE
    )
    median_income: Optional[Series[float]] = pa.Field(ge=0, nullable=True, alias=Col.MEDIAN_INCOME)
    median_age: Optional[Series[float]] = pa.Field(
        ge=0, le=120, nullable=True, alias=Col.MEDIAN_AGE
    )
    # Candidate quality — present when candidate_quality.csv is available
    candidate_quality_dem_share: Optional[Series[float]] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.CANDIDATE_QUALITY
    )
    experience_cycles_dem: Optional[Series[int]] = pa.Field(
        ge=0, nullable=True, alias=Col.EXPERIENCE_CYCLES_DEM
    )
    experience_cycles_rep: Optional[Series[int]] = pa.Field(
        ge=0, nullable=True, alias=Col.EXPERIENCE_CYCLES_REP
    )

    class Config:
        coerce = True
        strict = False


class FeatureMatrixSchema(pa.DataFrameModel):
    """Feature matrix passed to the model at training and prediction time.

    `dem_lean` — district-level Democratic lean relative to national (optional)
    `prev_dem_vote_share` — prior-cycle Dem vote share [0, 1] (optional)
    `net_pres_approval` — sign-adjusted net presidential approval (approval minus disapproval)
    `state_partisan_lean` — state-level partisan lean derived from presidential results (optional)
    """

    dem_lean: Series[float] = pa.Field(nullable=True, alias=Col.DEM_LEAN)
    prev_dem_vote_share: Series[float] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PREV_DEM_SHARE
    )
    net_pres_approval: Series[float] = pa.Field(alias=Col.NET_PRES_APPROVAL)
    state_partisan_lean: Series[float] = pa.Field(nullable=True, alias=Col.STATE_PARTISAN_LEAN)

    class Config:
        coerce = True
        strict = False


class ApprovalSchema(pa.DataFrameModel):
    """Raw presidential approval ratings on the 0–100 scale, before the Dem/Rep sign-flip.

    Validated before `loaders.load_approval` applies the sign adjustment, while the
    numbers still mean exactly what they say on the source spreadsheet.

    `year` — election year [1976, 2030]
    `pres_approval` — raw approval percentage [0, 100]
    `pres_disapproval` — raw disapproval percentage [0, 100]
    `unsure` — raw unsure/no-opinion percentage [0, 100]
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    pres_approval: Series[float] = pa.Field(ge=0, le=100, alias=Col.PRES_APPROVAL)
    pres_disapproval: Series[float] = pa.Field(ge=0, le=100, alias=Col.PRES_DISAPPROVAL)
    unsure: Series[float] = pa.Field(ge=0, le=100, alias=Col.UNSURE)

    class Config:
        coerce = True
        strict = False


class PresLeanSchema(pa.DataFrameModel):
    """Presidential vote share by congressional district, one row per district-election-year.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `pres_year` — presidential election year [2008, 2030]
    `pres_lean` — Democratic presidential candidate's share of the two-party vote [0, 1]
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    pres_year: Series[int] = pa.Field(ge=2008, le=2030)
    pres_lean: Series[float] = pa.Field(ge=0, le=1, alias=Col.PRES_LEAN)

    class Config:
        coerce = True
        strict = False


class CookCsvSchema(pa.DataFrameModel):
    """Cook Political Report district ratings converted to a Democratic win probability.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `cook_prob` — implied Democratic win probability derived from Cook rating [0, 1]
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    cook_prob: Series[float] = pa.Field(ge=0, le=1, alias=Col.COOK_PROB)

    class Config:
        coerce = True
        strict = False


class FundraisingSchema(pa.DataFrameModel):
    """FEC campaign finance features expressed as Dem share of combined Dem+Rep totals.

    `year` — election cycle year [1980, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number [0, 99]; codes > 53 are non-voting delegate races
    `fundraising_dem_share` — Dem share of total receipts [0, 1]
    `large_donor_dem_share` — Dem share of itemized (>=200) individual contributions [0, 1]
    `pac_dem_share` — Dem share of PAC and party committee contributions [0, 1]
    `spending_dem_share` — Dem share of total disbursements [0, 1]
    `surplus_dem_share` — Dem share of receipts-minus-disbursements; unbounded (can be < 0 or > 1)
    """

    year: Series[int] = pa.Field(ge=1980, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    # le=99 not le=53: raw FEC data includes non-voting delegate/territory candidates
    # whose district codes go beyond 53; the pipeline drops them on merge.
    district: Series[int] = pa.Field(ge=0, le=99, alias=Col.DISTRICT)
    fundraising_dem_share: Series[float] = pa.Field(ge=0, le=1, alias=Col.FUNDRAISING_DEM_SHARE)
    large_donor_dem_share: Series[float] = pa.Field(ge=0, le=1, alias=Col.LARGE_DONOR_DEM_SHARE)
    pac_dem_share: Series[float] = pa.Field(ge=0, le=1, alias=Col.PAC_DEM_SHARE)
    spending_dem_share: Series[float] = pa.Field(ge=0, le=1, alias=Col.SPENDING_DEM_SHARE)
    # surplus_dem_share is intentionally unbounded: negative when Dem spent more than raised
    # relative to Rep, >1 when Rep ran a deficit. Unlike the other share columns it is not
    # clamped to [0,1] by design — see fundraising.py _aggregate() for the formula.
    surplus_dem_share: Series[float] = pa.Field(alias=Col.SURPLUS_DEM_SHARE)

    class Config:
        coerce = True
        strict = False


class CensusSchema(pa.DataFrameModel):
    """ACS 5-year demographic features per congressional district.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `pct_bachelor_plus` — share of adults (25+) with a bachelor's degree or higher [0, 1] (optional)
    `pct_non_white` — share of total population identifying as non-white [0, 1] (optional)
    `median_income` — median household income in dollars [0, ∞) (optional)
    `median_age` — median age of district residents [0, 120] (optional)
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    pct_bachelor_plus: Series[float] = pa.Field(
        ge=0, le=1, nullable=True, alias=Col.PCT_BACHELOR_PLUS
    )
    pct_non_white: Series[float] = pa.Field(ge=0, le=1, nullable=True, alias=Col.PCT_NON_WHITE)
    median_income: Series[float] = pa.Field(ge=0, nullable=True, alias=Col.MEDIAN_INCOME)
    median_age: Series[float] = pa.Field(ge=0, le=120, nullable=True, alias=Col.MEDIAN_AGE)

    class Config:
        coerce = True
        strict = False


class CandidateQualitySchema(pa.DataFrameModel):
    """Candidate quality features derived from FEC filing history and NLP analysis.

    `year` — election cycle year [1980, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number [0, 99]; codes > 53 are non-voting delegate races
    `candidate_quality_dem_share` — Dem share of combined candidate quality score [0, 1]
    `experience_cycles_dem` — number of prior election cycles the Democratic candidate has run [0, ∞)
    `experience_cycles_rep` — number of prior election cycles the Republican candidate has run [0, ∞)
    """

    year: Series[int] = pa.Field(ge=1980, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=99, alias=Col.DISTRICT)
    candidate_quality_dem_share: Series[float] = pa.Field(ge=0, le=1, alias=Col.CANDIDATE_QUALITY)
    experience_cycles_dem: Series[int] = pa.Field(ge=0, alias=Col.EXPERIENCE_CYCLES_DEM)
    experience_cycles_rep: Series[int] = pa.Field(ge=0, alias=Col.EXPERIENCE_CYCLES_REP)

    class Config:
        coerce = True
        strict = False


class PollAggregatesSchema(pa.DataFrameModel):
    """Weighted generic-ballot aggregate with uncertainty estimate, one row per election cycle.

    `year` — election cycle year [2006, 2030]
    `generic_ballot` — house-effects-corrected Dem minus Rep margin in percentage points [-30, 30]
    `generic_ballot_sigma` — weighted standard deviation of adjusted polls (uncertainty estimate) [0, ∞)
    """

    year: Series[int] = pa.Field(ge=2006, le=2030, alias=Col.YEAR)
    generic_ballot: Series[float] = pa.Field(ge=-30, le=30, alias=Col.GENERIC_BALLOT)
    generic_ballot_sigma: Series[float] = pa.Field(ge=0, alias=Col.GENERIC_BALLOT_SIGMA)

    class Config:
        coerce = True
        strict = False


class GerrymanderingSchema(pa.DataFrameModel):
    """Polsby-Popper compactness scores derived from Census TIGER shapefiles.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `polsby_popper` — Polsby-Popper compactness score; 1.0 = perfect circle, 0.0 = maximally irregular [0, 1]
    `district_area_km2` — district area in square kilometres [0, ∞)
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    polsby_popper: Series[float] = pa.Field(ge=0.0, le=1.0)
    district_area_km2: Series[float] = pa.Field(ge=0.0)

    class Config:
        coerce = True
        strict = False


def validate_gerrymandering(
    df: pd.DataFrame, min_rows: int = 400
) -> DataFrame[GerrymanderingSchema]:
    """Validate gerrymandering compactness features. Raises on failure."""
    if len(df) < min_rows:
        raise ValueError(
            f"Gerrymandering data too small: {len(df)} rows (expected >= {min_rows}; "
            "check the TIGER shapefile fetch captured all districts)"
        )
    try:
        GerrymanderingSchema.validate(df, lazy=True)
        logger.info("Gerrymandering validation passed (%d districts)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Gerrymandering validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


def validate_house_data(df: pd.DataFrame) -> DataFrame[HouseDataSchema]:
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
    return df  # type: ignore[return-value]


def validate_feature_matrix(X: pd.DataFrame) -> DataFrame[FeatureMatrixSchema]:
    """Validate feature matrix before training. Raises pandera.errors.SchemaErrors on failure."""
    try:
        FeatureMatrixSchema.validate(X, lazy=True)
        logger.info("Feature matrix validation passed (%d rows, %d cols)", *X.shape)
    except pa.errors.SchemaErrors as e:
        logger.error("Feature matrix validation FAILED: %d issue(s)", len(e.failure_cases))
        raise
    return X  # type: ignore[return-value]


def validate_approval_data(df: pd.DataFrame, tolerance: float = 5.0) -> DataFrame[ApprovalSchema]:
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
    return df  # type: ignore[return-value]


def validate_approval_coverage(
    df: pd.DataFrame, required_years: set[int]
) -> DataFrame[ApprovalSchema]:
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
    return df  # type: ignore[return-value]


def validate_pres_lean(df: pd.DataFrame, min_rows: int = 400) -> DataFrame[PresLeanSchema]:
    """Validate presidential lean data. Raises on failure."""
    if len(df) < min_rows:
        raise ValueError(
            f"Presidential lean data too small: {len(df)} rows (expected >= {min_rows})"
        )
    try:
        PresLeanSchema.validate(df, lazy=True)
        logger.info("Presidential lean validation passed (%d district-year rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Presidential lean validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


def validate_cook_csv(df: pd.DataFrame, min_districts: int = 100) -> DataFrame[CookCsvSchema]:
    """Validate parsed Cook CSV. Raises on failure."""
    if len(df) < min_districts:
        raise ValueError(
            f"Cook CSV too small: {len(df)} districts (expected >= {min_districts}; "
            "check the Playwright fetch captured the full ratings table)"
        )
    try:
        CookCsvSchema.validate(df, lazy=True)
        logger.info("Cook CSV validation passed (%d districts)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Cook CSV validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


def validate_fundraising(df: pd.DataFrame, min_rows: int = 500) -> DataFrame[FundraisingSchema]:
    """Validate fundraising CSV. Raises on failure."""
    if len(df) < min_rows:
        raise ValueError(f"Fundraising data too small: {len(df)} rows (expected >= {min_rows})")
    try:
        FundraisingSchema.validate(df, lazy=True)
        logger.info("Fundraising validation passed (%d district-year rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Fundraising validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


def validate_census(df: pd.DataFrame, min_rows: int = 400) -> DataFrame[CensusSchema]:
    """Validate ACS census features. Raises on failure."""
    if len(df) < min_rows:
        raise ValueError(
            f"Census data too small: {len(df)} rows (expected >= {min_rows}; "
            "check the Census API fetch captured all districts)"
        )
    try:
        CensusSchema.validate(df, lazy=True)
        logger.info("Census validation passed (%d districts)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Census validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


def validate_candidate_quality(df: pd.DataFrame) -> DataFrame[CandidateQualitySchema]:
    """Validate candidate quality features. Raises on failure."""
    try:
        CandidateQualitySchema.validate(df, lazy=True)
        logger.info("Candidate quality validation passed (%d district-year rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Candidate quality validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


def validate_poll_aggregates(df: pd.DataFrame) -> DataFrame[PollAggregatesSchema]:
    """Validate poll aggregate features. Raises on failure."""
    try:
        PollAggregatesSchema.validate(df, lazy=True)
        logger.info("Poll aggregates validation passed (%d cycles)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Poll aggregates validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class SpeechSentimentSchema(pa.DataFrameModel):
    """Per-district VADER sentiment and bill activity from Congress.gov for the 2026 cycle.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `speech_sentiment` — mean VADER compound score across sponsored bill titles [-1, 1]
    `speech_count` — number of sponsored bill titles scored [0, ∞)
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    speech_sentiment: Series[float] = pa.Field(ge=-1.0, le=1.0)
    speech_count: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


def validate_speech_sentiment(df: pd.DataFrame) -> DataFrame[SpeechSentimentSchema]:
    """Validate Congress.gov speech sentiment features. Raises on failure."""
    try:
        SpeechSentimentSchema.validate(df, lazy=True)
        logger.info("Speech sentiment validation passed (%d districts)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Speech sentiment validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class SpeechSentimentHistSchema(pa.DataFrameModel):
    """Historical per-district VADER sentiment from Congress.gov, congresses 113–118.

    `year` — election year corresponding to the congress [2014, 2024]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `speech_sentiment` — mean VADER compound score across sponsored bill titles [-1, 1]
    `speech_count` — number of sponsored bill titles scored [0, ∞)
    """

    year: Series[int] = pa.Field(ge=2014, le=2024)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    speech_sentiment: Series[float] = pa.Field(ge=-1.0, le=1.0)
    speech_count: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


def validate_speech_sentiment_hist(df: pd.DataFrame) -> DataFrame[SpeechSentimentHistSchema]:
    """Validate historical Congress.gov speech sentiment. Raises on failure."""
    try:
        SpeechSentimentHistSchema.validate(df, lazy=True)
        logger.info("Historical speech sentiment validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Historical speech sentiment validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class DistrictAnalyticsSchema(pa.DataFrameModel):
    """Per-district-year political inertia features from SQLite window queries on election history.

    `year` — election year [1976, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `win_streak` — number of consecutive prior cycles won by the same party (0 = first observed cycle) [0, 20]
    `swing_volatility` — population std-dev of dem_vote_share over the 4 prior cycles; NULL when fewer than 4 prior cycles available [0, ∞) (optional)
    """

    year: Series[int] = pa.Field(ge=1976, le=2030, alias=Col.YEAR)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    win_streak: Series[int] = pa.Field(ge=0, le=20)
    swing_volatility: Optional[Series[float]] = pa.Field(ge=0.0, nullable=True)

    class Config:
        coerce = True
        strict = False


def validate_district_analytics(df: pd.DataFrame) -> DataFrame[DistrictAnalyticsSchema]:
    """Validate district analytics features. Raises on failure."""
    try:
        DistrictAnalyticsSchema.validate(df, lazy=True)
        logger.info("District analytics validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "District analytics validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class FipsStatesSchema(pa.DataFrameModel):
    """Mapping of FIPS codes to postal abbreviations and full names for the 50 US states.

    `fips` — two-digit zero-padded FIPS state code (e.g. "06" for California)
    `state_po` — two-letter uppercase postal abbreviation (e.g. "CA")
    `state_name` — title-cased full state name (e.g. "California")
    """

    fips: Series[str] = pa.Field(str_matches=r"^[0-9]{2}$")
    state_po: Series[str] = pa.Field(str_matches=r"^[A-Z]{2}$")
    state_name: Series[str] = pa.Field(str_matches=r"^.{4,}$")

    class Config:
        strict = True


def validate_fips_states(df: pd.DataFrame) -> DataFrame[FipsStatesSchema]:
    """Validate FIPS states mapping. Raises on failure."""
    if len(df) != 50:
        raise ValueError(f"Expected 50 states, got {len(df)}")
    try:
        FipsStatesSchema.validate(df, lazy=True)
        logger.info("FIPS states validation passed (50 states)")
    except pa.errors.SchemaErrors as e:
        logger.error(
            "FIPS states validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class BillEmbeddingsSchema(pa.DataFrameModel):
    """K-Means cluster labels derived from bill title embeddings, one row per district-year.

    `year` — election year [2014, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `bill_topic_cluster` — K-Means cluster index assigned to the district's bill titles [0, ∞)
    """

    year: Series[int] = pa.Field(ge=2014, le=2030)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    bill_topic_cluster: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


def validate_bill_embeddings(df: pd.DataFrame) -> DataFrame[BillEmbeddingsSchema]:
    """Validate bill embeddings cluster output. Raises on failure."""
    try:
        BillEmbeddingsSchema.validate(df, lazy=True)
        logger.info("Bill embeddings validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Bill embeddings validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class PolicyTopicsCongressSchema(pa.DataFrameModel):
    """Per-district Groq-classified policy topic fractions for a single congress, without a year column.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `topic_healthcare` — fraction of sponsored bills classified as healthcare [0, 1]
    `topic_economy` — fraction of sponsored bills classified as economy [0, 1]
    `topic_defense` — fraction of sponsored bills classified as defense [0, 1]
    `topic_education` — fraction of sponsored bills classified as education [0, 1]
    `topic_environment` — fraction of sponsored bills classified as environment [0, 1]
    `topic_social` — fraction of sponsored bills classified as social policy [0, 1]
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    topic_healthcare: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_economy: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_defense: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_education: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_environment: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_social: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config:
        coerce = True
        strict = False


class PolicyTopicsSchema(pa.DataFrameModel):
    """Per-district-year Groq-classified policy topic fractions from Congress.gov bill titles.

    `year` — election year [2014, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `topic_healthcare` — fraction of sponsored bills classified as healthcare [0, 1]
    `topic_economy` — fraction of sponsored bills classified as economy [0, 1]
    `topic_defense` — fraction of sponsored bills classified as defense [0, 1]
    `topic_education` — fraction of sponsored bills classified as education [0, 1]
    `topic_environment` — fraction of sponsored bills classified as environment [0, 1]
    `topic_social` — fraction of sponsored bills classified as social policy [0, 1]
    """

    year: Series[int] = pa.Field(ge=2014, le=2030)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    topic_healthcare: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_economy: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_defense: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_education: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_environment: Series[float] = pa.Field(ge=0.0, le=1.0)
    topic_social: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config:
        coerce = True
        strict = False


def validate_policy_topics(df: pd.DataFrame) -> DataFrame[PolicyTopicsSchema]:
    """Validate policy topics output. Raises on failure."""
    try:
        PolicyTopicsSchema.validate(df, lazy=True)
        logger.info("Policy topics validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Policy topics validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class RatingsEnsembleSchema(pa.DataFrameModel):
    """Multi-rater consensus Democratic win probability from the Wikipedia ratings table (2026 only).

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `ratings_ensemble_prob` — ensemble probability of a Democratic win, averaged across raters [0, 1]
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    ratings_ensemble_prob: Series[float] = pa.Field(ge=0.0, le=1.0)

    class Config:
        coerce = True
        strict = False


def validate_ratings_ensemble(
    df: pd.DataFrame, min_districts: int = 50
) -> DataFrame[RatingsEnsembleSchema]:
    """Validate ratings ensemble output. Raises on failure."""
    if len(df) < min_districts:
        raise ValueError(
            f"Ratings ensemble too small: {len(df)} districts (expected >= {min_districts})"
        )
    try:
        RatingsEnsembleSchema.validate(df, lazy=True)
        logger.info("Ratings ensemble validation passed (%d districts)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Ratings ensemble validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]


class CosponsorshipCongressSchema(pa.DataFrameModel):
    """Per-district cosponsorship network features for a single congress, without a year column.

    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `cosponsor_reach` — number of unique other members who cosponsored this member's bills [0, ∞)
    `bipartisan_cosponsor_pct` — fraction of cosponsors from the opposing party [0, 1]
    `cosponsored_count` — number of other members' bills this member cosponsored [0, ∞)
    """

    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    cosponsor_reach: Series[int] = pa.Field(ge=0)
    bipartisan_cosponsor_pct: Series[float] = pa.Field(ge=0.0, le=1.0)
    cosponsored_count: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


class CosponsorshipSchema(pa.DataFrameModel):
    """Per-district-year cosponsorship network features from Congress.gov.

    `year` — election year [2014, 2030]
    `state_po` — two-letter postal state abbreviation
    `district` — congressional district number; 0 = at-large [0, 53]
    `cosponsor_reach` — number of unique other members who cosponsored this member's bills [0, ∞)
    `bipartisan_cosponsor_pct` — fraction of cosponsors from the opposing party [0, 1]
    `cosponsored_count` — number of other members' bills this member cosponsored [0, ∞)
    """

    year: Series[int] = pa.Field(ge=2014, le=2030)
    state_po: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 2}, alias=Col.STATE)
    district: Series[int] = pa.Field(ge=0, le=53, alias=Col.DISTRICT)
    cosponsor_reach: Series[int] = pa.Field(ge=0)
    bipartisan_cosponsor_pct: Series[float] = pa.Field(ge=0.0, le=1.0)
    cosponsored_count: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


def validate_cosponsorship(df: pd.DataFrame) -> DataFrame[CosponsorshipSchema]:
    """Validate cosponsorship network features. Raises on failure."""
    try:
        CosponsorshipSchema.validate(df, lazy=True)
        logger.info("Cosponsorship validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.error(
            "Cosponsorship validation FAILED: %d issue(s)\n%s",
            len(e.failure_cases),
            e.failure_cases.head(10).to_string(),
        )
        raise
    return df  # type: ignore[return-value]
