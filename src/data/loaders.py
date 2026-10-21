from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from src.constants import (
    GENERIC_BALLOT,
    OPEN_SEATS_2026,
    PRES_PARTY,
    Col,
    Val,
)
from src.data.validators import (
    ApprovalDataSchema,
    HouseDataSchema,
    validate_approval_data,
    validate_house_data,
    validate_pres_lean,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_fips_states",
    "load_state_names",
    "load_pres_lean",
    "load_clean_house_data",
    "load_cook_ratings",
    "load_approval",
    "load_indicators",
    "load_generic_ballot",
    "load_poll_sigma",
    "load_open_seats",
    "load_census",
    "load_candidate_quality",
    "load_district_analytics",
    "load_gerrymandering",
    "load_speech_sentiment",
    "load_speech_sentiment_hist",
    "load_bill_embeddings",
    "load_policy_topics",
    "load_ratings_ensemble",
    "load_cosponsorship",
]


def load_fips_states() -> dict[str, str]:
    """Load state FIPS → postal abbreviation mapping from fips_states.csv.

    Run `make fips` to generate the file from the US DOT reference table.
    Raises FileNotFoundError if absent.
    """
    from src.paths import DataFile

    path = DataFile.FIPS_STATES
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make fips` to fetch it")
    df = pd.read_csv(path, dtype=str)
    from src.data.validators import validate_fips_states

    validate_fips_states(df)
    return dict(zip(df["fips"], df["state_po"]))


def load_state_names() -> dict[str, str]:
    """Load full state name → postal abbreviation mapping from fips_states.csv.

    Run `make fips` to generate the file. Raises FileNotFoundError if absent.
    """
    from src.paths import DataFile

    path = DataFile.FIPS_STATES
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make fips` to fetch it")
    df = pd.read_csv(path, dtype=str)
    from src.data.validators import validate_fips_states

    validate_fips_states(df)
    return dict(zip(df["state_name"], df["state_po"]))


def load_pres_lean(path_2008_2020: Path, path_2020_2024: Path) -> pd.DataFrame:
    """Parse presidential results by district from two Downballot CSVs.

    Returns long-format DataFrame: state_po, district, pres_year, pres_lean (Dem fraction 0-1).
    """

    def _parse(s: str) -> tuple[str, int]:
        state, dist = str(s).strip().split("-")
        return state, 0 if dist == "AL" else int(dist)

    def _load_and_parse(path: Path, skiprows: list[int], names: list[str]) -> pd.DataFrame:
        df = pd.read_csv(path, skiprows=skiprows, header=None, names=names)
        df = df.dropna(subset=["dist"])
        parsed = df["dist"].apply(_parse)
        df[Col.STATE] = [p[0] for p in parsed]
        df[Col.DISTRICT] = [p[1] for p in parsed]
        return df

    df1 = _load_and_parse(
        path_2008_2020,
        [0, 1],
        [
            "dist",
            "inc",
            "party",
            "biden_20",
            "trump_20",
            "clinton_16",
            "trump_16",
            "obama_12",
            "romney_12",
            "obama_08",
            "mccain_08",
        ],
    )
    df2 = _load_and_parse(
        path_2020_2024,
        [0, 1, 2],
        [
            "dist",
            "inc",
            "party",
            "harris_24",
            "trump_24",
            "margin_24",
            "biden_20b",
            "trump_20b",
            "margin_20",
        ],
    )

    year_col_map = {"obama_08": 2008, "obama_12": 2012, "clinton_16": 2016, "biden_20": 2020}
    df1_long = df1[[Col.STATE, Col.DISTRICT] + list(year_col_map.keys())].melt(
        id_vars=[Col.STATE, Col.DISTRICT], var_name="_col", value_name="_raw"
    )
    df1_long[Col.PRES_YEAR] = df1_long["_col"].map(year_col_map)
    df1_long[Col.PRES_LEAN] = pd.to_numeric(df1_long["_raw"], errors="coerce") / 100
    df1_long = df1_long.dropna(subset=[Col.PRES_LEAN])

    df2_long = df2[[Col.STATE, Col.DISTRICT, "harris_24"]].copy()
    df2_long[Col.PRES_YEAR] = 2024
    df2_long[Col.PRES_LEAN] = pd.to_numeric(df2_long["harris_24"], errors="coerce") / 100
    df2_long = df2_long.dropna(subset=[Col.PRES_LEAN])

    result = pd.concat(
        [
            df1_long[[Col.STATE, Col.DISTRICT, Col.PRES_YEAR, Col.PRES_LEAN]],
            df2_long[[Col.STATE, Col.DISTRICT, Col.PRES_YEAR, Col.PRES_LEAN]],
        ],
        ignore_index=True,
    )
    logger.info("Loaded presidential lean: %d district-year records", len(result))
    validate_pres_lean(result)
    return result


def load_clean_house_data(path: Path) -> DataFrame[HouseDataSchema]:
    """Load MIT Election Lab house CSV and apply standard filters (general, no special/runoff)."""
    logger.info("Loading house data from %s", path)
    df = pd.read_csv(path, low_memory=False)

    bool_cols = [Col.RUNOFF, Col.SPECIAL, Col.WRITE_IN, Col.UNOFFICIAL, Col.FUSION_TICKET]
    exclude_if_true = [Col.SPECIAL, Col.RUNOFF, Col.UNOFFICIAL, Col.WRITE_IN]

    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().str.upper() == "TRUE"

    df = df[df[Col.STAGE] == Val.GEN]
    for col in exclude_if_true:
        df = df[~df[col]]

    drop = [
        "state",
        Col.STATE_FIPS,
        Col.STATE_CEN,
        Col.STATE_IC,
        Col.FUSION_TICKET,
        Col.OFFICE,
        Col.MODE,
        Col.VERSION,
        Col.STAGE,
        Col.SPECIAL,
        Col.RUNOFF,
        Col.UNOFFICIAL,
        Col.WRITE_IN,
    ]
    df = df.drop(columns=drop, errors="ignore")
    df = df[df[Col.PARTY].notna()]

    # totalvotes == 0 shows up for genuinely uncontested races (MIT's dataset records
    # these with a 0 sentinel rather than a real count, e.g. FL-24 2016 - Frederica
    # Wilson ran unopposed). It's not bad data, but it's also unusable here: vote_share
    # would be a 0/0 division, and an uncontested race carries no dem_lean signal anyway.
    # Drop it explicitly (and loudly) instead of letting it slip through as a NaN that
    # happens to get dropna()'d three functions later for an unrelated-looking reason.
    uncontested = (df[Col.CANDIDATE_VOTES] >= 0) & (df[Col.TOTAL_VOTES] == 0)
    if uncontested.any():
        logger.info("Dropping %d uncontested-race row(s) with totalvotes == 0", uncontested.sum())
    df = df[(df[Col.CANDIDATE_VOTES] >= 0) & (df[Col.TOTAL_VOTES] > 0)]

    party_map = {
        Val.DEM_FARMER_LABOR: Val.DEMOCRAT,
        Val.FOGLIETTA_DEM: Val.DEMOCRAT,
        Val.INDEPENDENT_REP: Val.REPUBLICAN,
    }
    df[Col.PARTY] = df[Col.PARTY].replace(party_map)
    df = df.reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))
    validate_house_data(df)
    return df  # type: ignore[return-value]


def load_cook_ratings(path: Path) -> pd.DataFrame:
    """Load Cook Political cook.csv; return empty DataFrame with correct columns if file is absent."""
    if not path.exists():
        logger.info("cook.csv not found at %s — skipping Cook blend", path)
        return pd.DataFrame(columns=[Col.STATE, Col.DISTRICT, Col.COOK_PROB])
    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    logger.info("Loaded Cook ratings: %d districts", len(df))
    return df[[Col.STATE, Col.DISTRICT, Col.COOK_PROB]]


def load_approval(path: Path) -> DataFrame[ApprovalDataSchema]:
    """Load presidential approval TSV; sign-adjust so positive approval = Dem-favorable."""
    logger.info("Loading approval ratings from %s", path)
    approval = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[Col.START, Col.END, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL, Col.UNSURE],
    )
    approval[Col.YEAR] = pd.to_datetime(approval[Col.START]).dt.year
    # Validate on the raw 0-100 scale, before UNSURE is dropped and before the
    # sign-flip below — that's where "approve+disapprove+unsure ~= 100%" still means
    # what it says, and where a bad cell copy is easiest to catch.
    validate_approval_data(approval)
    approval = approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
    approval[Col.PRES_PARTY] = approval[Col.YEAR].map(PRES_PARTY)

    sign = np.where(approval[Col.PRES_PARTY] == Val.DEMOCRAT, 1, -1)
    approval[Col.PRES_APPROVAL] = approval[Col.PRES_APPROVAL] * sign
    approval[Col.PRES_DISAPPROVAL] = approval[Col.PRES_DISAPPROVAL] * (-sign)
    return approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]  # type: ignore[return-value]


def load_indicators() -> tuple[dict[int, float], dict[int, float]]:
    """Return (econ_growth_by_year, unemployment_by_year).

    Uses data/interim/indicators.csv — run `make indicators` (requires FRED_API_KEY) to build it.
    """
    from src.paths import DataFile

    path = DataFile.INDICATORS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `make indicators` (requires FRED_API_KEY) to fetch it"
        )

    df = pd.read_csv(path)
    econ = dict(zip(df[Col.YEAR].astype(int), df[Col.ECON_GROWTH].astype(float)))
    unemp = dict(zip(df[Col.YEAR].astype(int), df[Col.UNEMPLOYMENT].astype(float)))
    logger.info("Loaded indicators: %d years from %s", len(df), path)
    return econ, unemp


def load_generic_ballot() -> dict[int, float]:
    """Return year→margin dict (Dem minus Rep percentage points).

    Preference order:
      1. data/interim/poll_aggregates.csv  (from poll_aggregation.py — house-effects corrected)
      2. data/interim/generic_ballot.csv   (simple poll average)
      3. constants.GENERIC_BALLOT          (hardcoded fallback)
    """
    from src.paths import DataFile

    poll_agg_path = DataFile.POLL_AGGREGATES
    if poll_agg_path.exists():
        df = pd.read_csv(poll_agg_path)
        result = dict(zip(df[Col.YEAR].astype(int), df[Col.GENERIC_BALLOT].astype(float)))
        for year, val in GENERIC_BALLOT.items():
            result.setdefault(year, float(val))
        logger.info("Loaded generic ballot from poll aggregates: %d years", len(result))
        return result

    path = DataFile.GENERIC_BALLOT_CSV
    if not path.exists():
        logger.info("generic_ballot.csv absent — using constants.py fallback")
        return dict(GENERIC_BALLOT)

    df = pd.read_csv(path)
    result = dict(zip(df[Col.YEAR].astype(int), df["margin"].astype(float)))
    for year, val in GENERIC_BALLOT.items():
        result.setdefault(year, float(val))
    logger.info("Loaded generic ballot: %d years from %s", len(df), path)
    return result


def load_poll_sigma() -> dict[int, float]:
    """Return year→sigma dict from poll_aggregates.csv, or empty dict if absent."""
    from src.paths import DataFile

    path = DataFile.POLL_AGGREGATES
    if not path.exists():
        logger.info("poll_aggregates.csv absent — no sigma available")
        return {}

    df = pd.read_csv(path)
    result = dict(zip(df[Col.YEAR].astype(int), df[Col.GENERIC_BALLOT_SIGMA].astype(float)))
    logger.info("Loaded poll sigma: %d years from %s", len(result), path)
    return result


def load_census() -> pd.DataFrame | None:
    """Load ACS census features from disk; return None if file is absent."""
    from src.paths import DataFile

    path = DataFile.CENSUS
    if not path.exists():
        logger.info("census_acs.csv absent — census features will be skipped")
        return None

    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    from src.data.validators import validate_census

    validate_census(df)
    return df


def load_candidate_quality() -> pd.DataFrame | None:
    """Load candidate quality features from disk; return None if file is absent."""
    from src.paths import DataFile

    path = DataFile.CANDIDATE_QUALITY
    if not path.exists():
        logger.info("candidate_quality.csv absent — candidate quality features will be skipped")
        return None

    df = pd.read_csv(path, dtype={Col.DISTRICT: int, Col.YEAR: int})
    from src.data.validators import validate_candidate_quality

    validate_candidate_quality(df)
    return df


def load_district_analytics() -> pd.DataFrame | None:
    """Load SQL-derived win_streak/swing_volatility features; return None if absent."""
    from src.paths import DataFile

    path = DataFile.DISTRICT_ANALYTICS
    if not path.exists():
        logger.info("district_analytics.csv absent — win_streak/swing_volatility will be skipped")
        return None
    df = pd.read_csv(path, dtype={Col.DISTRICT: int, Col.YEAR: int})
    from src.data.validators import validate_district_analytics

    validate_district_analytics(df)
    return df


def load_gerrymandering() -> pd.DataFrame | None:
    """Load Polsby-Popper compactness features; return None if absent."""
    from src.paths import DataFile

    path = DataFile.GERRYMANDERING
    if not path.exists():
        logger.info("gerrymandering.csv absent — compactness features will be skipped")
        return None
    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    from src.data.validators import validate_gerrymandering

    validate_gerrymandering(df)
    return df


def load_speech_sentiment() -> pd.DataFrame | None:
    """Load Congress.gov speech sentiment features; return None if absent."""
    from src.paths import DataFile

    path = DataFile.SPEECH_SENTIMENT
    if not path.exists():
        logger.info("speech_sentiment.csv absent — sentiment features will be skipped")
        return None
    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    from src.data.validators import validate_speech_sentiment

    validate_speech_sentiment(df)
    return df


def load_speech_sentiment_hist() -> pd.DataFrame | None:
    """Load historical Congress.gov speech sentiment (congresses 113–118); return None if absent."""
    from src.paths import DataFile

    path = DataFile.SPEECH_SENTIMENT_HIST
    if not path.exists():
        logger.info(
            "speech_sentiment_hist.csv absent — historical sentiment features will be skipped"
        )
        return None
    df = pd.read_csv(path, dtype={"year": int, Col.DISTRICT: int})
    from src.data.validators import validate_speech_sentiment_hist

    validate_speech_sentiment_hist(df)
    return df


def load_bill_embeddings() -> pd.DataFrame | None:
    """Load K-Means district topic clusters; return None if absent."""
    from src.paths import DataFile

    path = DataFile.BILL_EMBEDDINGS
    if not path.exists():
        logger.info("bill_embeddings.csv absent — bill_topic_cluster will be skipped")
        return None
    df = pd.read_csv(path, dtype={"year": int, Col.DISTRICT: int, "bill_topic_cluster": int})
    from src.data.validators import validate_bill_embeddings

    validate_bill_embeddings(df)
    return df


def load_policy_topics() -> pd.DataFrame | None:
    """Load Groq-classified policy topic fractions; return None if absent."""
    from src.paths import DataFile

    path = DataFile.POLICY_TOPICS
    if not path.exists():
        logger.info("policy_topics.csv absent — topic_* features will be skipped")
        return None
    df = pd.read_csv(path, dtype={"year": int, Col.DISTRICT: int})
    from src.data.validators import validate_policy_topics

    validate_policy_topics(df)
    return df


def load_ratings_ensemble() -> pd.DataFrame | None:
    """Load multi-rater ensemble probabilities (2026 only); return None if absent."""
    from src.paths import DataFile

    path = DataFile.RATINGS_ENSEMBLE_CSV
    if not path.exists():
        logger.info("ratings_ensemble.csv absent — ratings_ensemble_prob will be skipped")
        return None
    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    from src.data.validators import validate_ratings_ensemble

    validate_ratings_ensemble(df)
    return df


def load_cosponsorship() -> pd.DataFrame | None:
    """Load cosponsorship network features; return None if absent."""
    from src.paths import DataFile

    path = DataFile.COSPONSORSHIP
    if not path.exists():
        logger.info("cosponsorship.csv absent — cosponsor features will be skipped")
        return None
    df = pd.read_csv(path, dtype={"year": int, Col.DISTRICT: int})
    from src.data.validators import validate_cosponsorship

    validate_cosponsorship(df)
    return df


def load_open_seats(year: int = 2026) -> set[tuple[str, int]]:
    """Return set of (state_po, district) open seats for the given year.

    Uses data/interim/open_seats_2026.csv if present and year==2026; falls back to constants.
    """
    from src.paths import DataFile

    if year != 2026:
        logger.info("load_open_seats: no CSV source for year %d — returning empty set", year)
        return set()

    path = DataFile.OPEN_SEATS_CSV
    if not path.exists():
        logger.info("open_seats_2026.csv absent — using constants.py fallback")
        return set(OPEN_SEATS_2026)

    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    seats: set[tuple[str, int]] = {
        (str(row[Col.STATE]), int(row[Col.DISTRICT])) for _, row in df.iterrows()
    }
    logger.info("Loaded %d open seats from %s", len(seats), path)
    return seats
