from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import PRES_PARTY, Col, Val
from src.data.validators import validate_approval_data, validate_house_data

logger = logging.getLogger(__name__)

__all__ = [
    "load_pres_lean",
    "load_clean_house_data",
    "load_cook_ratings",
    "load_approval",
]


def load_pres_lean(path_2008_2020: str | Path, path_2020_2024: str | Path) -> pd.DataFrame:
    """Parse presidential results by district from two Downballot CSVs.

    Returns long-format DataFrame: state_po, district, pres_year, pres_lean (Dem fraction 0-1).
    """

    def _parse(s: str):
        state, dist = str(s).strip().split("-")
        return state, 0 if dist == "AL" else int(dist)

    df1 = pd.read_csv(
        path_2008_2020,
        skiprows=[0, 1],
        header=None,
        names=[
            "dist", "inc", "party",
            "biden_20", "trump_20",
            "clinton_16", "trump_16",
            "obama_12", "romney_12",
            "obama_08", "mccain_08",
        ],
    )
    df1 = df1.dropna(subset=["dist"])
    parsed1 = df1["dist"].apply(_parse)
    df1[Col.STATE] = [p[0] for p in parsed1]
    df1[Col.DISTRICT] = [p[1] for p in parsed1]

    df2 = pd.read_csv(
        path_2020_2024,
        skiprows=[0, 1, 2],
        header=None,
        names=[
            "dist", "inc", "party",
            "harris_24", "trump_24", "margin_24",
            "biden_20b", "trump_20b", "margin_20",
        ],
    )
    df2 = df2.dropna(subset=["dist"])
    parsed2 = df2["dist"].apply(_parse)
    df2[Col.STATE] = [p[0] for p in parsed2]
    df2[Col.DISTRICT] = [p[1] for p in parsed2]

    year_col_map = {"obama_08": 2008, "obama_12": 2012, "clinton_16": 2016, "biden_20": 2020}
    df1_long = (
        df1[[Col.STATE, Col.DISTRICT] + list(year_col_map.keys())]
        .melt(id_vars=[Col.STATE, Col.DISTRICT], var_name="col_name", value_name="raw")
    )
    df1_long["pres_year"] = df1_long["col_name"].map(year_col_map)
    df1_long[Col.PRES_LEAN] = pd.to_numeric(df1_long["raw"], errors="coerce") / 100
    df1_long = df1_long.dropna(subset=[Col.PRES_LEAN])

    df2_long = df2[[Col.STATE, Col.DISTRICT, "harris_24"]].copy()
    df2_long["pres_year"] = 2024
    df2_long[Col.PRES_LEAN] = pd.to_numeric(df2_long["harris_24"], errors="coerce") / 100
    df2_long = df2_long.dropna(subset=[Col.PRES_LEAN])

    result = pd.concat(
        [df1_long[[Col.STATE, Col.DISTRICT, "pres_year", Col.PRES_LEAN]],
         df2_long[[Col.STATE, Col.DISTRICT, "pres_year", Col.PRES_LEAN]]],
        ignore_index=True,
    )
    logger.info("Loaded presidential lean: %d district-year records", len(result))
    return result


def load_clean_house_data(path: str | Path) -> pd.DataFrame:
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
        "state", Col.STATE_FIPS, Col.STATE_CEN, Col.STATE_IC, Col.FUSION_TICKET,
        Col.OFFICE, Col.MODE, Col.VERSION, Col.STAGE, Col.SPECIAL, Col.RUNOFF,
        Col.UNOFFICIAL, Col.WRITE_IN,
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
    return df


def load_cook_ratings(path: str | Path) -> pd.DataFrame:
    """Load Cook Political cook.csv; return empty DataFrame with correct columns if file is absent."""
    path = Path(path)
    if not path.exists():
        logger.info("cook.csv not found at %s — skipping Cook blend", path)
        return pd.DataFrame(columns=[Col.STATE, Col.DISTRICT, Col.COOK_PROB])
    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    logger.info("Loaded Cook ratings: %d districts", len(df))
    return df[[Col.STATE, Col.DISTRICT, Col.COOK_PROB]]


def load_approval(path: str | Path) -> pd.DataFrame:
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
    return approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
