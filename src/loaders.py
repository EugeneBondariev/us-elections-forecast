import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import Col, Val, PRES_PARTY

logger = logging.getLogger(__name__)


def load_pres_lean(path_2008_2020, path_2020_2024) -> pd.DataFrame:
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

    records = []
    for _, row in df1.iterrows():
        for col_name, year in [("obama_08", 2008), ("obama_12", 2012), ("clinton_16", 2016), ("biden_20", 2020)]:
            try:
                records.append({
                    Col.STATE: row[Col.STATE],
                    Col.DISTRICT: row[Col.DISTRICT],
                    "pres_year": year,
                    Col.PRES_LEAN: float(row[col_name]) / 100,
                })
            except (ValueError, TypeError):
                pass
    for _, row in df2.iterrows():
        try:
            records.append({
                Col.STATE: row[Col.STATE],
                Col.DISTRICT: row[Col.DISTRICT],
                "pres_year": 2024,
                Col.PRES_LEAN: float(row["harris_24"]) / 100,
            })
        except (ValueError, TypeError):
            pass

    logger.info("Loaded presidential lean: %d district-year records", len(records))
    return pd.DataFrame(records)


def load_clean_house_data(path: str) -> pd.DataFrame:
    logger.info("Loading house data from %s", path)
    df = pd.read_csv(path, low_memory=False)

    bool_cols = [Col.RUNOFF, Col.SPECIAL, Col.WRITE_IN, Col.UNOFFICIAL, Col.FUSION_TICKET]
    exclude_if_true = [Col.SPECIAL, Col.RUNOFF, Col.UNOFFICIAL, Col.WRITE_IN]

    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().str.upper() == "TRUE"

    df = df[df[Col.STAGE] == Val.GEN]
    for col in exclude_if_true:
        df = df[df[col] == False]

    drop = [
        "state", Col.STATE_FIPS, Col.STATE_CEN, Col.STATE_IC, Col.FUSION_TICKET,
        Col.OFFICE, Col.MODE, Col.VERSION, Col.STAGE, Col.SPECIAL, Col.RUNOFF,
        Col.UNOFFICIAL, Col.WRITE_IN,
    ]
    df = df.drop(columns=drop, errors="ignore")
    df = df[df[Col.PARTY].notna()]
    df = df[(df[Col.CANDIDATE_VOTES] >= 0) & (df[Col.TOTAL_VOTES] >= 0)]

    party_map = {
        Val.DEM_FARMER_LABOR: Val.DEMOCRAT,
        Val.FOGLIETTA_DEM: Val.DEMOCRAT,
        Val.INDEPENDENT_REP: Val.REPUBLICAN,
    }
    df[Col.PARTY] = df[Col.PARTY].replace(party_map)
    df = df.reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))
    return df


def load_cook_ratings(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        logger.info("cook.csv not found at %s — skipping Cook blend", path)
        return pd.DataFrame(columns=[Col.STATE, Col.DISTRICT, Col.COOK_PROB])
    df = pd.read_csv(path, dtype={Col.DISTRICT: int})
    logger.info("Loaded Cook ratings: %d districts", len(df))
    return df[[Col.STATE, Col.DISTRICT, Col.COOK_PROB]]


def load_approval(path: str) -> pd.DataFrame:
    logger.info("Loading approval ratings from %s", path)
    approval = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[Col.START, "end", Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL, "unsure"],
    )
    approval[Col.YEAR] = pd.to_datetime(approval[Col.START]).dt.year
    approval = approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
    approval[Col.PRES_PARTY] = approval[Col.YEAR].map(PRES_PARTY)

    sign = np.where(approval[Col.PRES_PARTY] == Val.DEMOCRAT, 1, -1)
    approval[Col.PRES_APPROVAL] = approval[Col.PRES_APPROVAL] * sign
    approval[Col.PRES_DISAPPROVAL] = approval[Col.PRES_DISAPPROVAL] * (-sign)
    return approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
