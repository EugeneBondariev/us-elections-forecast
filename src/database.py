"""Alternative data loaders that read from SQLite instead of CSV.

These produce identical output to load_clean_house_data() and load_approval()
in src/loaders.py.  They are drop-in replacements, not modifications of the
existing loaders.

Typical usage:
    from src.database import load_house_from_db, load_approval_from_db
    from pathlib import Path

    DB = Path("data/elections.db")
    df      = load_house_from_db(DB)
    approval = load_approval_from_db(DB)
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import Col, Val, PRES_PARTY

logger = logging.getLogger(__name__)


def load_house_from_db(db_path) -> pd.DataFrame:
    """Load and clean house data from SQLite — same output as load_clean_house_data."""
    db_path = Path(db_path)
    logger.info("Loading house data from %s (table house_results)", db_path)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM house_results", conn)
    finally:
        conn.close()

    # --- replicate load_clean_house_data cleaning logic exactly ---
    bool_cols = [Col.RUNOFF, Col.SPECIAL, Col.WRITE_IN, Col.UNOFFICIAL, Col.FUSION_TICKET]
    exclude_if_true = [Col.SPECIAL, Col.RUNOFF, Col.UNOFFICIAL, Col.WRITE_IN]

    # SQLite persists Python booleans as integers (0/1) and NaN as NULL.
    # pandas reads them back as float64 (0.0, 1.0, NaN).  Normalise to bool:
    # treat 1 (or 1.0) as True, everything else (0, NaN/NULL) as False.
    for col in bool_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) == 1

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


def load_approval_from_db(db_path) -> pd.DataFrame:
    """Load approval ratings from SQLite — same output as load_approval."""
    db_path = Path(db_path)
    logger.info("Loading approval ratings from %s (table approval)", db_path)

    conn = sqlite3.connect(db_path)
    try:
        approval = pd.read_sql("SELECT * FROM approval", conn)
    finally:
        conn.close()

    # --- replicate load_approval cleaning logic exactly ---
    approval[Col.YEAR] = pd.to_datetime(approval[Col.START]).dt.year
    approval = approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
    approval[Col.PRES_PARTY] = approval[Col.YEAR].map(PRES_PARTY)

    sign = np.where(approval[Col.PRES_PARTY] == Val.DEMOCRAT, 1, -1)
    approval[Col.PRES_APPROVAL] = approval[Col.PRES_APPROVAL] * sign
    approval[Col.PRES_DISAPPROVAL] = approval[Col.PRES_DISAPPROVAL] * (-sign)
    return approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
