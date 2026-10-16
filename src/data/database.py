"""SQLite helpers: source-data loaders and forecast persistence.

Source-data loaders (drop-in replacements for src/loaders.py):
    load_house_from_db(db_path)    — same output as load_clean_house_data()
    load_approval_from_db(db_path) — same output as load_approval()

Forecast persistence (used by api.py and /refresh):
    save_forecast(result)          — write latest forecast to DB, return run_id
    load_latest_forecast()         — read most recent forecast from DB, or None
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import PRES_PARTY, Col, Val
from src.paths import DataFile
from src.types import ForecastResult

logger = logging.getLogger(__name__)


_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at       TEXT    NOT NULL,
    n_districts  INTEGER NOT NULL,
    seats_sim    TEXT    NOT NULL,
    districts    TEXT    NOT NULL
)
"""


def save_forecast(result: ForecastResult, db_path: Path = DataFile.DB) -> int:
    """Persist a run_forecast() result dict to SQLite. Returns the new run_id."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_RUNS)
        cur = conn.execute(
            "INSERT INTO forecast_runs (run_at, n_districts, seats_sim, districts) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                int(result["n_districts"]),
                json.dumps(result["seats_sim"].tolist()),
                result["districts"].to_json(orient="records"),
            ),
        )
        conn.commit()
        run_id = cur.lastrowid
        if run_id is None:
            raise RuntimeError("INSERT did not return a row id")
        logger.info("Forecast saved to %s (run_id=%d)", db_path, run_id)
        return run_id
    finally:
        conn.close()


def load_latest_forecast(db_path: Path = DataFile.DB) -> ForecastResult | None:
    """Load the most recent saved forecast from SQLite. Returns None if none exists."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        try:
            row = conn.execute(
                "SELECT run_id, run_at, n_districts, seats_sim, districts "
                "FROM forecast_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None  # DB unreadable or table not yet created
        if row is None:
            return None
        run_id, run_at, n_districts, seats_sim_json, districts_json = row
        logger.info("Loaded forecast from DB (run_id=%d, run_at=%s)", run_id, run_at)
        return {
            "n_districts": n_districts,
            "seats_sim": np.array(json.loads(seats_sim_json)),
            "districts": pd.read_json(pd.io.common.StringIO(districts_json), orient="records"),  # type: ignore[call-overload,attr-defined]
        }
    finally:
        conn.close()


def load_house_from_db(db_path: Path) -> pd.DataFrame:
    """Load and clean house data from SQLite — same output as load_clean_house_data."""
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
        df = df[~df[col]]

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


def load_approval_from_db(db_path: Path) -> pd.DataFrame:
    """Load approval ratings from SQLite — same output as load_approval."""
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
