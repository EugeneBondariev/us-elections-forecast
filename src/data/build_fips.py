"""Fetch authoritative US state FIPS → postal abbreviation mapping from US DOT.

Data source: US Department of Transportation State/County/City FIPS Reference
Table (data.transportation.gov). Deduplicates to state level, keeps only the
50 US states, and writes a minimal two-column CSV.

Run with: make fips
Output: data/interim/fips_states.csv — fips, state_po
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

_DOT_URL = "https://data.transportation.gov/api/v3/views/eek5-pv8d/export.csv?accessType=DOWNLOAD"

# Known 50-state FIPS codes (01–56 with gaps for unassigned codes and territories).
# Used only to filter the DOT response — not as a data source in its own right.
_FIFTY_STATE_FIPS = {
    "01",
    "02",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
}


def fetch_fips_mapping() -> pd.DataFrame:
    """Download the DOT FIPS table and return a 50-row state-level DataFrame."""
    logger.info("Fetching US DOT FIPS reference table")
    resp = requests.get(_DOT_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    states = (
        df[["State FIPS Code", "State Code"]]
        .rename(columns={"State FIPS Code": "fips", "State Code": "state_po"})
        .drop_duplicates(subset=["fips"])
        .dropna()
    )
    states = states[states["fips"].isin(_FIFTY_STATE_FIPS)].reset_index(drop=True)
    logger.info("Fetched %d state FIPS codes", len(states))
    return states


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = fetch_fips_mapping()
    df.to_csv(ensure(DataFile.FIPS_STATES), index=False)
    logger.info("FIPS mapping saved → %s (%d states)", DataFile.FIPS_STATES, len(df))


if __name__ == "__main__":
    main()
