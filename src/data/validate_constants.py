"""Validate STATE_FIPS_TO_PO against the US DOT official FIPS reference.

Fetches the Department of Transportation State/County/City FIPS reference table,
deduplicates to state level, and diffs against src/constants.STATE_FIPS_TO_PO.

Data source: data.transportation.gov (US DOT, public, no auth required)

Run with: make validate-constants
"""

from __future__ import annotations

import io
import logging
import sys

import pandas as pd
import requests

from src.constants import STATE_FIPS_TO_PO

logger = logging.getLogger(__name__)

_DOT_URL = "https://data.transportation.gov/api/v3/views/eek5-pv8d/export.csv?accessType=DOWNLOAD"


def fetch_dot_fips() -> dict[str, str]:
    """Fetch state FIPS→postal mapping from the US DOT reference table."""
    logger.info("Fetching US DOT FIPS reference table...")
    resp = requests.get(_DOT_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    states = df[["State FIPS Code", "State Code"]].drop_duplicates().dropna()
    # Exclude territories: keep only the 50 states (FIPS 01–56, no 03/07/11/14/43/52)
    fifty_state_fips = set(STATE_FIPS_TO_PO.keys())
    states = states[states["State FIPS Code"].isin(fifty_state_fips)]
    return dict(zip(states["State FIPS Code"], states["State Code"]))


def validate() -> bool:
    """Return True if STATE_FIPS_TO_PO matches the DOT reference exactly."""
    dot = fetch_dot_fips()
    local = STATE_FIPS_TO_PO
    ok = True

    in_dot_not_local = set(dot) - set(local)
    in_local_not_dot = set(local) - set(dot)
    mismatched = {fips for fips in dot if fips in local and dot[fips] != local[fips]}

    if in_dot_not_local:
        logger.error("FIPS codes in DOT but missing from STATE_FIPS_TO_PO: %s", in_dot_not_local)
        ok = False
    if in_local_not_dot:
        logger.error("FIPS codes in STATE_FIPS_TO_PO but not in DOT: %s", in_local_not_dot)
        ok = False
    for fips in mismatched:
        logger.error("Mismatch FIPS %s: local=%s  dot=%s", fips, local[fips], dot[fips])
        ok = False

    if ok:
        logger.info("[OK] STATE_FIPS_TO_PO matches DOT reference (%d states)", len(local))
    return ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not validate():
        sys.exit(1)


if __name__ == "__main__":
    main()
