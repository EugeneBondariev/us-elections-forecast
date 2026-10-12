"""Compare hardcoded 2026 indicators in constants.py against live data.

Fetches UNEMPLOYMENT and ECON_GROWTH from FRED API.
Generic ballot requires manual check — RCP link is printed.

Requires:
    FRED_API_KEY env var — free at https://fred.stlouisfed.org/docs/api/api_key.html

Run from project root:
    python scripts/update_indicators.py
"""

import os
import sys

import requests

from src.constants import ECON_GROWTH, GENERIC_BALLOT, UNEMPLOYMENT

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
YEAR = 2026


def _fred(series_id: str, api_key: str, limit: int = 12) -> list[dict]:
    resp = requests.get(
        FRED_BASE,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["observations"]


def live_unemployment(api_key: str) -> float | None:
    """Most recent monthly unemployment rate at or before October of YEAR."""
    obs = _fred("UNRATE", api_key, limit=24)
    for o in obs:
        if o["value"] == ".":
            continue
        if o["date"] <= f"{YEAR}-10-31":
            return float(o["value"])
    return None


def live_gdp_growth(api_key: str) -> float | None:
    """YoY real GDP growth for the most recent available quarter of YEAR."""
    obs = _fred("GDPC1", api_key, limit=12)
    values = {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}

    # Try Q3 then Q2 then Q1 of YEAR
    for month in ("07", "04", "01"):
        current = f"{YEAR}-{month}-01"
        prior   = f"{YEAR - 1}-{month}-01"
        if current in values and prior in values:
            return round((values[current] / values[prior] - 1) * 100, 1)
    return None


def _status(live: float, hardcoded: float, tolerance: float) -> str:
    return "OK" if abs(live - hardcoded) <= tolerance else "*** UPDATE NEEDED ***"


def main() -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("Error: FRED_API_KEY not set.")
        print("Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    print(f"\nLive indicator check for {YEAR}")
    print("-" * 55)

    unemp_live = live_unemployment(api_key)
    unemp_hard = UNEMPLOYMENT.get(YEAR)
    if unemp_live is not None:
        print(f"Unemployment  hardcoded={unemp_hard}%   live={unemp_live}%   "
              f"{_status(unemp_live, unemp_hard, 0.3)}")
    else:
        print(f"Unemployment  hardcoded={unemp_hard}%   live=unavailable")

    gdp_live = live_gdp_growth(api_key)
    gdp_hard = ECON_GROWTH.get(YEAR)
    if gdp_live is not None:
        print(f"GDP growth    hardcoded={gdp_hard}%   live={gdp_live}%   "
              f"{_status(gdp_live, gdp_hard, 0.4)}")
    else:
        print(f"GDP growth    hardcoded={gdp_hard}%   live=unavailable")

    print(f"\nGeneric ballot  hardcoded={GENERIC_BALLOT.get(YEAR)}")
    print("  Check current average — search 'RealClearPolitics 2026 generic congressional ballot'")
    print("  or visit https://www.realclearpolitics.com/epolls/other/generic_congressional_vote/")

    print("\nTo update: edit src/constants.py and change the YEAR=2026 entries.")


if __name__ == "__main__":
    main()
