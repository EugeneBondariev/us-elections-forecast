"""Fetch FRED economic indicators for House election years and write data/interim/indicators.csv.

Fetches GDPC1 (Q3 YoY real GDP growth) and UNRATE (October unemployment rate) for every
election year 1976–2026. Falls back to the hardcoded values in constants.py for any year
FRED has no data. Prints a comparison table and flags discrepancies.

Data sources:
    FRED GDPC1 — https://fred.stlouisfed.org/series/GDPC1
    FRED UNRATE — https://fred.stlouisfed.org/series/UNRATE
    API key: free at https://fred.stlouisfed.org/docs/api/api_key.html (env var FRED_API_KEY)

Run:
    python -m src.data.build_indicators
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.constants import ECON_GROWTH, UNEMPLOYMENT
from src.paths import DataFile

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
ELECTION_YEARS = list(range(1976, 2028, 2))

GDP_DISCREPANCY_THRESHOLD = 0.5
UNEMP_DISCREPANCY_THRESHOLD = 0.3


def _fred_observations(series_id: str, api_key: str, limit: int = 200) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "limit": limit,
    }
    resp = requests.get(FRED_BASE, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()["observations"]


def fetch_gdp_growth(api_key: str) -> dict[int, float]:
    """Return {year: Q3_YoY_growth_pct} for all available election years."""
    obs = _fred_observations("GDPC1", api_key)
    values: dict[str, float] = {
        o["date"]: float(o["value"]) for o in obs if o["value"] != "."
    }
    result: dict[int, float] = {}
    for year in ELECTION_YEARS:
        # Prefer Q3 (July), fall back to Q2 (April) then Q1 (January)
        for month in ("07", "04", "01"):
            current = f"{year}-{month}-01"
            prior = f"{year - 1}-{month}-01"
            if current in values and prior in values:
                growth = round((values[current] / values[prior] - 1) * 100, 1)
                result[year] = growth
                break
    return result


def fetch_unemployment(api_key: str) -> dict[int, float]:
    """Return {year: october_unemployment_rate} for all available election years."""
    obs = _fred_observations("UNRATE", api_key, limit=1000)
    values: dict[str, float] = {
        o["date"]: float(o["value"]) for o in obs if o["value"] != "."
    }
    result: dict[int, float] = {}
    for year in ELECTION_YEARS:
        # Prefer October, fall back to September then August
        for month in ("10", "09", "08"):
            key = f"{year}-{month}-01"
            if key in values:
                result[year] = values[key]
                break
    return result


def _flag(diff: float, threshold: float) -> str:
    return "***" if abs(diff) > threshold else ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.info("FRED_API_KEY not set — set it at https://fred.stlouisfed.org/docs/api/api_key.html")
        logger.info("Falling back entirely to constants.py values.")
        fred_gdp: dict[int, float] = {}
        fred_unemp: dict[int, float] = {}
    else:
        logger.info("Fetching FRED GDPC1 and UNRATE...")
        fred_gdp = fetch_gdp_growth(api_key)
        fred_unemp = fetch_unemployment(api_key)
        logger.info("FRED: %d GDP years, %d unemployment years", len(fred_gdp), len(fred_unemp))

    rows: list[dict[str, str | float]] = []
    logger.info("")
    logger.info("%-6s  %-10s %-10s %-6s   %-10s %-10s %-6s",
                "Year", "GDP_FRED", "GDP_const", "flag", "UNEMP_FRED", "UNEMP_const", "flag")
    logger.info("-" * 70)

    for year in ELECTION_YEARS:
        gdp_fred = fred_gdp.get(year)
        gdp_const = ECON_GROWTH.get(year)
        unemp_fred = fred_unemp.get(year)
        unemp_const = UNEMPLOYMENT.get(year)

        gdp_final = gdp_fred if gdp_fred is not None else gdp_const
        unemp_final = unemp_fred if unemp_fred is not None else unemp_const

        if gdp_final is None or unemp_final is None:
            logger.info("%-6s  skipped (no data)", year)
            continue

        gdp_flag = (
            _flag(gdp_fred - float(gdp_const or 0), GDP_DISCREPANCY_THRESHOLD)
            if gdp_fred is not None and gdp_const is not None
            else ""
        )
        unemp_flag = (
            _flag(unemp_fred - float(unemp_const or 0), UNEMP_DISCREPANCY_THRESHOLD)
            if unemp_fred is not None and unemp_const is not None
            else ""
        )

        logger.info(
            "%-6s  %-10s %-10s %-6s   %-10s %-10s %-6s",
            year,
            f"{gdp_fred:.1f}" if gdp_fred is not None else "n/a",
            f"{gdp_const:.1f}" if gdp_const is not None else "n/a",
            gdp_flag,
            f"{unemp_fred:.1f}" if unemp_fred is not None else "n/a",
            f"{unemp_const:.1f}" if unemp_const is not None else "n/a",
            unemp_flag,
        )
        rows.append({"year": year, "econ_growth": gdp_final, "unemployment": unemp_final})

    import csv
    out_path = DataFile.INDICATORS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "econ_growth", "unemployment"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("")
    logger.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
