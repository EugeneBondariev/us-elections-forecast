"""Fetch FRED economic indicators for House election years and write data/interim/indicators.csv.

Fetches GDPC1 (Q3 YoY real GDP growth) and UNRATE (October unemployment rate) for every
election year 1976–2026. For any year FRED has no data, the year is skipped — except 2026 GDP,
which uses GDP_2026_ESTIMATE from constants.py until FRED publishes the Q3 advance release.

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

from src.constants import GDP_2026_ESTIMATE
from src.paths import DataFile

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
ELECTION_YEARS = list(range(1976, 2028, 2))


def _fred_observations(
    series_id: str,
    api_key: str,
    limit: int = 700,
    observation_start: str = "1975-01-01",
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "limit": limit,
        "observation_start": observation_start,
    }
    resp = requests.get(FRED_BASE, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()["observations"]


def fetch_indicator(
    series_id: str,
    api_key: str,
    months: tuple[str, ...],
    label: str,
    limit: int = 700,
    yoy: bool = False,
) -> dict[int, float]:
    """Fetch a FRED series and return {election_year: value}.

    yoy=True computes year-over-year % growth from level data (used for GDPC1).
    yoy=False returns the raw value for the preferred month (used for UNRATE).
    """
    obs = _fred_observations(series_id, api_key, limit=limit)
    values: dict[str, float] = {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}
    result: dict[int, float] = {}
    month_used: dict[int, str] = {}
    for year in ELECTION_YEARS:
        for month in months:
            if yoy:
                current, prior = f"{year}-{month}-01", f"{year - 1}-{month}-01"
                if current in values and prior in values:
                    result[year] = round((values[current] / values[prior] - 1) * 100, 1)
                    month_used[year] = month
                    break
            else:
                key = f"{year}-{month}-01"
                if key in values:
                    result[year] = values[key]
                    month_used[year] = month
                    break
    logger.info(
        "%s month used: %s",
        label,
        ", ".join(f"{y}={m}" for y, m in sorted(month_used.items())),
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error(
            "FRED_API_KEY not set — get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return

    logger.info("Fetching FRED GDPC1 and UNRATE...")
    fred_gdp = fetch_indicator(
        series_id="GDPC1", api_key=api_key, months=("07", "04", "01"), label="GDP", yoy=True
    )
    fred_unemployment = fetch_indicator(
        series_id="UNRATE",
        api_key=api_key,
        months=("10", "09", "08"),
        label="Unemployment",
    )
    logger.info("FRED: %d GDP years, %d unemployment years", len(fred_gdp), len(fred_unemployment))

    rows: list[dict[str, str | float]] = []
    logger.info("")
    logger.info("%-6s  %-10s %-10s", "Year", "GDP", "Unemployment")
    logger.info("-" * 30)

    for year in ELECTION_YEARS:
        gdp = fred_gdp.get(year)
        unemployment = fred_unemployment.get(year)

        if gdp is None and year == 2026:
            gdp = GDP_2026_ESTIMATE
            logger.warning(
                "2026 GDP not yet on FRED — using estimate %.1f%% (re-run after advance release)",
                gdp,
            )

        if gdp is None or unemployment is None:
            logger.info("%-6s  skipped (no FRED data)", year)
            continue

        logger.info("%-6s  %-10.1f %-10.1f", year, gdp, unemployment)
        rows.append({"year": year, "econ_growth": gdp, "unemployment": unemployment})

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
