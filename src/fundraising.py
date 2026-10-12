from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = ["load_fundraising", "fetch_and_cache"]

FEC_BASE = "https://api.open.fec.gov/v1"
DEFAULT_YEARS = [2014, 2016, 2018, 2020, 2022, 2024, 2026]


def _fetch_year(api_key: str, year: int) -> list[dict]:
    records = []
    page = 1
    while True:
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{FEC_BASE}/candidates/totals/",
                    params={
                        "api_key": api_key,
                        "office": "H",
                        "election_year": year,
                        "page": page,
                        "per_page": 100,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                logger.warning("FEC API error year=%d page=%d attempt=%d: %s", year, page, attempt + 1, exc)
                if attempt == 2:
                    logger.error("Giving up on year=%d page=%d after 3 attempts", year, page)
                    data = None
                time.sleep(2 ** attempt)
        if data is None:
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            district_raw = str(r.get("district") or "0").lstrip("0") or "0"
            try:
                district = int(district_raw)
            except ValueError:
                district = 0
            records.append({
                "year": year,
                "state_po": r.get("state"),
                "district": district,
                "party": r.get("party"),
                "receipts": float(r.get("receipts") or 0),
            })

        pages = data.get("pagination", {}).get("pages", 1)
        if page >= pages:
            break
        page += 1
        time.sleep(0.12)

    logger.info("  %d: fetched %d candidate records", year, len(records))
    return records


def _save(records: list[dict], cache_path: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df = df[df["state_po"].notna()]
    dem = (
        df[df["party"] == "DEM"]
        .groupby(["year", "state_po", "district"])["receipts"]
        .max().reset_index(name="dem_receipts")
    )
    rep = (
        df[df["party"] == "REP"]
        .groupby(["year", "state_po", "district"])["receipts"]
        .max().reset_index(name="rep_receipts")
    )
    result = dem.merge(rep, on=["year", "state_po", "district"], how="outer").fillna(0)
    total = result["dem_receipts"] + result["rep_receipts"]
    result["fundraising_dem_share"] = (result["dem_receipts"] / total.clip(lower=1)).where(total > 0, 0.5)
    result = result[["year", "state_po", "district", "fundraising_dem_share"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)


def fetch_and_cache(cache_path, api_key: str, years=None) -> pd.DataFrame:
    """Fetch House fundraising totals from FEC API and cache to CSV."""
    if years is None:
        years = DEFAULT_YEARS
    cache_path = Path(cache_path)

    # Load existing cache before overwriting so other years are preserved.
    existing = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()

    all_records = []
    for year in years:
        all_records.extend(_fetch_year(api_key, year))
        _save(all_records, cache_path)

    new = pd.read_csv(cache_path)
    if not existing.empty:
        other_years = existing[~existing["year"].isin(years)]
        result = pd.concat([other_years, new], ignore_index=True).sort_values(["year", "state_po", "district"])
        result.to_csv(cache_path, index=False)
    else:
        result = new

    logger.info("Cached %d district-year records to %s", len(result), cache_path)
    return result


def load_fundraising(
    cache_path: str | Path,
    api_key: str | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame | None:
    """Load fundraising data from cache if available, otherwise fetch from FEC API.

    Returns None if no API key is provided and no cache exists — callers treat
    None as "feature unavailable" and skip fundraising columns.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info("Loading fundraising from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    if api_key is None:
        api_key = os.getenv("FEC_API_KEY")
    if api_key is None:
        logger.warning("FEC_API_KEY not set and no cache found — skipping fundraising feature")
        return None

    logger.info("Fetching fundraising data from FEC API for years %s", years or DEFAULT_YEARS)
    return fetch_and_cache(cache_path, api_key, years)
