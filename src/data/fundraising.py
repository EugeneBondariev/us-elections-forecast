"""FEC campaign finance data loader and feature builder.

Raw fields fetched from /v1/candidates/totals/ (per candidate per cycle):
  receipts                              – Grand total of all money received
  individual_receipts                   – Total from individuals (itemized + unitemized)
  individual_itemized_contributions     – Donations >=200 (reported by name); large/bundled donors
  individual_unitemized_contributions   – Donations <200 (lump sum only); small/grassroots donors
                                          NOTE: always 0 from this endpoint; use /schedules/schedule_a/ instead
  other_political_committee_contributions – PAC, Super PAC, and party committee money
  candidate_contribution                – Candidate's own money (loans or direct contribution)
  disbursements                         – Total spending; not income but useful as activity proxy

Derived features saved to fundraising.csv (all expressed as Dem share of Dem+Rep combined):
  fundraising_dem_share   – Share of total receipts; overall money advantage (1980+)
  large_donor_dem_share   – Share of itemized contributions; big-donor/bundler support (2002+)
  pac_dem_share           – Share of PAC money; institutional/establishment backing (1986+)
  spending_dem_share      – Share of total disbursements; who is running the more active campaign
  surplus_dem_share       – Share of (receipts - disbursements); negative means Dem spent more than raised
                            relative to Republican; large positive = incumbent war chest dominance
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.constants import Col

logger = logging.getLogger(__name__)

__all__ = ["load_fundraising", "fetch_and_cache"]

FEC_BASE = "https://api.open.fec.gov/v1"

# House elections every 2 years; FEC electronic data reliable from 1994 onward,
# older years may be sparse but worth requesting
DEFAULT_YEARS = list(range(1980, 2027, 2))

# Fields to extract from each candidate record
_RAW_FIELDS = [
    "receipts",
    "individual_receipts",
    "individual_itemized_contributions",
    "individual_unitemized_contributions",
    "other_political_committee_contributions",
    "candidate_contribution",
    "disbursements",
]


def _fetch_year(api_key: str, year: int) -> list[dict[str, Any]]:
    records = []
    page = 1
    while True:
        for attempt in range(3):
            try:
                fec_params: dict[str, Any] = {
                    "api_key": api_key,
                    "office": "H",
                    "election_year": year,
                    "page": page,
                    "per_page": 100,
                }
                resp = requests.get(
                    f"{FEC_BASE}/candidates/totals/",
                    params=fec_params,
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
            record = {
                Col.YEAR: year,
                Col.STATE: r.get("state"),
                Col.DISTRICT: district,
                Col.PARTY: r.get("party"),
                "candidate_id": r.get("candidate_id"),
                "candidate_name": r.get("name"),
            }
            for field in _RAW_FIELDS:
                record[field] = float(r.get(field) or 0)
            records.append(record)

        pages = data.get("pagination", {}).get("pages", 1)
        if page >= pages:
            break
        page += 1
        time.sleep(0.12)

    logger.info("  %d: fetched %d candidate records", year, len(records))
    return records


def _save_raw(records: list[dict[str, Any]], raw_path: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df = df[df[Col.STATE].notna()].sort_values([Col.YEAR, Col.STATE, Col.DISTRICT, Col.PARTY])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    logger.info("Raw fundraising saved → %s (%d rows)", raw_path, len(df))


def _top_candidate_per_group(df_party: pd.DataFrame) -> pd.DataFrame:
    """Pick the single candidate with the highest receipts per (year, state, district).

    Using groupby(...).max() per field independently (the previous behavior) could
    stitch together columns from two *different* candidates in a multi-candidate
    primary (e.g. candidate A's receipts with candidate B's disbursements). Selecting
    one row via idxmax keeps every field internally consistent.
    """
    key_cols: list[str] = [Col.YEAR, Col.STATE, Col.DISTRICT]
    if df_party.empty:
        return df_party[key_cols + _RAW_FIELDS]
    idx = df_party.groupby([Col.YEAR, Col.STATE, Col.DISTRICT])["receipts"].idxmax()
    return df_party.loc[idx, key_cols + _RAW_FIELDS].reset_index(drop=True)


def _aggregate(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[df[Col.STATE].notna()]
    dem = _top_candidate_per_group(df[df[Col.PARTY] == "DEM"]).rename(
        columns={f: f"dem_{f}" for f in _RAW_FIELDS}
    )
    rep = _top_candidate_per_group(df[df[Col.PARTY] == "REP"]).rename(
        columns={f: f"rep_{f}" for f in _RAW_FIELDS}
    )
    result = dem.merge(rep, on=[Col.YEAR, Col.STATE, Col.DISTRICT], how="outer").fillna(0)

    def _share(dem_col: str, rep_col: str) -> pd.Series:
        # FEC's raw receipts/disbursements/contributions figures can occasionally come
        # back slightly negative (refunds or amendments exceeding the period's activity)
        # even though total is still comfortably positive - clip each side to zero first
        # so a negative numerator can't push the resulting share below 0 (and, since
        # dem/rep are both >= 0 here, dem <= dem+rep guarantees the result stays <= 1 too).
        dem_val = result[dem_col].clip(lower=0)
        rep_val = result[rep_col].clip(lower=0)
        total = dem_val + rep_val
        return (dem_val / total.clip(lower=1)).where(total > 0, 0.5)

    result["fundraising_dem_share"] = _share("dem_receipts", "rep_receipts")
    result["large_donor_dem_share"] = _share(
        "dem_individual_itemized_contributions",
        "rep_individual_itemized_contributions",
    )
    result["pac_dem_share"] = _share(
        "dem_other_political_committee_contributions",
        "rep_other_political_committee_contributions",
    )
    result["spending_dem_share"] = _share("dem_disbursements", "rep_disbursements")

    # Surplus = receipts - disbursements; share of combined surplus (clipped to avoid
    # division by zero when both candidates spent more than they raised)
    result["dem_surplus"] = result["dem_receipts"] - result["dem_disbursements"]
    result["rep_surplus"] = result["rep_receipts"] - result["rep_disbursements"]
    total_surplus = result["dem_surplus"] + result["rep_surplus"]
    result["surplus_dem_share"] = (
        (result["dem_surplus"] / total_surplus.clip(lower=1))
        .where(total_surplus.abs() > 0, 0.5)
    )

    return result[[
        Col.YEAR, Col.STATE, Col.DISTRICT,
        Col.FUNDRAISING_DEM_SHARE, Col.LARGE_DONOR_DEM_SHARE,
        Col.PAC_DEM_SHARE, Col.SPENDING_DEM_SHARE, Col.SURPLUS_DEM_SHARE,
    ]]


def fetch_and_cache(
    cache_path: Path,
    api_key: str,
    years: list[int] | None = None,
    raw_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch House fundraising totals from FEC API and cache to CSV."""
    from src.paths import DataFile
    if years is None:
        years = DEFAULT_YEARS
    cache_path = Path(cache_path)
    raw_path = Path(raw_path) if raw_path else DataFile.FUNDRAISING_RAW

    existing = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()

    all_records = []
    for i, year in enumerate(years, 1):
        logger.info("  Fetching FEC year %d (%d/%d)...", year, i, len(years))
        all_records.extend(_fetch_year(api_key, year))

    if not all_records:
        logger.warning("No records fetched")
        return existing

    _save_raw(all_records, raw_path)

    new = _aggregate(all_records)

    if not existing.empty:
        other_years = existing[~existing["year"].isin(years)]
        result = (
            pd.concat([other_years, new], ignore_index=True)
            .sort_values(["year", "state_po", "district"])
        )
    else:
        result = new

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    logger.info("Cached %d district-year records → %s", len(result), cache_path)
    return result


def load_fundraising(
    cache_path: Path,
    api_key: str | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame | None:
    """Load fundraising data from cache if available, otherwise fetch from FEC API.

    Returns None if no API key is provided and no cache exists — callers treat
    None as "feature unavailable" and skip fundraising columns.
    """
    if cache_path.exists():
        logger.info("Loading fundraising from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    if api_key is None:
        api_key = os.getenv("DATA_GOV_API_KEY") or os.getenv("FEC_API_KEY")
    if api_key is None:
        logger.error("No API key found and no cache — fundraising features will be missing")
        return None

    logger.info("Fetching fundraising data from FEC API for years %s", years or DEFAULT_YEARS)
    return fetch_and_cache(cache_path, api_key, years)
