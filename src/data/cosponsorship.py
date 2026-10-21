"""Cosponsorship network features from the Congress.gov API.

For each House district, measures how effectively the member builds coalitions
across the chamber — a legislative-influence signal correlated with district
competitiveness and incumbency strength.

Three features:
    cosponsor_reach        — total unique colleagues who cosponsored the member's bills
                             (proxy for how broadly their legislative agenda is supported)
    bipartisan_cosponsor_pct — fraction of those cosponsors from the opposite party
                             (measures cross-aisle appeal)
    cosponsored_count      — total bills the member cosponsored themselves
                             (activity / engagement signal)

API calls per member (capped at 5 sponsored bills for cosponsor lookup):
    1 call  — cosponsored-legislation (pagination count only)
    1 call  — sponsored-legislation (bill list, reused from bill_embeddings cache if available)
    ≤5 calls — bill cosponsors endpoint for up to 5 sponsored bills

Results are cached per congress in data/external/sponsored_bills_{congress}.json
(shared with bill_embeddings) and a new cosponsorship_{congress}.json for the
computed features, so re-runs are cheap.

Run:
    python -m src.data.cosponsorship
or:
    make cosponsorship
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd

from src.constants import Col
from src.data.bill_embeddings import _build_bio_to_district, fetch_sponsored_bills_for_congress
from src.paths import EXTERNAL_DIR, DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_cosponsored_count",
    "fetch_bill_cosponsors",
    "build_cosponsorship_for_congress",
    "build_all_cosponsorship",
]

_BASE_URL = "https://api.congress.gov/v3"
_REQUEST_TIMEOUT = 30
_FIRST_CONGRESS = 113
_DEFAULT_CONGRESS = 119
_ELECTION_YEAR_OFFSET = 1788
_MAX_BILLS_FOR_COSPONSOR_LOOKUP = 5  # caps API calls per member at ~5 extra calls
_RATE_PAUSE = 0.2  # seconds between calls; Congress.gov free tier = 1000/hour


def _cosponsor_features_cache(congress: int) -> str:
    return str(EXTERNAL_DIR / f"cosponsorship_{congress}.json")


def fetch_cosponsored_count(
    bioguide_id: str,
    api_key: str | None = None,
) -> int:
    """Return total bills this member has cosponsored via the cosponsored-legislation endpoint.

    Reads only the pagination count (limit=1 request) — does not page through all bills.
    Returns 0 on any HTTP error.
    """
    import requests

    params: dict[str, str | int] = {"limit": 1}
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(
            f"{_BASE_URL}/member/{bioguide_id}/cosponsored-legislation",
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        pagination = data.get("pagination", {})
        return int(pagination.get("count", 0))
    except Exception as exc:
        logger.warning("fetch_cosponsored_count %s: %s", bioguide_id, exc)
        return 0


def fetch_bill_cosponsors(
    congress: int,
    bill_type: str,
    bill_number: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return cosponsor objects for one bill.

    Each cosponsor dict has at least: bioguideId, party (D/R/I), state.
    Returns empty list on any HTTP error.
    """
    import requests

    params: dict[str, str | int] = {"limit": 250}
    if api_key:
        params["api_key"] = api_key
    bill_type_lower = bill_type.lower()
    try:
        resp = requests.get(
            f"{_BASE_URL}/bill/{congress}/{bill_type_lower}/{bill_number}/cosponsors",
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("cosponsors", [])
    except Exception as exc:
        logger.warning(
            "fetch_bill_cosponsors %s %d/%s/%s: %s",
            bill_type,
            congress,
            bill_type_lower,
            bill_number,
            exc,
        )
        return []


def _compute_cosponsor_features(
    bioguide_id: str,
    bills: list[dict[str, Any]],
    sponsor_party: str,
    congress: int,
    api_key: str | None,
) -> dict[str, Any]:
    """Compute cosponsor_reach and bipartisan_cosponsor_pct for one member."""
    actionable = [b for b in bills if b.get("number") and b.get("type")][
        :_MAX_BILLS_FOR_COSPONSOR_LOOKUP
    ]

    all_cosponsor_ids: set[str] = set()
    opposite_count = 0

    for bill in actionable:
        cosponsors = fetch_bill_cosponsors(
            congress=congress,
            bill_type=bill["type"],
            bill_number=str(bill["number"]),
            api_key=api_key,
        )
        time.sleep(_RATE_PAUSE)
        for cs in cosponsors:
            cs_id = cs.get("bioguideId", "")
            if cs_id and cs_id != bioguide_id:
                all_cosponsor_ids.add(cs_id)
                cs_party = cs.get("party", "")
                if cs_party and cs_party != sponsor_party:
                    opposite_count += 1

    reach = len(all_cosponsor_ids)
    bipartisan = (opposite_count / reach) if reach > 0 else 0.0
    return {"cosponsor_reach": reach, "bipartisan_cosponsor_pct": bipartisan}


def build_cosponsorship_for_congress(
    congress: int,
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build cosponsorship features for all House districts in one congress.

    Uses the shared sponsored_bills cache (built by bill_embeddings) for the
    bill list, and a new cosponsorship_{congress}.json cache for computed features.
    Incremental: already-computed members are skipped on re-run.

    Returns DataFrame: state_po, district, cosponsor_reach, bipartisan_cosponsor_pct,
    cosponsored_count.
    """
    # Load per-congress feature cache (incremental checkpointing)
    feat_cache_path = _cosponsor_features_cache(congress)
    feat_cache: dict[str, dict[str, Any]] = {}
    try:
        with open(feat_cache_path) as f:
            feat_cache = json.load(f)
        logger.info("Congress %d: %d member features loaded from cache", congress, len(feat_cache))
    except FileNotFoundError:
        pass

    bills_by_bio = fetch_sponsored_bills_for_congress(
        congress=congress, api_key=api_key, state_name_map=state_name_map
    )
    bio_to_district = _build_bio_to_district(congress=congress, state_name_map=state_name_map)

    # Load member list to get party info
    member_cache = DataFile.congress_members_cache(congress)
    party_by_bio: dict[str, str] = {}
    if member_cache.exists():
        with open(member_cache) as f:
            members = json.load(f)
        for m in members:
            bio = m.get("bioguideId", "")
            party_name = m.get("partyName", "")
            if bio:
                party_by_bio[bio] = "D" if "democrat" in party_name.lower() else "R"

    newly_computed = 0
    for bio, bills in bills_by_bio.items():
        if bio not in bio_to_district:
            continue
        if bio in feat_cache:
            continue
        sponsor_party = party_by_bio.get(bio, "")
        cosponsor_feats = _compute_cosponsor_features(
            bioguide_id=bio,
            bills=bills,
            sponsor_party=sponsor_party,
            congress=congress,
            api_key=api_key,
        )
        cosponsored = fetch_cosponsored_count(bioguide_id=bio, api_key=api_key)
        time.sleep(_RATE_PAUSE)
        feat_cache[bio] = {**cosponsor_feats, "cosponsored_count": cosponsored}
        newly_computed += 1
        if newly_computed % 25 == 0:
            with open(feat_cache_path, "w") as f:
                json.dump(feat_cache, f)
            logger.info("Congress %d: checkpoint at %d members", congress, newly_computed)

    with open(feat_cache_path, "w") as f:
        json.dump(feat_cache, f)
    logger.info(
        "Congress %d: %d total members cached (%d new)", congress, len(feat_cache), newly_computed
    )

    rows: list[dict[str, Any]] = []
    district_feats: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for bio, feats in feat_cache.items():
        loc = bio_to_district.get(bio)
        if loc is None:
            continue
        district_feats.setdefault(loc, []).append(feats)

    for (state, dist), member_feats_list in district_feats.items():
        cosponsor_reach = int(sum(f["cosponsor_reach"] for f in member_feats_list))
        bipartisan = sum(f["bipartisan_cosponsor_pct"] for f in member_feats_list) / len(
            member_feats_list
        )
        cosponsored = int(sum(f["cosponsored_count"] for f in member_feats_list))
        rows.append(
            {
                "state_po": state,
                "district": dist,
                "cosponsor_reach": cosponsor_reach,
                "bipartisan_cosponsor_pct": bipartisan,
                "cosponsored_count": cosponsored,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "state_po",
                "district",
                Col.COSPONSOR_REACH,
                Col.BIPARTISAN_COSPONSOR_PCT,
                Col.COSPONSORED_COUNT,
            ]
        )
    df["district"] = df["district"].astype(int)
    logger.info("Congress %d: cosponsorship features for %d districts", congress, len(df))
    return df


def build_all_cosponsorship(
    start_congress: int = _FIRST_CONGRESS,
    end_congress: int = _DEFAULT_CONGRESS,
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build cosponsorship features for all congresses.

    Returns: year, state_po, district, cosponsor_reach, bipartisan_cosponsor_pct,
    cosponsored_count.
    """
    frames: list[pd.DataFrame] = []
    for congress in range(start_congress, end_congress + 1):
        year = _ELECTION_YEAR_OFFSET + 2 * congress
        logger.info("Congress %d → year %d: building cosponsorship features", congress, year)
        df = build_cosponsorship_for_congress(
            congress=congress, api_key=api_key, state_name_map=state_name_map
        )
        if not df.empty:
            df = df.copy()
            df.insert(0, "year", year)
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "year",
                "state_po",
                "district",
                Col.COSPONSOR_REACH,
                Col.BIPARTISAN_COSPONSOR_PCT,
                Col.COSPONSORED_COUNT,
            ]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import logging as _logging
    import os

    from dotenv import load_dotenv

    load_dotenv()
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    if DataFile.COSPONSORSHIP.exists():
        logger.info("cosponsorship.csv already exists — delete to rebuild")
        return

    api_key = os.getenv("CONGRESS_API_KEY")
    if not api_key:
        logger.warning(
            "CONGRESS_API_KEY not set — requests will be rate-limited to 5/min. "
            "Register free at https://api.congress.gov/sign-up/"
        )

    from src.data.loaders import load_state_names

    state_name_map = load_state_names()
    result = build_all_cosponsorship(api_key=api_key, state_name_map=state_name_map)

    if not result.empty:
        result.to_csv(ensure(DataFile.COSPONSORSHIP), index=False)
        logger.info(
            "Cosponsorship features saved -> %s (%d rows, %d congresses)",
            DataFile.COSPONSORSHIP,
            len(result),
            result["year"].nunique(),
        )


if __name__ == "__main__":
    main()
