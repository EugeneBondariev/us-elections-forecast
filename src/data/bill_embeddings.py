"""Bill title embeddings and K-Means district topic clusters.

Embeds each congressional district's sponsored bill titles with sentence-transformers
(all-MiniLM-L6-v2) and clusters the resulting district-level vectors with K-Means.
The cluster label captures the broad policy focus of a district's representative —
a structural signal that persists across cycles independently of electoral environment.

Sponsored bill objects (title + number + type) are cached per congress in
data/external/sponsored_bills_{congress}.json. This cache is also consumed by
policy_topics.py and cosponsorship.py so the three modules share one set of API calls.

Training flow:
    1. For each congress 113–119, fetch sponsored bills (API or cache).
    2. Embed titles and average per district.
    3. Fit K-Means on congresses 113–118 (historical only, no leakage).
    4. Apply to all including congress 119 (2026).
    5. Save KMeans model to models/bill_embeddings_kmeans.pkl.
    6. Write data/interim/bill_embeddings.csv (year, state_po, district, bill_topic_cluster).

Run:
    python -m src.data.bill_embeddings
or:
    make bill-embeddings
"""

from __future__ import annotations

import json
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd

from src.data.speech_sentiment import fetch_house_members
from src.paths import DataFile, ensure

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # optional dependency — tests skip via pytest.importorskip
    SentenceTransformer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_sponsored_bills_for_congress",
    "embed_titles",
    "build_district_embeddings",
    "build_all_embeddings",
]

_N_CLUSTERS = 8
_EMBED_MODEL = "all-MiniLM-L6-v2"
_FIRST_CONGRESS = 113
_DEFAULT_CONGRESS = 119
_ELECTION_YEAR_OFFSET = 1788
_BASE_URL = "https://api.congress.gov/v3"
_BILL_FETCH_LIMIT = 20
_REQUEST_TIMEOUT = 30


def fetch_sponsored_bills_for_congress(
    congress: int,
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {bioguide_id: [{title, number, type, congress}, ...]} for a congress.

    Loads from data/external/sponsored_bills_{congress}.json if present.
    Otherwise fetches the Congress.gov member list (from member cache if available)
    and calls sponsored-legislation for each valid House member.

    Stores full bill objects so policy_topics and cosponsorship can reuse this
    cache without extra API calls.
    """
    cache_path = DataFile.congress_sponsored_bills_cache(congress)
    if cache_path.exists():
        logger.info("Congress %d: loading sponsored bills from cache", congress)
        with open(cache_path) as f:
            return json.load(f)

    import requests

    members = fetch_house_members(
        congress=congress,
        api_key=api_key,
        cache_path=DataFile.congress_members_cache(congress),
        current_members_only=(congress == _DEFAULT_CONGRESS),
    )

    # Term-overlap filter for historical congresses (same logic as speech_sentiment)
    if congress < _DEFAULT_CONGRESS:
        c_start = 1787 + 2 * congress
        c_end = 1789 + 2 * congress
        members = [
            m
            for m in members
            if any(
                t.get("chamber") == "House of Representatives"
                and t.get("startYear", 9999) < c_end
                and t.get("endYear", 9999) > c_start
                for t in m.get("terms", {}).get("item", [])
            )
        ]

    valid = [
        m
        for m in members
        if m.get("state")
        and m.get("bioguideId")
        and m.get("district") is not None
        and (state_name_map is None or m["state"] in state_name_map)
    ]
    logger.info("Congress %d: %d valid House members to fetch", congress, len(valid))

    bills_by_bio: dict[str, list[dict[str, Any]]] = {}
    for i, member in enumerate(valid, 1):
        bio = member["bioguideId"]
        params: dict[str, str | int] = {"limit": _BILL_FETCH_LIMIT}
        if api_key:
            params["api_key"] = api_key
        try:
            resp = requests.get(
                f"{_BASE_URL}/member/{bio}/sponsored-legislation",
                params=params,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            raw_bills = resp.json().get("sponsoredLegislation", [])
            bills_by_bio[bio] = [
                {
                    "title": b.get("title", ""),
                    "number": b.get("number", ""),
                    "type": b.get("type", ""),
                    "congress": congress,
                }
                for b in raw_bills
                if b.get("title")
            ]
        except Exception as exc:
            logger.warning("Congress %d [%d/%d] %s: %s", congress, i, len(valid), bio, exc)
            bills_by_bio[bio] = []
        if i % 50 == 0:
            logger.info("Congress %d: fetched %d/%d members", congress, i, len(valid))

    ensure(cache_path).write_text(json.dumps(bills_by_bio))
    logger.info(
        "Congress %d: %d members, bills cached -> %s", congress, len(bills_by_bio), cache_path
    )
    return bills_by_bio


def _build_bio_to_district(
    congress: int,
    state_name_map: dict[str, str] | None,
) -> dict[str, tuple[str, int]]:
    """Map bioguide_id → (state_po, district) using the member JSON cache."""
    member_cache = DataFile.congress_members_cache(congress)
    if not member_cache.exists():
        logger.warning("Member cache missing for congress %d", congress)
        return {}
    with open(member_cache) as f:
        members = json.load(f)
    result: dict[str, tuple[str, int]] = {}
    for m in members:
        bio = m.get("bioguideId")
        state_raw = m.get("state")
        district = m.get("district")
        if not (bio and state_raw and district is not None):
            continue
        if state_name_map is not None:
            state = state_name_map.get(state_raw)
            if state is None:
                continue
        else:
            state = state_raw
        result[bio] = (state, int(district))
    return result


def embed_titles(titles: list[str]) -> np.ndarray | None:
    """Return mean sentence embedding for a list of titles; None if list is empty."""
    if not titles:
        return None
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers required — pip install sentence-transformers")
    model = SentenceTransformer(_EMBED_MODEL)
    vecs: np.ndarray = model.encode(titles, show_progress_bar=False, batch_size=64)
    return np.mean(vecs, axis=0)


def build_district_embeddings(
    congress: int,
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build per-district mean embedding vector for one congress.

    Batches all titles in one sentence-transformers call per congress for efficiency.
    Returns a DataFrame with columns: state_po, district, _embedding (ndarray, object dtype).
    Districts with no bill titles are excluded.
    """
    bills_by_bio = fetch_sponsored_bills_for_congress(
        congress=congress, api_key=api_key, state_name_map=state_name_map
    )
    bio_to_district = _build_bio_to_district(congress=congress, state_name_map=state_name_map)

    # Collect all titles across all members in one list for a single encode() call
    valid_bios: list[str] = [
        bio for bio, bills in bills_by_bio.items() if bills and bio in bio_to_district
    ]
    if not valid_bios:
        return pd.DataFrame(columns=["state_po", "district", "_embedding"])

    all_titles: list[str] = []
    slices: list[tuple[int, int]] = []
    idx = 0
    for bio in valid_bios:
        titles = [b["title"] for b in bills_by_bio[bio] if b.get("title")]
        slices.append((idx, idx + len(titles)))
        all_titles.extend(titles)
        idx += len(titles)

    if not all_titles:
        return pd.DataFrame(columns=["state_po", "district", "_embedding"])

    if SentenceTransformer is None:
        raise ImportError("sentence-transformers required — pip install sentence-transformers")
    model = SentenceTransformer(_EMBED_MODEL)
    all_vecs: np.ndarray = model.encode(all_titles, show_progress_bar=False, batch_size=64)
    logger.info(
        "Congress %d: encoded %d titles for %d members", congress, len(all_titles), len(valid_bios)
    )

    # Average per member, then average per district (handles multi-member districts)
    district_embeds: dict[tuple[str, int], list[np.ndarray]] = {}
    for bio, (start, end) in zip(valid_bios, slices):
        if end > start:
            member_vec = np.mean(all_vecs[start:end], axis=0)
            loc = bio_to_district[bio]
            district_embeds.setdefault(loc, []).append(member_vec)

    rows = []
    for (state, dist), vecs in district_embeds.items():
        rows.append({"state_po": state, "district": dist, "_embedding": np.mean(vecs, axis=0)})
    df = pd.DataFrame(rows)
    logger.info("Congress %d: %d districts with embeddings", congress, len(df))
    return df


def build_all_embeddings(
    start_congress: int = _FIRST_CONGRESS,
    end_congress: int = _DEFAULT_CONGRESS,
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build bill_topic_cluster for all congresses and return a combined DataFrame.

    K-Means is fitted on congresses 113–118 (historical) to avoid using 2026 bill
    content to define what the clusters ARE. The same model is then applied to all
    congresses including 119. Saves the fitted model to DataFile.BILL_EMBEDDINGS_MODEL
    for reproducible cluster labels on future runs.

    Returns: year, state_po, district, bill_topic_cluster.
    """
    frames: list[tuple[int, pd.DataFrame]] = []
    for congress in range(start_congress, end_congress + 1):
        year = _ELECTION_YEAR_OFFSET + 2 * congress
        logger.info("Congress %d → year %d: building district embeddings", congress, year)
        df = build_district_embeddings(
            congress=congress, api_key=api_key, state_name_map=state_name_map
        )
        if not df.empty:
            df.insert(0, "year", year)
            frames.append((congress, df))

    if not frames:
        logger.warning("No embeddings built — check API key and caches")
        return pd.DataFrame(columns=["year", "state_po", "district", "bill_topic_cluster"])

    # Gather historical embeddings to fit K-Means (no leakage: exclude end_congress = 2026)
    hist_vecs: list[np.ndarray] = [
        emb for congress, df in frames if congress < end_congress for emb in df["_embedding"]
    ]

    from sklearn.cluster import KMeans

    n_clusters = min(_N_CLUSTERS, len(hist_vecs) if hist_vecs else sum(len(df) for _, df in frames))
    fit_vecs = hist_vecs if hist_vecs else [emb for _, df in frames for emb in df["_embedding"]]

    if not fit_vecs:
        logger.warning("No embeddings to fit K-Means on")
        return pd.DataFrame(columns=["year", "state_po", "district", "bill_topic_cluster"])

    X_fit: np.ndarray = np.array(fit_vecs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_fit)
    logger.info(
        "K-Means fitted on %d district-year vectors (%d clusters)", len(fit_vecs), n_clusters
    )

    km_path = DataFile.BILL_EMBEDDINGS_MODEL
    ensure(km_path)
    with open(km_path, "wb") as f:
        pickle.dump(kmeans, f)
    logger.info("K-Means model saved -> %s", km_path)

    # Assign cluster labels to all congresses
    result_frames: list[pd.DataFrame] = []
    for _, df in frames:
        X: np.ndarray = np.array(df["_embedding"].tolist())
        out = df[["year", "state_po", "district"]].copy()
        out["bill_topic_cluster"] = kmeans.predict(X).astype(int)
        result_frames.append(out)

    return pd.concat(result_frames, ignore_index=True)


def main() -> None:
    import logging as _logging
    import os

    from dotenv import load_dotenv

    load_dotenv()
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    if DataFile.BILL_EMBEDDINGS.exists():
        logger.info("bill_embeddings.csv already exists — delete to rebuild")
        return

    api_key = os.getenv("CONGRESS_API_KEY")
    if not api_key:
        logger.warning(
            "CONGRESS_API_KEY not set — requests will be rate-limited to 5/min. "
            "Register free at https://api.congress.gov/sign-up/"
        )

    from src.data.loaders import load_state_names

    state_name_map = load_state_names()
    result = build_all_embeddings(api_key=api_key, state_name_map=state_name_map)

    if not result.empty:
        result.to_csv(ensure(DataFile.BILL_EMBEDDINGS), index=False)
        logger.info(
            "Bill embeddings saved -> %s (%d rows, %d congresses)",
            DataFile.BILL_EMBEDDINGS,
            len(result),
            result["year"].nunique(),
        )


if __name__ == "__main__":
    main()
