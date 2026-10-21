"""Congressional floor speech and bill sponsorship sentiment features.

Uses the Congress.gov API (https://api.congress.gov) to fetch all House members for
the current congress and their recent sponsored bill titles. Computes VADER compound
sentiment (range -1 to +1) averaged over those titles as a proxy for how incumbents
are framing their public legislative agenda, plus a raw bill count as an activity proxy.

NLP signal: incumbents in difficult races tend to shift their public messaging — speech
sentiment and legislative activity correlate with the district's political environment.

No historical training data is built here: these features apply only to the 2026
prediction cycle and are merged in at forecast time, not during model training.

The module degrades gracefully: if the API key is absent the API still responds but at
5 requests/minute; if the network is unavailable load_speech_sentiment() returns None
so the pipeline continues without these features.

Free API key registration: https://api.congress.gov/sign-up/
Set CONGRESS_API_KEY in environment.

Run with:
    python -m src.data.speech_sentiment
or:
    make speech-sentiment

Output: data/interim/speech_sentiment.csv — state_po, district, speech_sentiment, speech_count
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_house_members",
    "fetch_member_bills",
    "score_sentiment",
    "build_speech_features",
    "build_historical_speech_features",
]

_BASE_URL = "https://api.congress.gov/v3"
_DEFAULT_CONGRESS = 119  # 119th Congress, 2025–2027
_FIRST_CONGRESS = 113  # 113th Congress → 2014 election (earliest with reliable bill data)
_ELECTION_YEAR_OFFSET = 1788  # election_year = offset + 2 × congress_number
_PAGE_LIMIT = 250  # max per Congress.gov API page
_BILL_FETCH_LIMIT = 20  # sponsored bills per member (recent only)
_REQUEST_TIMEOUT = 30


def fetch_house_members(
    congress: int = _DEFAULT_CONGRESS,
    api_key: str | None = None,
    cache_path: Path | None = None,
    current_members_only: bool = False,
) -> list[dict]:
    """Fetch all House members for the given congress with pagination.

    current_members_only: pass True only for the active congress — the API's
    currentMember filter excludes members who have since left, which is correct
    for the live cycle but wrong for historical congresses where we want everyone
    who served.

    Writes the raw list to cache_path on success and reads from it on subsequent
    calls to avoid redundant API round-trips.
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    members: list[dict] = []
    offset = 0
    while True:
        params: dict[str, str | int] = {
            "congress": congress,
            "chamber": "House",
            "limit": _PAGE_LIMIT,
            "offset": offset,
        }
        if current_members_only:
            params["currentMember"] = "true"
        if api_key:
            params["api_key"] = api_key
        resp = requests.get(f"{_BASE_URL}/member", params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        page = data.get("members", [])
        members.extend(page)
        if len(page) < _PAGE_LIMIT:
            break
        offset += _PAGE_LIMIT

    if cache_path is not None:
        ensure(Path(cache_path)).write_text(json.dumps(members))
    logger.info("Fetched %d House members for congress %d", len(members), congress)
    return members


def fetch_member_bills(bioguide_id: str, api_key: str | None = None) -> list[str]:
    """Fetch recent sponsored bill titles for one member.

    Returns a list of title strings. On any HTTP error, logs a warning and returns
    an empty list so a single bad bioguide_id doesn't abort the whole build.
    """
    params: dict[str, str | int] = {"limit": _BILL_FETCH_LIMIT}
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(
            f"{_BASE_URL}/member/{bioguide_id}/sponsored-legislation",
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        bills = resp.json().get("sponsoredLegislation", [])
        return [b.get("title", "") for b in bills if b.get("title")]
    except requests.RequestException as exc:
        logger.warning("Failed to fetch bills for %s: %s", bioguide_id, exc)
        return []


def score_sentiment(titles: list[str]) -> float:
    """Compute mean VADER compound score across bill titles.

    VADER compound ranges from -1 (most negative) to +1 (most positive).
    Returns 0.0 for an empty list rather than raising.
    """
    if not titles:
        return 0.0
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t)["compound"] for t in titles]
    return float(sum(scores) / len(scores))


def build_speech_features(
    members: list[dict],
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
    log_prefix: str = "",
) -> pd.DataFrame:
    """Build per-district sentiment and activity features from member bill titles.

    Congress.gov member dicts contain: bioguideId, state (full state name), district
    (int or None for at-large), name, partyName, terms. Members missing state or
    bioguideId are silently skipped. Pass state_name_map (full name → postal code) to
    translate state names to abbreviations and skip non-voting delegates (DC, PR, etc.).
    """
    rows = []
    valid = [
        m
        for m in members
        if m.get("state")
        and m.get("bioguideId")
        and m.get("district")
        is not None  # senators have no district; at-large reps have district=0
        and (state_name_map is None or m["state"] in state_name_map)
    ]
    total = len(valid)
    for i, member in enumerate(valid, 1):
        state_raw = member["state"]
        state = state_name_map[state_raw] if state_name_map else state_raw
        district_raw = member.get("district")
        assert district_raw is not None
        bioguide = member["bioguideId"]
        district = int(district_raw)
        titles = fetch_member_bills(bioguide_id=bioguide, api_key=api_key)
        sentiment = score_sentiment(titles)
        logger.info(
            "%s[%d/%d] %s-%d (%s): %d bills, sentiment=%.3f",
            log_prefix,
            i,
            total,
            state,
            district,
            bioguide,
            len(titles),
            sentiment,
        )
        rows.append(
            {
                "state_po": state,
                "district": district,
                "speech_sentiment": sentiment,
                "speech_count": len(titles),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["state_po", "district", "speech_sentiment", "speech_count"])
    df["district"] = df["district"].astype(int)
    # Multiple members may have served the same district in the 119th Congress
    # (resignation + special election replacement). Average their sentiments.
    df = df.groupby(["state_po", "district"], as_index=False).agg(
        speech_sentiment=("speech_sentiment", "mean"), speech_count=("speech_count", "sum")
    )
    logger.info(
        "Speech sentiment built: %d districts, sentiment range [%.3f, %.3f]",
        len(df),
        df["speech_sentiment"].min(),
        df["speech_sentiment"].max(),
    )
    return df


def build_historical_speech_features(
    start_congress: int = _FIRST_CONGRESS,
    end_congress: int = _DEFAULT_CONGRESS - 1,
    api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build speech sentiment for historical election cycles.

    Loops start_congress–end_congress, fetches members and their sponsored bills,
    scores VADER sentiment, and returns a DataFrame with a leading year column.
    Congress → election year: year = 1788 + 2 × congress_number.

    Historical congresses use all members who served (not currentMember filter).
    Results are cached per-congress under data/external/congress_members_{n}.json.
    """
    frames = []
    for congress in range(start_congress, end_congress + 1):
        election_year = _ELECTION_YEAR_OFFSET + 2 * congress
        cache = DataFile.congress_sentiment_cache(congress)
        if cache.exists():
            cached = pd.read_csv(cache, dtype={"year": int, "district": int})
            if "WY" in cached["state_po"].values:
                logger.info("Congress %d (%d): loading from cache", congress, election_year)
                frames.append(cached)
                continue
            logger.warning("Congress %d cache incomplete (WY missing) — re-fetching", congress)
            cache.unlink()
        logger.info("Congress %d → election year %d", congress, election_year)
        members = fetch_house_members(
            congress=congress,
            api_key=api_key,
            cache_path=DataFile.congress_members_cache(congress),
            current_members_only=False,
        )
        # Keep members whose House service overlapped with this congress's years.
        # Each term entry spans multiple congresses (startYear–endYear for a continuous
        # block of service), so we need an interval overlap check, not an exact match.
        # Congress N runs calendar years [1787+2N, 1789+2N); absent endYear = still serving.
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
        logger.info("Congress %d: %d members after term filter", congress, len(members))
        df = build_speech_features(
            members=members,
            api_key=api_key,
            state_name_map=state_name_map,
            log_prefix=f"[{election_year}] ",
        )
        df.insert(0, "year", election_year)
        df.to_csv(ensure(cache), index=False)
        frames.append(df)
        logger.info("Congress %d: %d districts, cached -> %s", congress, len(df), cache)
    if not frames:
        return pd.DataFrame(
            columns=["year", "state_po", "district", "speech_sentiment", "speech_count"]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import logging as _logging
    import os

    from dotenv import load_dotenv

    load_dotenv()
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    api_key = os.getenv("CONGRESS_API_KEY")
    if not api_key:
        logger.warning(
            "CONGRESS_API_KEY not set — requests will be rate-limited to 5/min. "
            "Register free at https://api.congress.gov/sign-up/"
        )
    from src.data.loaders import load_state_names

    state_name_map = load_state_names()
    if DataFile.SPEECH_SENTIMENT.exists():
        logger.info("speech_sentiment.csv already exists — skipping 2026 fetch")
    else:
        members = fetch_house_members(
            congress=_DEFAULT_CONGRESS,
            api_key=api_key,
            cache_path=DataFile.congress_members_cache(_DEFAULT_CONGRESS),
            current_members_only=True,
        )
        features = build_speech_features(
            members=members, api_key=api_key, state_name_map=state_name_map
        )
        features.to_csv(ensure(DataFile.SPEECH_SENTIMENT), index=False)
        logger.info(
            "Speech sentiment saved -> %s (%d districts)",
            DataFile.SPEECH_SENTIMENT,
            len(features),
        )
    if DataFile.SPEECH_SENTIMENT_HIST.exists():
        logger.info("speech_sentiment_hist.csv already exists — skipping historical fetch")
    else:
        hist = build_historical_speech_features(api_key=api_key, state_name_map=state_name_map)
        hist.to_csv(ensure(DataFile.SPEECH_SENTIMENT_HIST), index=False)
        logger.info(
            "Historical speech sentiment saved -> %s (%d rows, congresses %d–%d)",
            DataFile.SPEECH_SENTIMENT_HIST,
            len(hist),
            _FIRST_CONGRESS,
            _DEFAULT_CONGRESS - 1,
        )


if __name__ == "__main__":
    main()
