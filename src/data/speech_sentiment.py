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
    "load_speech_sentiment",
]

_BASE_URL = "https://api.congress.gov/v3"
_DEFAULT_CONGRESS = 119   # 119th Congress, 2025–2027
_PAGE_LIMIT = 250         # max per Congress.gov API page
_BILL_FETCH_LIMIT = 20    # sponsored bills per member (recent only)
_REQUEST_TIMEOUT = 30


def fetch_house_members(
    congress: int = _DEFAULT_CONGRESS,
    api_key: str | None = None,
    cache_path: Path | None = None,
) -> list[dict]:
    """Fetch all House members for the given congress with pagination.

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
) -> pd.DataFrame:
    """Build per-district sentiment and activity features from member bill titles.

    Congress.gov member dicts contain: bioguideId, state (2-letter postal), district
    (int or None for at-large), name, partyName, terms. Members missing state or
    bioguideId are silently skipped.
    """
    rows = []
    for member in members:
        state = member.get("state")
        district_raw = member.get("district")
        bioguide = member.get("bioguideId")
        if not state or not bioguide:
            continue
        district = int(district_raw) if district_raw is not None else 0
        titles = fetch_member_bills(bioguide_id=bioguide, api_key=api_key)
        sentiment = score_sentiment(titles)
        rows.append({
            "state_po": state,
            "district": district,
            "speech_sentiment": sentiment,
            "speech_count": len(titles),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["state_po", "district", "speech_sentiment", "speech_count"])
    df["district"] = df["district"].astype(int)
    logger.info(
        "Speech sentiment built: %d districts, sentiment range [%.3f, %.3f]",
        len(df),
        df["speech_sentiment"].min(),
        df["speech_sentiment"].max(),
    )
    return df


def load_speech_sentiment(cache_path: Path | None = None) -> pd.DataFrame | None:
    """Load the speech sentiment CSV, or return None if the file is absent.

    Validates the loaded DataFrame against SpeechSentimentSchema before returning.
    Pass cache_path to override the default DataFile.SPEECH_SENTIMENT location
    (useful in tests).
    """
    path = cache_path if cache_path is not None else DataFile.SPEECH_SENTIMENT
    if not Path(path).exists():
        logger.info("Speech sentiment file not found at %s — skipping feature", path)
        return None
    df = pd.read_csv(path, dtype={"district": int})
    from src.data import validators
    validators.validate_speech_sentiment(df)
    return df


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
    members = fetch_house_members(
        congress=_DEFAULT_CONGRESS,
        api_key=api_key,
        cache_path=DataFile.CONGRESS_MEMBERS_CACHE,
    )
    features = build_speech_features(members=members, api_key=api_key)
    features.to_csv(ensure(DataFile.SPEECH_SENTIMENT), index=False)
    logger.info(
        "Speech sentiment saved -> %s (%d districts)",
        DataFile.SPEECH_SENTIMENT, len(features),
    )


if __name__ == "__main__":
    main()
