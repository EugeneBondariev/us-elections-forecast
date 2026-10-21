"""LLM-based policy topic classification from bill titles.

Uses the Groq API to classify each congressional district's sponsored bill titles
into six policy topic fractions. The fractions are continuous features in [0, 1]
that sum to 1.0 and represent the relative legislative focus of a district's rep.

Topics: healthcare, economy, defense, education, environment, social.

Reuses the sponsored_bills_{congress}.json cache built by bill_embeddings.py so
no additional Congress.gov API calls are needed if that module has already run.

Results are cached per congress in data/external/policy_topics_{congress}.json
(maps (state_po, district) → {topic: fraction, ...}) so re-runs are cheap.

Graceful degradation: if the Groq API is unavailable or the parse fails, equal
fractions (1/6 each) are used rather than raising, keeping the pipeline runnable
without a Groq key.

Run:
    python -m src.data.policy_topics
or:
    make policy-topics
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
from groq import Groq

from src.data.bill_embeddings import _build_bio_to_district, fetch_sponsored_bills_for_congress
from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "TOPIC_KEYS",
    "classify_titles_with_groq",
    "build_policy_topics_for_congress",
    "build_all_policy_topics",
]

# Canonical topic names — must match Col.TOPIC_* values
TOPIC_KEYS: list[str] = [
    "healthcare",
    "economy",
    "defense",
    "education",
    "environment",
    "social",
]
_N_TOPICS = len(TOPIC_KEYS)
_EQUAL_FRACTIONS: dict[str, float] = {k: 1.0 / _N_TOPICS for k in TOPIC_KEYS}
_FIRST_CONGRESS = 113
_DEFAULT_CONGRESS = 119
_ELECTION_YEAR_OFFSET = 1788
_GROQ_MODEL = "llama-3.1-8b-instant"
_RATE_LIMIT_PAUSE = 2.0  # seconds between API calls (stay under 30 req/min free tier)


def classify_titles_with_groq(
    titles: list[str],
    api_key: str,
    model: str = _GROQ_MODEL,
) -> dict[str, float]:
    """Classify bill titles into topic fractions via Groq.

    Returns a dict mapping each topic key to a fraction in [0, 1] summing to 1.0.
    Falls back to equal fractions on parse failure or empty input.
    """
    if not titles:
        return dict(_EQUAL_FRACTIONS)

    sample = titles[:15]  # cap to keep token count low
    titles_text = "\n".join(f"- {t}" for t in sample)
    prompt = (
        "Given these US House bill titles sponsored by one representative, "
        "estimate what fraction of their legislative focus goes to each policy area. "
        "Return ONLY a JSON object with exactly these keys: "
        "healthcare, economy, defense, education, environment, social. "
        "Values must be non-negative floats that sum to 1.0. No other keys or text.\n\n"
        f"Bill titles:\n{titles_text}"
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=128,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        # Extract the JSON object even if there's surrounding text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in response: {raw!r}")
        parsed: dict[str, Any] = json.loads(raw[start:end])
        fractions: dict[str, float] = {}
        for key in TOPIC_KEYS:
            val = float(parsed.get(key, 0.0))
            fractions[key] = max(0.0, val)
        total = sum(fractions.values())
        if total <= 0:
            return dict(_EQUAL_FRACTIONS)
        return {k: v / total for k, v in fractions.items()}
    except Exception as exc:
        logger.warning("Groq classify failed (%s) — using equal fractions", exc)
        return dict(_EQUAL_FRACTIONS)


def build_policy_topics_for_congress(
    congress: int,
    groq_api_key: str,
    congress_api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
    groq_model: str = _GROQ_MODEL,
) -> pd.DataFrame:
    """Build per-district topic fractions for one congress.

    Reads sponsored bills from the shared sponsored_bills cache (built by
    bill_embeddings.py). Caches Groq results per congress so re-runs are cheap.

    Returns DataFrame: state_po, district, topic_healthcare, …, topic_social.
    """
    # Load or seed per-congress cache {"{state_po}_{district}": {topic: fraction}}
    cache_path = DataFile.congress_policy_topics_cache(congress)
    cached: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info(
            "Congress %d: %d district topic results loaded from cache", congress, len(cached)
        )

    bills_by_bio = fetch_sponsored_bills_for_congress(
        congress=congress, api_key=congress_api_key, state_name_map=state_name_map
    )
    bio_to_district = _build_bio_to_district(congress=congress, state_name_map=state_name_map)

    # Group bill titles by district
    district_titles: dict[tuple[str, int], list[str]] = {}
    for bio, bills in bills_by_bio.items():
        loc = bio_to_district.get(bio)
        if loc is None:
            continue
        titles = [b["title"] for b in bills if b.get("title")]
        district_titles.setdefault(loc, []).extend(titles)

    # Classify each district not already in cache
    newly_classified = 0
    for (state, dist), titles in district_titles.items():
        cache_key = f"{state}_{dist}"
        if cache_key in cached:
            continue
        fractions = classify_titles_with_groq(titles=titles, api_key=groq_api_key, model=groq_model)
        cached[cache_key] = fractions
        newly_classified += 1
        if newly_classified % 10 == 0:
            # Checkpoint cache every 10 calls to survive interrupts
            ensure(cache_path).write_text(json.dumps(cached))
        time.sleep(_RATE_LIMIT_PAUSE)

    ensure(cache_path).write_text(json.dumps(cached))
    logger.info(
        "Congress %d: %d districts classified (%d new), cached -> %s",
        congress,
        len(cached),
        newly_classified,
        cache_path,
    )

    rows = []
    for cache_key, fractions in cached.items():
        parts = cache_key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        state_po, dist_str = parts
        row: dict[str, Any] = {"state_po": state_po, "district": int(dist_str)}
        for key in TOPIC_KEYS:
            row[f"topic_{key}"] = fractions.get(key, 1.0 / _N_TOPICS)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["state_po", "district"] + [f"topic_{k}" for k in TOPIC_KEYS])
    df["district"] = df["district"].astype(int)
    return df


def build_all_policy_topics(
    start_congress: int = _FIRST_CONGRESS,
    end_congress: int = _DEFAULT_CONGRESS,
    groq_api_key: str = "",
    congress_api_key: str | None = None,
    state_name_map: dict[str, str] | None = None,
    groq_model: str = _GROQ_MODEL,
) -> pd.DataFrame:
    """Build topic fractions for all congresses and return a combined DataFrame.

    Returns: year, state_po, district, topic_healthcare, …, topic_social.
    """
    frames: list[pd.DataFrame] = []
    for congress in range(start_congress, end_congress + 1):
        year = _ELECTION_YEAR_OFFSET + 2 * congress
        logger.info("Congress %d → year %d: classifying policy topics", congress, year)
        df = build_policy_topics_for_congress(
            congress=congress,
            groq_api_key=groq_api_key,
            congress_api_key=congress_api_key,
            state_name_map=state_name_map,
            groq_model=groq_model,
        )
        if not df.empty:
            df.insert(0, "year", year)
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["year", "state_po", "district"] + [f"topic_{k}" for k in TOPIC_KEYS]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import logging as _logging
    import os

    from dotenv import load_dotenv

    load_dotenv()
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        logger.error(
            "GROQ_API_KEY not set — cannot classify topics. "
            "Register free at https://console.groq.com"
        )
        return

    if DataFile.POLICY_TOPICS.exists():
        logger.info("policy_topics.csv already exists — delete to rebuild")
        return

    congress_api_key = os.getenv("CONGRESS_API_KEY")
    from src.data.loaders import load_state_names

    state_name_map = load_state_names()
    groq_model = os.getenv("GROQ_MODEL", _GROQ_MODEL)

    result = build_all_policy_topics(
        groq_api_key=groq_api_key,
        congress_api_key=congress_api_key,
        state_name_map=state_name_map,
        groq_model=groq_model,
    )
    if not result.empty:
        result.to_csv(ensure(DataFile.POLICY_TOPICS), index=False)
        logger.info(
            "Policy topics saved -> %s (%d rows, %d congresses)",
            DataFile.POLICY_TOPICS,
            len(result),
            result["year"].nunique(),
        )


if __name__ == "__main__":
    main()
