"""Fetch the Wikipedia House election ratings article and write data/interim/cook.csv
and data/interim/ratings_ensemble.csv.

Wikipedia's ratings article aggregates Cook Political Report, Sabato's Crystal Ball,
Inside Elections (formerly Rothenberg), and other raters. cook.csv contains only Cook
probabilities for the existing Cook-blend model; ratings_ensemble.csv blends all
available raters into a single consensus probability (ratings_ensemble_prob).

For each new election cycle, update RATINGS_URL to the new article, e.g.:
    https://en.wikipedia.org/wiki/2028_United_States_House_of_Representatives_election_ratings

Run:
    python -m src.data.parse_cook
or via fetch_data.py / make fetch (called automatically if cook.csv is missing).
"""

from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request
from collections import Counter
from typing import Any

import pandas as pd

from src.constants import Col
from src.paths import DataFile, ensure

# Rater name → regex pattern to find its column in the Wikipedia table.
# Order matters: first match wins. We skip CPVI since it's an index, not a rating.
_RATER_PATTERNS: dict[str, str] = {
    "cook": r"\bcook\b",
    "sabato": r"\bsabato\b",
    "inside": r"\binside\b",
    "rcp": r"\brcp\b|real\s+clear",
}

logger = logging.getLogger(__name__)

RATINGS_URL = (
    "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_election_ratings"
)

_UA = "Mozilla/5.0 (compatible; us-elections-forecast/1.0; educational use)"

_STATE_ABBR: dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

COOK_PROB: dict[str, float] = {
    "solid-d": 0.97,
    "likely-d": 0.87,
    "lean-d": 0.72,
    "tossup": 0.50,
    "lean-r": 0.28,
    "likely-r": 0.13,
    "solid-r": 0.03,
}

_WIKI_TO_KEY: dict[str, str] = {
    "solid d": "solid-d",
    "likely d": "likely-d",
    "lean d": "lean-d",
    "tossup": "tossup",
    "lean r": "lean-r",
    "likely r": "likely-r",
    "solid r": "solid-r",
}


def _normalize_rating(raw: str) -> str | None:
    """'Likely R (flip)' → 'likely-r'; returns None if not a known rating."""
    clean = re.sub(r"\([^)]*\)", "", raw).strip().lower()
    return _WIKI_TO_KEY.get(clean)


def _find_rater_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {rater_name: column_label} for all raters found in the table columns."""
    cols = list(df.columns)
    found: dict[str, str] = {}
    for rater, pattern in _RATER_PATTERNS.items():
        match = next((c for c in cols if re.search(pattern, c, re.IGNORECASE)), None)
        if match:
            found[rater] = match
    return found


def _parse_constituency(raw: str) -> tuple[str, int] | None:
    """
    'Alabama 2' → ('AL', 2)
    'Alaska at-large' → ('AK', 0)
    'Montana' → ('MT', 0)
    """
    raw = raw.strip()
    m = re.match(r"^(.+?)\s+at[- ]large$", raw, re.IGNORECASE)
    if m:
        abbr = _STATE_ABBR.get(m.group(1).strip())
        return (abbr, 0) if abbr else None
    m = re.match(r"^(.+?)\s+(\d+)$", raw)
    if m:
        abbr = _STATE_ABBR.get(m.group(1).strip())
        return (abbr, int(m.group(2))) if abbr else None
    abbr = _STATE_ABBR.get(raw)
    return (abbr, 0) if abbr else None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for col in df.columns:
        parts = [str(c) for c in col if not str(c).startswith("Unnamed")]
        new_cols.append(" | ".join(parts) if parts else str(col))
    df.columns = new_cols
    return df


def _find_table(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, str, str] | None:
    for df in tables:
        df = _flatten_columns(df)
        cols = list(df.columns)
        constituency_col = next(
            (c for c in cols if re.search(r"constituency", c, re.IGNORECASE)), None
        )
        cook_col = next((c for c in cols if re.search(r"\bcook\b", c, re.IGNORECASE)), None)
        if constituency_col and cook_col:
            return df, constituency_col, cook_col
    return None


def _fetch_html(url: str = RATINGS_URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_ensemble(html: str) -> list[dict[str, Any]]:
    """Parse all available raters and return a consensus ensemble probability.

    Extracts every rater column matched by _RATER_PATTERNS, converts each rating
    to a Cook-scale probability, and averages them. Districts where no raters have
    a valid rating are excluded. The result always includes cook_prob (if available)
    plus ratings_ensemble_prob.

    Returns a list of dicts: state_po, district, ratings_ensemble_prob, plus
    individual {rater}_prob columns for diagnostics.
    """
    tables = pd.read_html(io.StringIO(html), flavor="lxml", header=[0, 1])
    result = _find_table(tables)
    if result is None:
        tables = pd.read_html(io.StringIO(html), flavor="lxml", header=0)
        result = _find_table(tables)
    if result is None:
        raise ValueError(
            "No ratings table found — expected a table with 'Constituency' and "
            "'Cook' columns. The Wikipedia page structure may have changed."
        )
    df, constituency_col, _ = result
    rater_cols = _find_rater_columns(df)
    logger.info("Found rater columns: %s", list(rater_cols.keys()))

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        parsed = _parse_constituency(str(row[constituency_col]))
        if parsed is None:
            continue
        state_po, district = parsed

        rater_probs: dict[str, float] = {}
        for rater, col in rater_cols.items():
            key = _normalize_rating(str(row[col]))
            if key is not None:
                rater_probs[rater] = COOK_PROB[key]

        if not rater_probs:
            continue

        ensemble = sum(rater_probs.values()) / len(rater_probs)
        entry: dict[str, Any] = {
            Col.STATE: state_po,
            Col.DISTRICT: district,
            Col.RATINGS_ENSEMBLE: ensemble,
        }
        for rater, prob in rater_probs.items():
            entry[f"{rater}_prob"] = prob
        rows.append(entry)

    return rows


def parse(html: str) -> list[dict[str, Any]]:
    tables = pd.read_html(io.StringIO(html), flavor="lxml", header=[0, 1])
    result = _find_table(tables)
    if result is None:
        tables = pd.read_html(io.StringIO(html), flavor="lxml", header=0)
        result = _find_table(tables)
    if result is None:
        raise ValueError(
            "No ratings table found — expected a table with 'Constituency' and "
            "'Cook' columns. The Wikipedia page structure may have changed."
        )
    df, constituency_col, cook_col = result
    rows: list[dict] = []
    for _, row in df.iterrows():
        parsed = _parse_constituency(str(row[constituency_col]))
        if parsed is None:
            continue
        rating_key = _normalize_rating(str(row[cook_col]))
        if rating_key is None:
            continue
        state_po, district = parsed
        rows.append(
            {
                Col.STATE: state_po,
                Col.DISTRICT: district,
                "rating": rating_key,
                Col.COOK_PROB: COOK_PROB[rating_key],
            }
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Fetching Wikipedia election ratings...")
    html = _fetch_html()

    # --- Cook CSV (unchanged) ---
    rows = parse(html)
    seen: set[tuple[str, int]] = set()
    unique = []
    for row in rows:
        key = (row[Col.STATE], row[Col.DISTRICT])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    csv_path = DataFile.COOK
    with ensure(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[Col.STATE, Col.DISTRICT, "rating", Col.COOK_PROB])
        writer.writeheader()
        writer.writerows(unique)
    logger.info("Wrote %d districts to %s", len(unique), csv_path)
    counts = Counter(r["rating"] for r in unique)
    for rating in ["solid-d", "likely-d", "lean-d", "tossup", "lean-r", "likely-r", "solid-r"]:
        if counts[rating]:
            logger.info("  %s: %d", rating, counts[rating])

    # --- Ratings ensemble CSV ---
    ensemble_rows = parse_ensemble(html)
    seen_e: set[tuple[str, int]] = set()
    unique_e: list[dict] = []
    for row in ensemble_rows:
        key = (row[Col.STATE], row[Col.DISTRICT])
        if key not in seen_e:
            seen_e.add(key)
            unique_e.append(row)
    if unique_e:
        ensemble_df = pd.DataFrame(unique_e)
        ensemble_df.to_csv(ensure(DataFile.RATINGS_ENSEMBLE_CSV), index=False)
        logger.info(
            "Ratings ensemble: %d districts -> %s (raters: %s)",
            len(unique_e),
            DataFile.RATINGS_ENSEMBLE_CSV,
            [c for c in ensemble_df.columns if c.endswith("_prob")],
        )


if __name__ == "__main__":
    main()
