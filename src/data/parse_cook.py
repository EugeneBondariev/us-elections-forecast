"""Fetch the Wikipedia House election ratings article and write data/interim/cook.csv.

Wikipedia's ratings article aggregates Cook Political Report, Sabato's Crystal Ball,
and several other raters. The Cook column is extracted and converted to a probability.

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
from src.paths import DataFile

logger = logging.getLogger(__name__)

RATINGS_URL = (
    "https://en.wikipedia.org/wiki/"
    "2026_United_States_House_of_Representatives_election_ratings"
)

_UA = "Mozilla/5.0 (compatible; us-elections-forecast/1.0; educational use)"

_STATE_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

COOK_PROB: dict[str, float] = {
    "solid-d":  0.97,
    "likely-d": 0.87,
    "lean-d":   0.72,
    "tossup":   0.50,
    "lean-r":   0.28,
    "likely-r": 0.13,
    "solid-r":  0.03,
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
        cook_col = next(
            (c for c in cols if re.search(r"\bcook\b", c, re.IGNORECASE)), None
        )
        if constituency_col and cook_col:
            return df, constituency_col, cook_col
    return None


def _fetch_html(url: str = RATINGS_URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


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
        rows.append({
            Col.STATE: state_po,
            Col.DISTRICT: district,
            "rating": rating_key,
            Col.COOK_PROB: COOK_PROB[rating_key],
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Fetching Wikipedia election ratings...")
    html = _fetch_html()
    rows = parse(html)

    seen: set[tuple[str, int]] = set()
    unique = []
    for row in rows:
        key = (row[Col.STATE], row[Col.DISTRICT])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    csv_path = DataFile.COOK
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[Col.STATE, Col.DISTRICT, "rating", Col.COOK_PROB])
        writer.writeheader()
        writer.writerows(unique)

    logger.info("Wrote %d districts to %s", len(unique), csv_path)
    counts = Counter(r["rating"] for r in unique)
    for rating in ["solid-d", "likely-d", "lean-d", "tossup", "lean-r", "likely-r", "solid-r"]:
        if counts[rating]:
            logger.info("  %s: %d", rating, counts[rating])


if __name__ == "__main__":
    main()
