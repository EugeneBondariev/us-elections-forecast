"""Fetch incumbents not seeking re-election in 2026 and write data/interim/open_seats_2026.csv.

Parses the Wikipedia 2026 House elections article for the retirements/not-seeking-re-election
section. Falls back to OPEN_SEATS_2026 from constants.py if parsing fails.

Data source:
    https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections

Run:
    python -m src.data.build_open_seats
"""
from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request

import pandas as pd

from src.constants import OPEN_SEATS_2026
from src.data.parse_cook import _STATE_ABBR
from src.paths import DataFile

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; us-elections-forecast/1.0; educational use)"
_WIKI_URL = (
    "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections"
)

# Only the 50 US states (exclude DC, territories)
_US_STATES: frozenset[str] = frozenset(_STATE_ABBR.values())

_ORDINAL_MAP: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25,
    "twenty-sixth": 26, "twenty-seventh": 27, "twenty-eighth": 28,
    "twenty-ninth": 29, "thirtieth": 30, "thirty-first": 31,
    "thirty-second": 32, "thirty-third": 33, "thirty-fourth": 34,
    "thirty-fifth": 35, "thirty-sixth": 36, "thirty-seventh": 37,
    "thirty-eighth": 38, "thirty-ninth": 39, "fortieth": 40,
    "forty-first": 41, "forty-second": 42, "forty-third": 43,
    "forty-fourth": 44, "forty-fifth": 45, "forty-sixth": 46,
    "forty-seventh": 47, "forty-eighth": 48, "forty-ninth": 49,
    "fiftieth": 50, "fifty-first": 51, "fifty-second": 52,
}


def _parse_district(text: str) -> tuple[str, int] | None:
    """Parse district text in several formats:

    "California 11"              → ('CA', 11)
    "South Dakota at-large"      → ('SD', 0)
    "Wyoming at-large"           → ('WY', 0)
    "California's 26th"          → ('CA', 26)  (possessive/ordinal form)
    "Illinois's 4th congressional district" → ('IL', 4)
    """
    text = text.strip()

    # "State N" plain numeric
    m = re.match(r"^(.+?)\s+(\d+)$", text)
    if m:
        abbr = _STATE_ABBR.get(m.group(1).strip())
        if abbr:
            return (abbr, int(m.group(2)))

    # "State at-large" or "State's at-large"
    m = re.match(r"^(.+?)(?:'s?)?\s+at[- ]large$", text, re.IGNORECASE)
    if m:
        abbr = _STATE_ABBR.get(m.group(1).strip())
        if abbr:
            return (abbr, 0)

    # "State's Nth congressional district"
    m = re.match(
        r"^(.+?)(?:'s?)?\s+(\d+)(?:st|nd|rd|th)\s*(?:congressional\s+district)?$",
        text,
        re.IGNORECASE,
    )
    if m:
        abbr = _STATE_ABBR.get(m.group(1).strip())
        if abbr:
            return (abbr, int(m.group(2)))

    # "State's Nth" (ordinal word form)
    m = re.match(r"^(.+?)(?:'s?)?\s+(\w[\w-]*)(?:\s+congressional\s+district)?$", text, re.IGNORECASE)
    if m:
        abbr = _STATE_ABBR.get(m.group(1).strip())
        dist_word = m.group(2).strip().lower().rstrip("stndrdth")
        if abbr and dist_word in _ORDINAL_MAP:
            return (abbr, _ORDINAL_MAP[dist_word])

    # Bare state name (at-large)
    abbr = _STATE_ABBR.get(text)
    if abbr:
        return (abbr, 0)

    return None


def _fetch_html() -> str:
    req = urllib.request.Request(_WIKI_URL, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _parse_open_seats_from_html(html: str) -> set[tuple[str, int]] | None:
    """Extract open seats from Wikipedia HTML tables near the retirements section."""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception as exc:
        logger.info("HTML table parse failed: %s", exc)
        return None

    seats: set[tuple[str, int]] = set()
    for df in tables:
        # Look for tables with a district/constituency column
        cols_lower = {str(c).lower(): c for c in df.columns}
        dist_col = next(
            (cols_lower[c] for c in cols_lower if "district" in c or "constituency" in c),
            None,
        )
        if dist_col is None:
            # Try first column if it looks like district names
            first_col = df.columns[0]
            sample = df[first_col].dropna().astype(str).head(5).tolist()
            if any(re.search(r"\d+|at.large", s, re.IGNORECASE) for s in sample):
                dist_col = first_col
        if dist_col is None:
            continue

        for val in df[dist_col].dropna().astype(str):
            parsed = _parse_district(val)
            if parsed and parsed[0] in _US_STATES:
                seats.add(parsed)

    return seats if seats else None


def _parse_open_seats_from_text(html: str) -> set[tuple[str, int]]:
    """Fallback: scan raw HTML text for district patterns near retirement keywords."""
    seats: set[tuple[str, int]] = set()

    # Find blocks near "not seeking re-election" / "retirement" sections
    # Look for patterns like "State N" or "State at-large" in list items and headings
    patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d{1,2})\b",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+at[-\s]large",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, html):
            text = m.group(0)
            parsed = _parse_district(text)
            if parsed and parsed[0] in _US_STATES:
                seats.add(parsed)

    return seats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Fetching Wikipedia 2026 House elections article...")
    try:
        html = _fetch_html()
    except Exception as exc:
        logger.info("Wikipedia fetch failed: %s — falling back to constants.py", exc)
        _write(set(OPEN_SEATS_2026))
        return

    parsed: set[tuple[str, int]] | None = _parse_open_seats_from_html(html)
    if not parsed or len(parsed) < 10:
        logger.info(
            "Table parse found %d seats (too few) — trying text scan",
            len(parsed) if parsed else 0,
        )
        parsed = _parse_open_seats_from_text(html)

    if not parsed or len(parsed) < 10:
        logger.info(
            "Could not reliably parse open seats (%d found) — falling back to constants.py",
            len(parsed) if parsed else 0,
        )
        _write(set(OPEN_SEATS_2026))
        return

    _write(parsed)


def _write(seats: set[tuple[str, int]]) -> None:
    out_path = DataFile.OPEN_SEATS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seats)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state_po", "district"])
        writer.writeheader()
        for state_po, district in rows:
            writer.writerow({"state_po": state_po, "district": district})
    logger.info("Wrote %d open seats to %s", len(rows), out_path)

    # Log by state for review
    from collections import defaultdict
    by_state: dict[str, list[int]] = defaultdict(list)
    for state_po, district in rows:
        by_state[state_po].append(district)
    for state in sorted(by_state):
        logger.info("  %s: %s", state, sorted(by_state[state]))


if __name__ == "__main__":
    main()
