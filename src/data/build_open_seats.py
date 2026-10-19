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
import logging
import re
import urllib.request

from src.constants import OPEN_SEATS_2026
from src.data.parse_cook import _STATE_ABBR
from src.paths import DataFile, ensure

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


_TITLE_RE = re.compile(
    r"^(.+?)'s (?:(\d+)(?:st|nd|rd|th)|at-large) congressional district$"
)


def _parse_title_attr(title: str) -> tuple[str, int] | None:
    """Parse an anchor's title attribute, e.g.:

    "California's 11th congressional district"       -> ('CA', 11)
    "South Dakota's at-large congressional district"  -> ('SD', 0)
    """
    m = _TITLE_RE.match(title)
    if not m:
        return None
    abbr = _STATE_ABBR.get(m.group(1).strip())
    if not abbr:
        return None
    district = int(m.group(2)) if m.group(2) else 0
    return (abbr, district)


def _extract_section_list_html(html: str, heading_id: str) -> str | None:
    """Return the HTML of the <ol>...</ol> list that follows the heading with this id.

    The retirements lists render as a plain <ol><li>...</li></ol> block, but each <li>
    also embeds a sibling data-mw attribute carrying the raw wikitext source for the
    same entry (for VisualEditor) - that blob sits *before* the real <ol> and would
    itself match any naive "find <ol>, grab everything to the next </ol>" scan if we
    started searching too early, so this anchors on id="{heading_id}" (the rendered
    heading itself, not its table-of-contents entry, which uses id="toc-{heading_id}"
    instead) and only starts scanning for <ol> after that point.
    """
    heading_idx = html.find(f'id="{heading_id}"')
    if heading_idx == -1:
        return None
    ol_start = html.find("<ol", heading_idx)
    if ol_start == -1:
        return None
    ol_end = html.find("</ol>", ol_start)
    if ol_end == -1:
        return None
    return html[ol_start : ol_end + len("</ol>")]


def _parse_open_seats_from_html(html: str) -> set[tuple[str, int]] | None:
    """Extract open seats from the Wikipedia "Retirements" section's Democratic/Republican
    subsections - each a plain numbered list of "<State>'s Nth congressional district:
    <Member> is retiring." entries. Pulls the district identity from each entry's anchor
    `title` attribute (e.g. "California's 11th congressional district") rather than the
    visible text, which is more fragile (names/dates/footnotes mixed in with the district
    reference, interior entity characters, etc.)
    """
    seats: set[tuple[str, int]] = set()
    for heading_id in ("Democratic", "Republican"):
        list_html = _extract_section_list_html(html, heading_id)
        if list_html is None:
            continue
        for title in re.findall(r'title="([^"]*congressional district)"', list_html):
            parsed = _parse_title_attr(title)
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

    # Sanity bounds on the number of open seats found: too few suggests a parse that
    # found nothing useful; too many suggests it matched something much broader than
    # the retirements list (e.g. a full districts-results table - see git history for
    # the bug this guards against, which silently "succeeded" with all 435 districts).
    # ~60 is the real ballpark; 100 leaves headroom without being wide enough to miss
    # a similar false-positive match in a future cycle.
    parsed: set[tuple[str, int]] | None = _parse_open_seats_from_html(html)
    if not parsed or not (10 <= len(parsed) <= 100):
        logger.info(
            "Section-list parse found %d seats (outside expected range) — trying text scan",
            len(parsed) if parsed else 0,
        )
        parsed = _parse_open_seats_from_text(html)

    if not parsed or not (10 <= len(parsed) <= 100):
        logger.info(
            "Could not reliably parse open seats (%d found) — falling back to constants.py",
            len(parsed) if parsed else 0,
        )
        _write(set(OPEN_SEATS_2026))
        return

    _write(parsed)


def _write(seats: set[tuple[str, int]]) -> None:
    out_path = DataFile.OPEN_SEATS_CSV
    rows = sorted(seats)
    with ensure(out_path).open("w", newline="", encoding="utf-8") as f:
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
