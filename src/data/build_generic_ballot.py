"""Fetch generic congressional ballot history and write data/interim/generic_ballot.csv.

Strategy:
  1976–2004: hardcoded Gallup-era values (not machine-readable from a clean single source).
  2006–2024: FiveThirtyEight GitHub CSV (generic_ballot_averages.csv); falls back to
             constants.py if the file is missing or lacks the expected columns.
  2026: Wikipedia 2026 House elections article generic ballot table; falls back to
        GENERIC_BALLOT[2026] from constants.py if not parseable.

Data sources:
    FiveThirtyEight generic ballot averages:
        https://raw.githubusercontent.com/fivethirtyeight/data/master/generic-ballot/generic_ballot_averages.csv
    Wikipedia 2026 House elections:
        https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections

Run:
    python -m src.data.build_generic_ballot
"""
from __future__ import annotations

import csv
import io
import logging
import re
import ssl
import urllib.request

import certifi
import pandas as pd

from src.constants import GENERIC_BALLOT
from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; us-elections-forecast/1.0; educational use)"

_FTE_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "congress-generic-ballot/generic_topline_historical.csv"
)
_WIKI_URL = (
    "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections"
)

# Gallup-era pre-computed values (not machine-readable from a clean source).
_HARDCODED: dict[int, float] = {
    1976: 14, 1978: 9, 1980: -5, 1982: 12, 1984: 9,
    1986: 10, 1988: 4, 1990: 8, 1992: 12, 1994: -7,
    1996: 0, 1998: 1, 2000: 1, 2002: -2, 2004: 1,
}

ELECTION_YEARS_538 = list(range(2006, 2025, 2))


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _fetch_538() -> tuple[dict[int, float], str]:
    """Return ({year: margin}, source_label) for 2006–2024."""
    try:
        raw = _fetch(_FTE_URL)
    except Exception as exc:
        logger.info("FiveThirtyEight fetch failed: %s — falling back to constants.py", exc)
        return {y: GENERIC_BALLOT[y] for y in ELECTION_YEARS_538 if y in GENERIC_BALLOT}, "constants.py"

    DataFile.GENERIC_BALLOT_RAW.write_text(raw, encoding="utf-8")
    logger.info("Saved raw 538 generic ballot to %s", DataFile.GENERIC_BALLOT_RAW)

    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as exc:
        logger.info("FiveThirtyEight CSV parse failed: %s — falling back to constants.py", exc)
        return {y: GENERIC_BALLOT[y] for y in ELECTION_YEARS_538 if y in GENERIC_BALLOT}, "constants.py"

    required = {"modeldate", "dem_estimate", "rep_estimate"}
    if not required.issubset(df.columns):
        logger.info(
            "FiveThirtyEight CSV missing expected columns — falling back. Columns: %s",
            list(df.columns),
        )
        return {y: GENERIC_BALLOT[y] for y in ELECTION_YEARS_538 if y in GENERIC_BALLOT}, "constants.py"

    # Prefer likely-voter subgroup; fall back to all polls if absent.
    if "subgroup" in df.columns:
        lv = df[df["subgroup"] == "Likely voters"]
        df = lv if not lv.empty else df[df["subgroup"] == "All polls"]

    df["_date"] = pd.to_datetime(df["modeldate"], errors="coerce")
    df["_year"] = df["_date"].dt.year
    df["_margin"] = pd.to_numeric(df["dem_estimate"], errors="coerce") - pd.to_numeric(
        df["rep_estimate"], errors="coerce"
    )
    df = df.dropna(subset=["_date", "_margin"])

    result: dict[int, float] = {}
    for year in ELECTION_YEARS_538:
        cutoff = pd.Timestamp(f"{year}-11-08")
        year_df = df[df["_date"] <= cutoff]
        year_df = year_df[year_df["_year"] >= year - 1]
        if year_df.empty:
            logger.info("No FiveThirtyEight data for %s — using constants.py", year)
            result[year] = GENERIC_BALLOT.get(year, 0.0)
        else:
            result[year] = round(float(year_df.sort_values("_date")["_margin"].iloc[-1]), 1)

    return result, "FiveThirtyEight"


def _fetch_2026_wiki() -> float | None:
    """Extract most recent generic ballot average from the 2026 Wikipedia elections article."""
    try:
        html = _fetch(_WIKI_URL)
    except Exception as exc:
        logger.info("Wikipedia fetch failed: %s", exc)
        return None

    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception as exc:
        logger.info("Wikipedia table parse failed: %s", exc)
        return None

    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        # Look for a table that has both republican/democrat columns and a margin column
        has_dem = any("democrat" in c or "dem" in c for c in cols_lower)
        has_rep = any("republican" in c or "rep" in c for c in cols_lower)
        if not (has_dem and has_rep):
            continue

        # Find margin column or compute from dem/rep columns
        margin_col = next(
            (df.columns[i] for i, c in enumerate(cols_lower) if "margin" in c), None
        )
        if margin_col is not None:
            # Parse margin strings like "Democrats +8.2%" or "+8.2"
            margin_vals = df[margin_col].astype(str)
            for val in margin_vals:
                # Look for "Democrats +N" or "Republicans +N" patterns
                m = re.search(r"(Democrat|Republican)s?\s*[+\-]?\s*(\d+\.?\d*)", val, re.IGNORECASE)
                if m:
                    party = m.group(1).lower()
                    num = float(m.group(2))
                    return num if party.startswith("dem") else -num
            # Try plain numeric margin
            numeric = pd.to_numeric(margin_vals.str.extract(r"([+-]?\d+\.?\d*)")[0], errors="coerce")
            valid = numeric.dropna()
            if not valid.empty:
                return round(float(valid.iloc[-1]), 1)

        # Compute from dem% - rep% columns
        dem_col = next(
            (df.columns[i] for i, c in enumerate(cols_lower) if "democrat" in c or c == "dem"), None
        )
        rep_col = next(
            (df.columns[i] for i, c in enumerate(cols_lower) if "republican" in c or c == "rep"),
            None,
        )
        if dem_col and rep_col:
            dem_vals = pd.to_numeric(
                df[dem_col].astype(str).str.replace("%", "").str.strip(), errors="coerce"
            )
            rep_vals = pd.to_numeric(
                df[rep_col].astype(str).str.replace("%", "").str.strip(), errors="coerce"
            )
            margin = dem_vals - rep_vals
            valid = margin.dropna()
            if not valid.empty:
                return round(float(valid.iloc[-1]), 1)

    logger.info("Could not extract 2026 generic ballot from Wikipedia tables")
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    margins: dict[int, float] = dict(_HARDCODED)

    logger.info("Fetching 2006–2024 generic ballot from FiveThirtyEight...")
    fte_margins, fte_source = _fetch_538()
    for year, margin in fte_margins.items():
        margins[year] = margin
        logger.info("  %s: %.1f (%s)", year, margin, fte_source)

    logger.info("Fetching 2026 generic ballot from Wikipedia...")
    wiki_margin = _fetch_2026_wiki()
    if wiki_margin is not None:
        margins[2026] = wiki_margin
        logger.info("  2026: %.1f (Wikipedia)", wiki_margin)
    else:
        margins[2026] = GENERIC_BALLOT.get(2026, 0.0)
        logger.info("  2026: %.1f (constants.py fallback)", margins[2026])

    all_years = sorted(margins)
    out_path = DataFile.GENERIC_BALLOT_CSV
    with ensure(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "margin"])
        writer.writeheader()
        for year in all_years:
            writer.writerow({"year": year, "margin": margins[year]})

    logger.info("Wrote %d rows to %s", len(all_years), out_path)


if __name__ == "__main__":
    main()
