"""Reconstruct data/interim/Approval-Ratings-for-POTUS-raw.xls from presidency.ucsb.edu's
dedicated per-president approval pages, picking the last poll reading before each
House election.

This lives in interim/, not raw/, despite the filename: it's a selection built
from 9 pooled external sources (closest-before-election, staleness-filtered),
not a passthrough of anything downloaded untouched - the same shape of
transformation as cook.html -> cook.csv.

LOCAL USE ONLY. This data is Gallup-sourced and explicitly copyrighted by UCSB (see the
licensing note in src/data/fetch_data.py's module docstring) - the output of this script
must never be committed, mirrored, uploaded as a release asset, or otherwise redistributed.
It's for this project's own local analysis, same as if you'd copied the cells by hand.

UCSB publishes one page per president at a predictable URL
(presidency.ucsb.edu/statistics/data/<slug>-public-approval), each with the full-term
approval history as an Excel-exported HTML table. PRESIDENT_SLUGS below covers every
president from Ford (1976 is the earliest House election year this project needs)
through Biden.

An earlier version of this script instead scraped a public Google Sheet with one tab
per president, found by enumerating every "gid=" reference in its /pubhtml view. That
sheet's Biden tab turned out to be stale (stopped in January 2022, ~3 years before the
2024 election), while every other president's tab matched these dedicated pages
exactly (identical row counts and date ranges, verified directly) except being less
current for whoever was the most recent president. Since the dedicated pages are both
more current and more authoritative (UCSB's own site, not a third party's copy of it),
this version uses only them - one consistent source type instead of two.

For each House election year (every even year, 1976-2024 - the range the source house
results file actually covers), this takes every row across all pages and picks the one
with the latest End Date strictly before that year's election day (the Tuesday after the
first Monday in November) - i.e. the last available reading going into the election, with
no closer reading available. Years with no sufficiently-recent qualifying row are reported
as missing, not silently skipped or guessed at.

Run from project root:
    python -m src.data.build_approval_history
"""
import datetime
import re
import sys
from io import StringIO

import pandas as pd
import requests

from src.paths import DataFile

UCSB_BASE = "https://www.presidency.ucsb.edu/statistics/data"

# One dedicated-page slug per president, Ford through Biden (the range of House
# election years, 1976-2024, this project needs). Verified directly: every one of
# these resolves and contains the expected approval-rating table.
PRESIDENT_SLUGS = [
    "gerald-r-ford-public-approval",
    "jimmy-carter-public-approval",
    "ronald-reagan-public-approval",
    "george-bush-public-approval",
    "william-j-clinton-public-approval",
    "george-w-bush-public-approval",
    "barack-obama-public-approval",
    "donald-j-trump-public-approval",
    "joseph-r-biden-public-approval",
]

REQUIRED_YEARS = list(range(1976, 2025, 2))

# A "last poll before the election" that's actually ~300+ days old isn't a legitimate
# pre-election reading, even though it technically satisfies "End Date < election date
# with nothing closer available" - it means the source has a real coverage gap, not that
# this is a good data point. Every genuine match in this sheet is within ~20 days (1976's
# 141-day gap is the one exception, plausibly just lower 1970s Gallup polling frequency,
# not a hole in the source) - 180 days cleanly separates real matches from stale ones.
MAX_PLAUSIBLE_GAP_DAYS = 180


def _election_date(year: int) -> datetime.date:
    """US House election day: the Tuesday after the first Monday in November."""
    d = datetime.date(year, 11, 1)
    while d.weekday() != 0:  # Monday
        d += datetime.timedelta(days=1)
    return d + datetime.timedelta(days=1)


def _fetch_html_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return _parse_html_table(resp.text)


def _normalize_col(col: object) -> str:
    """Collapse whitespace and lowercase, so 'Unsure/NoData', 'Unsure/ NoData', and
    'Unsure / No Data' (all three appear across different UCSB president pages) match."""
    return re.sub(r"\s+", "", str(col)).lower()


# Normalized column name -> the 5 columns we need, in output order.
_REQUIRED_COLS = {
    "startdate": "start",
    "enddate": "end",
    "approving": "approve",
    "disapproving": "disapprove",
    "unsure/nodata": "unsure",
}


def _parse_html_table(html_text: str) -> pd.DataFrame:
    """Parse a UCSB per-president approval page's main data table into a
    start/end/approve/disapprove/unsure/end_date DataFrame.

    These pages' approval table is an Excel export with extra demographic-breakdown
    columns after the 5 we need (Democrats/Independents/Republicans approving, etc.) -
    select by name instead of position so added/reordered columns don't silently
    shift what we read. Column names also aren't consistent across presidents' pages
    (whitespace varies in the "Unsure/NoData" column especially), hence _normalize_col.
    """
    tables = pd.read_html(StringIO(html_text))
    for table in tables:
        norm_to_actual = {_normalize_col(c): c for c in table.columns}
        if set(_REQUIRED_COLS) <= set(norm_to_actual):
            actual_cols = [norm_to_actual[k] for k in _REQUIRED_COLS]
            df = table[actual_cols].copy()
            df.columns = list(_REQUIRED_COLS.values())
            df["end_date"] = pd.to_datetime(df["end"], errors="coerce")
            return df.dropna(subset=["end_date"])
    raise ValueError("No table with the expected approval-rating columns found on this page")


def select_readings(
    all_rows: pd.DataFrame,
    required_years: list[int],
    max_gap_days: int = MAX_PLAUSIBLE_GAP_DAYS,
) -> list[dict]:
    """For each required year, pick the row with the latest end_date strictly before
    that year's election day, rejecting anything more than max_gap_days stale.

    Returns one dict per required year: {year, election_date, row, gap_days, status},
    where status is "ok" (row is usable), "stale" (row exists but too old), or
    "no_candidate" (nothing before the election at all). row/gap_days are None unless
    status == "ok".
    """
    results = []
    for year in required_years:
        election = _election_date(year)
        candidates = all_rows[all_rows["end_date"] < pd.Timestamp(election)]
        if candidates.empty:
            results.append({
                "year": year, "election_date": election,
                "row": None, "gap_days": None, "status": "no_candidate",
            })
            continue
        best = candidates.loc[candidates["end_date"].idxmax()]  # type: ignore[call-overload]
        gap_days = (pd.Timestamp(election) - best["end_date"]).days
        if gap_days > max_gap_days:
            results.append({
                "year": year, "election_date": election,
                "row": best, "gap_days": gap_days, "status": "stale",
            })
            continue
        results.append({
            "year": year, "election_date": election,
            "row": best, "gap_days": gap_days, "status": "ok",
        })
    return results


def main() -> None:
    print(f"Fetching {len(PRESIDENT_SLUGS)} per-president pages from {UCSB_BASE}...")
    frames = []
    for slug in PRESIDENT_SLUGS:
        url = f"{UCSB_BASE}/{slug}"
        try:
            df = _fetch_html_table(url)
        except Exception as e:
            print(f"  {slug}: FAILED ({e})")
            continue
        if df.empty:
            print(f"  {slug}: 0 usable rows")
            continue
        frames.append(df)
        print(f"  {slug}: {len(df)} rows, {df['end_date'].min().date()} to {df['end_date'].max().date()}")

    if not frames:
        print("No data fetched from any page - aborting.")
        sys.exit(1)

    all_rows = pd.concat(frames, ignore_index=True)
    print(f"Total rows across all pages: {len(all_rows)}")

    print("\nSelecting last poll before each election:")
    results = select_readings(all_rows, REQUIRED_YEARS)
    selected, missing = [], []
    for r in results:
        if r["status"] == "no_candidate":
            missing.append(r["year"])
            print(f"  {r['year']} (election {r['election_date']}): NO qualifying reading found")
        elif r["status"] == "stale":
            missing.append(r["year"])
            row = r["row"]
            print(
                f"  {r['year']} (election {r['election_date']}): closest available reading "
                f"ends {row['end_date'].date()} ({r['gap_days']}d before) - too stale to use, "
                f"treating as missing"
            )
        else:
            row = r["row"]
            selected.append(row)
            print(
                f"  {r['year']} (election {r['election_date']}): poll ending "
                f"{row['end_date'].date()} ({r['gap_days']}d before) - approve={row['approve']} "
                f"disapprove={row['disapprove']} unsure={row['unsure']}"
            )

    result = pd.DataFrame(selected)[["start", "end", "approve", "disapprove", "unsure"]]

    DataFile.APPROVAL.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DataFile.APPROVAL, sep="\t", header=False, index=False)
    print(f"\nWrote {len(result)} row(s) to {DataFile.APPROVAL}")

    if missing:
        print(f"\nWARNING: no pre-election reading found for year(s): {missing}")
        print(
            "These are missing from the output. validate_approval_coverage() will raise a "
            "clear error naming them if you try to run the pipeline as-is - that's not a bug,"
            " it's the file genuinely being incomplete. You'll need a different source "
            "(e.g. a more current approval-tracking site) to fill in exactly those years,"
            " in the same 5-column tab-separated format, before the pipeline will run."
        )

    print("\nRemember: data/interim/Approval-Ratings-for-POTUS-raw.xls is for local use only - "
          "never commit it, mirror it, or upload it anywhere public.")
    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()
