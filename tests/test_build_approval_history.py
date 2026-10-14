"""Tests for scripts/build_approval_history.py. All synthetic data, no network calls."""
from __future__ import annotations

import datetime

import pandas as pd
import pytest

from scripts.build_approval_history import (
    MAX_PLAUSIBLE_GAP_DAYS,
    _election_date,
    _normalize_col,
    _parse_html_table,
    select_readings,
)


class TestElectionDate:
    def test_known_dates(self):
        # Verified against the actual historical US House election days.
        assert _election_date(2024) == datetime.date(2024, 11, 5)
        assert _election_date(2016) == datetime.date(2016, 11, 8)
        assert _election_date(2000) == datetime.date(2000, 11, 7)

    def test_always_a_tuesday(self):
        for year in range(1976, 2030, 2):
            assert _election_date(year).weekday() == 1  # Tuesday

    def test_always_after_first_monday_in_november(self):
        for year in range(1976, 2030, 2):
            election = _election_date(year)
            first_of_nov = datetime.date(year, 11, 1)
            first_monday = first_of_nov
            while first_monday.weekday() != 0:
                first_monday += datetime.timedelta(days=1)
            assert election == first_monday + datetime.timedelta(days=1)


def _rows(*entries: tuple[str, str, int, int, int]) -> pd.DataFrame:
    """Build a synthetic all_rows DataFrame from (start, end, approve, disapprove, unsure) tuples."""
    df = pd.DataFrame(entries, columns=["start", "end", "approve", "disapprove", "unsure"])
    df["end_date"] = pd.to_datetime(df["end"])
    return df


class TestSelectReadings:
    def test_picks_latest_reading_before_election(self):
        all_rows = _rows(
            ("10/01/2024", "10/10/2024", 45, 50, 5),
            ("10/15/2024", "10/25/2024", 46, 49, 5),  # closest to the 2024 election
            ("11/10/2024", "11/15/2024", 47, 48, 5),  # after the election - must be excluded
        )
        results = select_readings(all_rows, [2024])
        assert len(results) == 1
        r = results[0]
        assert r["status"] == "ok"
        assert r["row"]["end"] == "10/25/2024"
        assert r["gap_days"] == (datetime.date(2024, 11, 5) - datetime.date(2024, 10, 25)).days

    def test_no_candidate_when_nothing_before_election(self):
        all_rows = _rows(("12/01/2024", "12/10/2024", 45, 50, 5))
        results = select_readings(all_rows, [2024])
        assert results[0]["status"] == "no_candidate"
        assert results[0]["row"] is None

    def test_stale_reading_rejected_even_though_it_is_the_closest(self):
        all_rows = _rows(("01/01/2022", "01/16/2022", 40, 56, 4))
        results = select_readings(all_rows, [2024], max_gap_days=MAX_PLAUSIBLE_GAP_DAYS)
        r = results[0]
        assert r["status"] == "stale"
        assert r["row"] is not None  # the row exists, it's just rejected as too old
        assert r["gap_days"] > MAX_PLAUSIBLE_GAP_DAYS

    def test_reading_within_gap_threshold_is_accepted(self):
        election = _election_date(2024)
        close_date = election - datetime.timedelta(days=MAX_PLAUSIBLE_GAP_DAYS - 1)
        all_rows = _rows((close_date.strftime("%m/%d/%Y"), close_date.strftime("%m/%d/%Y"), 50, 45, 5))
        results = select_readings(all_rows, [2024])
        assert results[0]["status"] == "ok"

    def test_reading_just_past_gap_threshold_is_rejected(self):
        election = _election_date(2024)
        far_date = election - datetime.timedelta(days=MAX_PLAUSIBLE_GAP_DAYS + 1)
        all_rows = _rows((far_date.strftime("%m/%d/%Y"), far_date.strftime("%m/%d/%Y"), 50, 45, 5))
        results = select_readings(all_rows, [2024])
        assert results[0]["status"] == "stale"

    def test_handles_multiple_years_independently(self):
        all_rows = _rows(
            ("10/20/2016", "10/27/2016", 53, 45, 2),
            ("10/29/2018", "11/04/2018", 40, 54, 6),
        )
        results = select_readings(all_rows, [2016, 2018, 2020])
        by_year = {r["year"]: r for r in results}
        assert by_year[2016]["status"] == "ok"
        assert by_year[2016]["row"]["approve"] == 53
        assert by_year[2018]["status"] == "ok"
        assert by_year[2018]["row"]["approve"] == 40
        # 2020's closest candidate is the 2018 row (~735 days out) - a real candidate
        # exists, it's just too old, which is "stale" not "no_candidate".
        assert by_year[2020]["status"] == "stale"

    def test_no_candidate_distinct_from_stale(self):
        """no_candidate means nothing at all exists before the election; stale means
        something exists but is too old. Different years, different reasons."""
        all_rows = _rows(("10/29/2018", "11/04/2018", 40, 54, 6))
        results = select_readings(all_rows, [2018, 2016])
        by_year = {r["year"]: r for r in results}
        assert by_year[2018]["status"] == "ok"
        # Nothing in the data precedes the 2016 election at all.
        assert by_year[2016]["status"] == "no_candidate"
        assert by_year[2016]["row"] is None

    def test_picks_closest_not_just_any_valid_row(self):
        """Regression guard: must pick the row closest to the election, not an
        arbitrary/earlier one that also satisfies end_date < election_date."""
        all_rows = _rows(
            ("01/01/2016", "01/10/2016", 50, 45, 5),   # far from the election
            ("10/20/2016", "10/27/2016", 53, 45, 2),   # close to the election
            ("06/01/2016", "06/10/2016", 48, 47, 5),   # also far, different direction
        )
        results = select_readings(all_rows, [2016])
        assert results[0]["row"]["end"] == "10/27/2016"


class TestNormalizeCol:
    def test_collapses_whitespace_variants(self):
        # These three literally appear across different UCSB presidents' pages for
        # what is semantically the same column - this bug was caught by actually
        # running the script against Ford's and Bush Sr.'s real pages, not by a test
        # written in advance, which is exactly why it's pinned down here now.
        assert _normalize_col("Unsure/NoData") == "unsure/nodata"
        assert _normalize_col("Unsure/ NoData") == "unsure/nodata"
        assert _normalize_col("Unsure / No Data") == "unsure/nodata"

    def test_lowercases(self):
        assert _normalize_col("Start Date") == _normalize_col("START DATE")


class TestParseHtmlTable:
    def test_handles_inconsistent_unsure_column_whitespace(self):
        """Regression test for the Ford/Bush Sr. page parsing failure: their 'Unsure'
        column header has different whitespace than Biden's page, which an exact
        string match against "Unsure/NoData" missed entirely."""
        html = """
        <table>
        <tr><th>Start Date</th><th>End Date</th><th>Approving</th><th>Disapproving</th>
            <th>Unsure / No Data</th></tr>
        <tr><td>10/18/1982</td><td>10/20/1982</td><td>42</td><td>48</td><td>10</td></tr>
        </table>
        """
        df = _parse_html_table(html)
        assert len(df) == 1
        assert df.iloc[0]["unsure"] == 10
    def test_parses_table_with_extra_demographic_columns(self):
        """The real UCSB pages have Democrats/Independents/Republicans-approving
        columns after the 5 we need - selection must be by column name, not position."""
        html = """
        <table>
        <tr><th>Start Date</th><th>End Date</th><th>Approving</th><th>Disapproving</th>
            <th>Unsure/NoData</th><th>Democrats Approving</th><th>Republicans Approving</th></tr>
        <tr><td>10/14/2024</td><td>10/27/2024</td><td>41</td><td>56</td><td>3</td>
            <td>85</td><td>4</td></tr>
        </table>
        """
        df = _parse_html_table(html)
        assert list(df.columns) == ["start", "end", "approve", "disapprove", "unsure", "end_date"]
        assert len(df) == 1
        assert df.iloc[0]["approve"] == 41
        assert df.iloc[0]["end_date"] == pd.Timestamp("2024-10-27")

    def test_picks_the_right_table_when_page_has_multiple(self):
        """The real pages have a second, unrelated table (source citations) below the
        data table - parsing must find the one with the expected columns, not table[0]
        blindly."""
        html = """
        <table><tr><th>Some</th><th>Other</th><th>Citation</th><th>Table</th></tr>
        <tr><td>a</td><td>b</td><td>c</td><td>d</td></tr></table>
        <table>
        <tr><th>Start Date</th><th>End Date</th><th>Approving</th><th>Disapproving</th>
            <th>Unsure/NoData</th></tr>
        <tr><td>10/14/2024</td><td>10/27/2024</td><td>41</td><td>56</td><td>3</td></tr>
        </table>
        """
        df = _parse_html_table(html)
        assert len(df) == 1
        assert df.iloc[0]["approve"] == 41

    def test_raises_when_no_matching_table_found(self):
        html = "<table><tr><th>Nothing</th><th>Relevant</th></tr><tr><td>1</td><td>2</td></tr></table>"
        with pytest.raises(ValueError, match="No table"):
            _parse_html_table(html)
