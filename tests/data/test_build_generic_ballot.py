"""Unit tests for src.data.build_generic_ballot. No real HTTP calls — all mocked."""

from __future__ import annotations

import csv
import csv as _csv
import io as _io
from pathlib import Path
from unittest.mock import patch

import pytest

from src.constants import GENERIC_BALLOT
from src.data.build_generic_ballot import _fetch_538, _fetch_2026_wiki, main


def _fte_csv(rows: list[dict]) -> str:
    buf = _io.StringIO()
    writer = _csv.DictWriter(
        buf, fieldnames=["subgroup", "modeldate", "dem_estimate", "rep_estimate"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _row(subgroup: str, date: str, dem: float, rep: float) -> dict:
    return {
        "subgroup": subgroup,
        "modeldate": date,
        "dem_estimate": str(dem),
        "rep_estimate": str(rep),
    }


class TestFetch538:
    def test_returns_fte_label_on_success(self):
        with patch(
            "src.data.build_generic_ballot._fetch",
            return_value=_fte_csv([_row("All polls", "11/1/2006", 52.0, 45.0)]),
        ):
            _, label = _fetch_538()
        assert label == "FiveThirtyEight"

    def test_returns_margin_for_year(self):
        with patch(
            "src.data.build_generic_ballot._fetch",
            return_value=_fte_csv([_row("All polls", "11/1/2006", 52.0, 45.0)]),
        ):
            result, _ = _fetch_538()
        assert result[2006] == pytest.approx(7.0, abs=0.1)

    def test_falls_back_on_fetch_error(self):
        with patch("src.data.build_generic_ballot._fetch", side_effect=Exception("404")):
            result, label = _fetch_538()
        assert label == "constants.py"
        assert all(result[y] == GENERIC_BALLOT[y] for y in result)

    def test_falls_back_on_missing_columns(self):
        with patch("src.data.build_generic_ballot._fetch", return_value="wrong,columns\n1,2"):
            _, label = _fetch_538()
        assert label == "constants.py"

    def test_prefers_likely_voters_subgroup(self):
        rows = [
            _row("All polls", "11/1/2006", 50.0, 45.0),
            _row("Likely voters", "11/1/2006", 52.0, 45.0),
        ]
        with patch("src.data.build_generic_ballot._fetch", return_value=_fte_csv(rows)):
            result, _ = _fetch_538()
        assert result[2006] == pytest.approx(7.0, abs=0.1)

    def test_falls_back_to_all_polls_when_no_lv(self):
        rows = [_row("All polls", "11/1/2006", 53.0, 45.0)]
        with patch("src.data.build_generic_ballot._fetch", return_value=_fte_csv(rows)):
            result, _ = _fetch_538()
        assert result[2006] == pytest.approx(8.0, abs=0.1)

    def test_takes_last_pre_election_day_observation(self):
        rows = [
            _row("All polls", "10/1/2006", 50.0, 45.0),
            _row("All polls", "11/1/2006", 52.0, 45.0),
            _row("All polls", "11/10/2006", 55.0, 45.0),  # after Nov 8 cutoff
        ]
        with patch("src.data.build_generic_ballot._fetch", return_value=_fte_csv(rows)):
            result, _ = _fetch_538()
        assert result[2006] == pytest.approx(7.0, abs=0.1)

    def test_falls_back_per_year_when_no_data(self):
        # CSV has data for 2006 only — 2008+ should fall back to constants
        rows = [_row("All polls", "11/1/2006", 52.0, 45.0)]
        with patch("src.data.build_generic_ballot._fetch", return_value=_fte_csv(rows)):
            result, _ = _fetch_538()
        assert result[2008] == GENERIC_BALLOT[2008]


class TestFetch2026Wiki:
    def test_returns_none_on_fetch_error(self):
        with patch("src.data.build_generic_ballot._fetch", side_effect=Exception("SSL")):
            assert _fetch_2026_wiki() is None

    def test_parses_democrats_plus_margin(self):
        html = (
            "<table><tr><th>Democrat</th><th>Republican</th><th>Margin</th></tr>"
            "<tr><td>48</td><td>41</td><td>Democrats +7.9</td></tr></table>"
        )
        with patch("src.data.build_generic_ballot._fetch", return_value=html):
            result = _fetch_2026_wiki()
        assert result == pytest.approx(7.9, abs=0.1)

    def test_parses_republicans_plus_margin(self):
        html = (
            "<table><tr><th>Democrat</th><th>Republican</th><th>Margin</th></tr>"
            "<tr><td>45</td><td>48</td><td>Republicans +3.0</td></tr></table>"
        )
        with patch("src.data.build_generic_ballot._fetch", return_value=html):
            result = _fetch_2026_wiki()
        assert result == pytest.approx(-3.0, abs=0.1)

    def test_computes_from_dem_rep_columns(self):
        html = (
            "<table><tr><th>Democrat</th><th>Republican</th></tr>"
            "<tr><td>50.5%</td><td>43.2%</td></tr></table>"
        )
        with patch("src.data.build_generic_ballot._fetch", return_value=html):
            result = _fetch_2026_wiki()
        assert result == pytest.approx(7.3, abs=0.1)

    def test_returns_none_when_no_matching_table(self):
        html = (
            "<table><tr><th>Unrelated</th><th>Data</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        with patch("src.data.build_generic_ballot._fetch", return_value=html):
            assert _fetch_2026_wiki() is None


class TestMain:
    def _run(self, tmp_path: Path, fte: tuple, wiki: float | None) -> dict[int, float]:
        with (
            patch("src.data.build_generic_ballot._fetch_538", return_value=fte),
            patch("src.data.build_generic_ballot._fetch_2026_wiki", return_value=wiki),
            patch("src.data.build_generic_ballot.DataFile") as mock_df,
        ):
            mock_df.GENERIC_BALLOT_CSV = tmp_path / "generic_ballot.csv"
            main()
        return {
            int(r["year"]): float(r["margin"])
            for r in csv.DictReader((tmp_path / "generic_ballot.csv").open())
        }

    def test_writes_hardcoded_years(self, tmp_path: Path):
        rows = self._run(tmp_path, ({}, "FiveThirtyEight"), None)
        assert 1976 in rows
        assert 2004 in rows

    def test_writes_fte_year(self, tmp_path: Path):
        rows = self._run(tmp_path, ({2006: 7.0}, "FiveThirtyEight"), None)
        assert rows[2006] == pytest.approx(7.0, abs=0.01)

    def test_uses_wiki_margin_for_2026(self, tmp_path: Path):
        rows = self._run(tmp_path, ({}, "FiveThirtyEight"), 9.5)
        assert rows[2026] == pytest.approx(9.5, abs=0.01)

    def test_falls_back_to_constants_when_wiki_fails(self, tmp_path: Path):
        rows = self._run(tmp_path, ({}, "FiveThirtyEight"), None)
        assert rows[2026] == pytest.approx(GENERIC_BALLOT.get(2026, 0.0), abs=0.01)
