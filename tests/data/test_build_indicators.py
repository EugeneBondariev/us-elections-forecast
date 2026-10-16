"""Unit tests for src.data.build_indicators. No real FRED API calls — all mocked."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from src.constants import GDP_2026_ESTIMATE
from src.data.build_indicators import fetch_indicator, main


def _obs(date: str, value: str) -> dict:
    return {"date": date, "value": value}


_ALL_YEARS = list(range(1976, 2028, 2))  # 26 election years


class TestFetchIndicatorDirect:
    def test_returns_preferred_month(self):
        obs = [_obs("2020-10-01", "5.5"), _obs("2020-09-01", "5.3")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="UNRATE", api_key="x", months=("10", "09", "08"), label="test"
            )
        assert result[2020] == 5.5

    def test_falls_back_to_secondary_month(self):
        obs = [_obs("2020-09-01", "5.3")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="UNRATE", api_key="x", months=("10", "09", "08"), label="test"
            )
        assert result[2020] == 5.3

    def test_skips_year_with_no_matching_month(self):
        obs = [_obs("2020-11-01", "5.5")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="UNRATE", api_key="x", months=("10", "09", "08"), label="test"
            )
        assert 2020 not in result

    def test_ignores_dot_placeholder_values(self):
        obs = [_obs("2020-10-01", "."), _obs("2020-09-01", "5.3")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="UNRATE", api_key="x", months=("10", "09", "08"), label="test"
            )
        assert result[2020] == 5.3


class TestFetchIndicatorYoY:
    def test_computes_yoy_growth_from_levels(self):
        obs = [_obs("2019-07-01", "20000.0"), _obs("2020-07-01", "21000.0")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="GDPC1", api_key="x", months=("07", "04", "01"), label="test", yoy=True
            )
        assert result[2020] == pytest.approx(5.0, abs=0.1)

    def test_falls_back_to_secondary_quarter(self):
        obs = [_obs("2019-04-01", "20000.0"), _obs("2020-04-01", "21000.0")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="GDPC1", api_key="x", months=("07", "04", "01"), label="test", yoy=True
            )
        assert result[2020] == pytest.approx(5.0, abs=0.1)

    def test_skips_year_when_prior_missing(self):
        obs = [_obs("2020-07-01", "21000.0")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="GDPC1", api_key="x", months=("07", "04", "01"), label="test", yoy=True
            )
        assert 2020 not in result

    def test_negative_growth(self):
        obs = [_obs("2019-07-01", "21000.0"), _obs("2020-07-01", "20000.0")]
        with patch("src.data.build_indicators._fred_observations", return_value=obs):
            result = fetch_indicator(
                series_id="GDPC1", api_key="x", months=("07", "04", "01"), label="test", yoy=True
            )
        assert result[2020] < 0


class TestMain:
    def _run_main(
        self,
        tmp_path: Path,
        gdp_data: dict,
        unemp_data: dict,
    ) -> list[dict]:
        with (
            patch(
                "src.data.build_indicators.fetch_indicator",
                side_effect=[gdp_data, unemp_data],
            ),
            patch("src.data.build_indicators.DataFile") as mock_df,
            patch.dict("os.environ", {"FRED_API_KEY": "test"}),
        ):
            mock_df.INDICATORS = tmp_path / "indicators.csv"
            main()
        return list(csv.DictReader((tmp_path / "indicators.csv").open()))

    def test_writes_csv_for_all_years_with_data(self, tmp_path: Path):
        gdp = {y: 2.0 for y in _ALL_YEARS}
        unemp = {y: 5.0 for y in _ALL_YEARS}
        rows = self._run_main(tmp_path, gdp, unemp)
        assert len(rows) == len(_ALL_YEARS)

    def test_uses_gdp_estimate_when_2026_missing_from_fred(self, tmp_path: Path):
        gdp = {y: 2.0 for y in _ALL_YEARS if y != 2026}
        unemp = {y: 5.0 for y in _ALL_YEARS}
        rows = self._run_main(tmp_path, gdp, unemp)
        row_2026 = next(r for r in rows if int(r["year"]) == 2026)
        assert float(row_2026["econ_growth"]) == GDP_2026_ESTIMATE

    def test_skips_year_when_unemployment_missing(self, tmp_path: Path):
        gdp = {y: 2.0 for y in _ALL_YEARS}
        unemp = {y: 5.0 for y in _ALL_YEARS if y != 2020}
        rows = self._run_main(tmp_path, gdp, unemp)
        assert not any(int(r["year"]) == 2020 for r in rows)

    def test_no_api_key_returns_early_without_writing(self, tmp_path: Path):
        with (
            patch("src.data.build_indicators.fetch_indicator") as mock_fetch,
            patch("src.data.build_indicators.DataFile") as mock_df,
            patch.dict("os.environ", {"FRED_API_KEY": ""}),
        ):
            mock_df.INDICATORS = tmp_path / "indicators.csv"
            main()
        mock_fetch.assert_not_called()
        assert not (tmp_path / "indicators.csv").exists()
