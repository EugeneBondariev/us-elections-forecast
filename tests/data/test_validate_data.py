"""Unit tests for src.data.validate_data. All synthetic — no data files needed."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.validate_data import (
    _validate_file,
    validate_all,
    validate_cook,
    validate_fundraising,
    validate_required,
)


class TestValidateFile:
    def test_absent_file_skips_and_returns_true(self, tmp_path: Path) -> None:
        result = _validate_file(
            label="test",
            paths=[tmp_path / "missing.csv"],
            skip_reason="not present",
            load=lambda: pd.DataFrame(),
            ok_detail=lambda _: "",
        )
        assert result is True

    def test_one_of_two_paths_absent_skips_and_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "exists.csv").touch()
        result = _validate_file(
            label="test",
            paths=[tmp_path / "exists.csv", tmp_path / "missing.csv"],
            skip_reason="not present",
            load=lambda: pd.DataFrame(),
            ok_detail=lambda _: "",
        )
        assert result is True

    def test_present_file_load_succeeds_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").touch()
        result = _validate_file(
            label="test",
            paths=[tmp_path / "data.csv"],
            skip_reason="",
            load=lambda: pd.DataFrame({"a": [1, 2]}),
            ok_detail=lambda df: f"{len(df)} rows",
        )
        assert result is True

    def test_present_file_load_raises_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").touch()

        def bad_load() -> pd.DataFrame:
            raise ValueError("bad data")

        result = _validate_file(
            label="test",
            paths=[tmp_path / "data.csv"],
            skip_reason="",
            load=bad_load,
            ok_detail=lambda _: "",
        )
        assert result is False

    def test_min_rows_not_met_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").touch()
        result = _validate_file(
            label="test",
            paths=[tmp_path / "data.csv"],
            skip_reason="",
            load=lambda: pd.DataFrame({"a": [1, 2]}),
            ok_detail=lambda _: "",
            min_rows=100,
        )
        assert result is False

    def test_min_rows_met_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").touch()
        result = _validate_file(
            label="test",
            paths=[tmp_path / "data.csv"],
            skip_reason="",
            load=lambda: pd.DataFrame({"a": range(10)}),
            ok_detail=lambda df: f"{len(df)} rows",
            min_rows=5,
        )
        assert result is True


class TestValidateCook:
    def test_absent_returns_true(self, tmp_path: Path) -> None:
        with patch("src.data.validate_data.DataFile") as mock:
            mock.COOK = tmp_path / "missing.csv"
            assert validate_cook() is True

    def test_present_load_fails_returns_false(self, tmp_path: Path) -> None:
        cook = tmp_path / "cook.csv"
        cook.touch()
        with patch("src.data.validate_data.DataFile") as mock:
            mock.COOK = cook
            with patch("src.data.validate_data._load_cook", side_effect=ValueError("bad")):
                assert validate_cook() is False

    def test_present_load_succeeds_returns_true(self, tmp_path: Path) -> None:
        cook = tmp_path / "cook.csv"
        cook.touch()
        with patch("src.data.validate_data.DataFile") as mock:
            mock.COOK = cook
            with patch("src.data.validate_data._load_cook", return_value=pd.DataFrame({"a": [1]})):
                assert validate_cook() is True


class TestValidateFundraising:
    def test_absent_returns_true(self, tmp_path: Path) -> None:
        with patch("src.data.validate_data.DataFile") as mock:
            mock.FUNDRAISING = tmp_path / "missing.csv"
            assert validate_fundraising() is True

    def test_present_load_fails_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "fundraising.csv"
        f.touch()
        with patch("src.data.validate_data.DataFile") as mock:
            mock.FUNDRAISING = f
            with patch("src.data.validate_data._load_fundraising", side_effect=ValueError("bad")):
                assert validate_fundraising() is False


class TestValidateRequired:
    def test_all_pass_returns_true(self) -> None:
        with patch("src.data.validate_data._validate_file", return_value=True):
            assert validate_required() is True

    def test_one_fails_returns_false(self) -> None:
        with patch("src.data.validate_data._validate_file", side_effect=[False, True, True]):
            assert validate_required() is False


class TestValidateAll:
    def test_all_pass_returns_true(self) -> None:
        with (
            patch("src.data.validate_data.validate_required", return_value=True),
            patch("src.data.validate_data.validate_cook", return_value=True),
            patch("src.data.validate_data.validate_fundraising", return_value=True),
        ):
            assert validate_all() is True

    def test_required_fails_returns_false(self) -> None:
        with (
            patch("src.data.validate_data.validate_required", return_value=False),
            patch("src.data.validate_data.validate_cook", return_value=True),
            patch("src.data.validate_data.validate_fundraising", return_value=True),
        ):
            assert validate_all() is False

    def test_optional_fails_returns_false(self) -> None:
        with (
            patch("src.data.validate_data.validate_required", return_value=True),
            patch("src.data.validate_data.validate_cook", return_value=False),
            patch("src.data.validate_data.validate_fundraising", return_value=True),
        ):
            assert validate_all() is False
