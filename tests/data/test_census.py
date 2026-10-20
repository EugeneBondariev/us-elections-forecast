"""Unit tests for src/data/census.py. No network access, no files on disk."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.constants import STATE_FIPS_TO_PO, Col
from src.data.census import (
    _parse_district_code,
    build_census_features,
    load_census,
    parse_census_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HEADER = [
    "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
    "B15003_001E", "B19013_001E", "B02001_001E", "B02001_002E",
    "B01002_001E", "state", "congressional district",
]

# CA-12: FIPS "06", district "12"
_ROW_CA_12 = ["42000", "20000", "8000", "5000", "220000", "95000", "800000", "500000", "38", "06", "12"]
# TX-05: FIPS "48", district "05"
_ROW_TX_05 = ["30000", "15000", "5000", "3000", "195000", "72000", "650000", "480000", "37", "48", "05"]
# NY-10: FIPS "36", district "10"
_ROW_NY_10 = ["50000", "25000", "9000", "7000", "240000", "88000", "750000", "550000", "40", "36", "10"]
# AK-00 (at-large): FIPS "02", district "00"
_ROW_AK_00 = ["18000", "8000", "2000", "1000", "160000", "78000", "740000", "620000", "36", "02", "00"]
# PR (territory, FIPS "72" — not in STATE_FIPS_TO_PO → must be dropped)
_ROW_PR_01 = ["10000", "4000", "1000", "500", "120000", "35000", "900000", "400000", "33", "72", "01"]


@pytest.fixture
def raw_response() -> list[list[str]]:
    return [_HEADER, _ROW_CA_12, _ROW_TX_05, _ROW_NY_10, _ROW_AK_00, _ROW_PR_01]


# ---------------------------------------------------------------------------
# _parse_district_code
# ---------------------------------------------------------------------------

def test_parse_district_code_normal() -> None:
    assert _parse_district_code("12") == 12


def test_parse_district_code_at_large() -> None:
    assert _parse_district_code("00") == 0


def test_parse_district_code_invalid() -> None:
    assert _parse_district_code("ZZ") == 0


# ---------------------------------------------------------------------------
# parse_census_response
# ---------------------------------------------------------------------------

def test_parse_census_response_drops_territories(raw_response: list[list[str]]) -> None:
    df = parse_census_response(raw_response)
    assert "PR" not in df[Col.STATE].values
    # Verify only the 4 valid US-state rows remain
    assert len(df) == 4


def test_parse_census_response_at_large_district(raw_response: list[list[str]]) -> None:
    df = parse_census_response(raw_response)
    ak_rows = df[df[Col.STATE] == "AK"]
    assert len(ak_rows) == 1
    assert ak_rows.iloc[0][Col.DISTRICT] == 0


def test_parse_census_response_column_names(raw_response: list[list[str]]) -> None:
    df = parse_census_response(raw_response)
    expected = {Col.STATE, Col.DISTRICT, "bachelor", "master", "professional",
                "doctorate", "educ_total", "median_income", "pop_total",
                "pop_white", "median_age"}
    assert expected.issubset(set(df.columns))
    # Raw API columns must be gone
    assert "congressional district" not in df.columns
    assert "state" not in df.columns


def test_parse_census_response_missing_sentinel(raw_response: list[list[str]]) -> None:
    sentinel_row = [
        "-666666666", "20000", "8000", "5000", "220000",
        "-666666666", "800000", "500000", "38", "06", "99",
    ]
    raw = [_HEADER, sentinel_row]
    df = parse_census_response(raw)
    assert pd.isna(df.iloc[0]["bachelor"])
    assert pd.isna(df.iloc[0]["median_income"])


# ---------------------------------------------------------------------------
# build_census_features
# ---------------------------------------------------------------------------

def test_build_census_features_output_columns(raw_response: list[list[str]]) -> None:
    parsed = parse_census_response(raw_response)
    features = build_census_features(parsed)
    expected_cols = [
        Col.STATE, Col.DISTRICT,
        Col.PCT_BACHELOR_PLUS, Col.PCT_NON_WHITE,
        Col.MEDIAN_INCOME, Col.MEDIAN_AGE,
    ]
    assert list(features.columns) == expected_cols


def test_build_census_features_pct_bachelor_range(raw_response: list[list[str]]) -> None:
    parsed = parse_census_response(raw_response)
    features = build_census_features(parsed)
    col = features[Col.PCT_BACHELOR_PLUS].dropna()
    assert (col >= 0).all() and (col <= 1).all()


def test_build_census_features_pct_nonwhite_range(raw_response: list[list[str]]) -> None:
    parsed = parse_census_response(raw_response)
    features = build_census_features(parsed)
    col = features[Col.PCT_NON_WHITE].dropna()
    assert (col >= 0).all() and (col <= 1).all()


def test_build_census_features_bachelor_computation(raw_response: list[list[str]]) -> None:
    parsed = parse_census_response(raw_response)
    features = build_census_features(parsed)
    # Verify CA-12 manually: (42000+20000+8000+5000) / 220000
    ca_row = features[features[Col.STATE] == "CA"].iloc[0]
    expected = (42000 + 20000 + 8000 + 5000) / 220000
    assert abs(ca_row[Col.PCT_BACHELOR_PLUS] - expected) < 1e-6


def test_build_census_features_missing_income_preserved_as_nan() -> None:
    df = pd.DataFrame({
        Col.STATE: ["TX"],
        Col.DISTRICT: [5],
        "bachelor": [30000.0],
        "master": [15000.0],
        "professional": [5000.0],
        "doctorate": [3000.0],
        "educ_total": [195000.0],
        "median_income": [float("nan")],
        "pop_total": [650000.0],
        "pop_white": [480000.0],
        "median_age": [37.0],
    })
    features = build_census_features(df)
    assert pd.isna(features.iloc[0][Col.MEDIAN_INCOME])


def test_build_census_features_no_extra_rows(raw_response: list[list[str]]) -> None:
    parsed = parse_census_response(raw_response)
    features = build_census_features(parsed)
    # 4 valid US-state rows in; same 4 rows out
    assert len(features) == len(parsed)


# ---------------------------------------------------------------------------
# STATE_FIPS_TO_PO completeness
# ---------------------------------------------------------------------------

def test_state_fips_mapping_complete() -> None:
    assert len(STATE_FIPS_TO_PO) == 50


# ---------------------------------------------------------------------------
# load_census
# ---------------------------------------------------------------------------

def test_load_census_returns_none_if_missing() -> None:
    result = load_census(Path("nonexistent/path.csv"))
    assert result is None
