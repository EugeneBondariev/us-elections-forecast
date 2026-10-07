import numpy as np
import pandas as pd
import pytest

from src.features import (
    GENERIC_BALLOT,
    PRES_PARTY,
    REDISTRICTING_YEARS,
    build_dems_reps,
    load_approval,
)

ELECTION_YEARS = set(range(1976, 2025, 2))


# --- constants ---

def test_generic_ballot_covers_all_election_years():
    assert ELECTION_YEARS.issubset(GENERIC_BALLOT.keys())


def test_generic_ballot_2026_dem_favorable():
    assert GENERIC_BALLOT[2026] > 0


def test_pres_party_covers_all_election_years():
    assert ELECTION_YEARS.issubset(PRES_PARTY.keys())


def test_redistricting_years():
    assert REDISTRICTING_YEARS == {1982, 1992, 2002, 2012, 2022}


# --- dem_lean: no data leakage ---

def _make_party_votes(shares: dict, state_po: str = "CA", district: int = 1) -> pd.DataFrame:
    rows = []
    for year, dem_share in shares.items():
        dem_v = round(dem_share * 100)
        rep_v = 100 - dem_v
        rows += [
            {"year": year, "state_po": state_po, "district": district,
             "party": "DEMOCRAT",   "candidatevotes": dem_v, "totalvotes": 100},
            {"year": year, "state_po": state_po, "district": district,
             "party": "REPUBLICAN", "candidatevotes": rep_v, "totalvotes": 100},
        ]
    return pd.DataFrame(rows)


def test_dem_lean_uses_only_prior_elections():
    # Increasing share 0.4→0.8; dem_lean for 1984 must equal mean(0.4,0.5,0.6,0.7)=0.55,
    # not 0.6 which would happen if the current year (0.8) leaked in.
    shares = {1976: 0.4, 1978: 0.5, 1980: 0.6, 1982: 0.7, 1984: 0.8}
    pv = _make_party_votes(shares)
    dr = build_dems_reps(pv)
    row = dr[(dr["state_po"] == "CA") & (dr["year"] == 1984)]
    assert pytest.approx(row["dem_lean"].values[0], abs=1e-6) == 0.55


def test_dem_lean_nan_before_window_filled():
    # With window=4, the first 4 elections (1976-1982) must all have NaN dem_lean.
    shares = {1976: 0.5, 1978: 0.5, 1980: 0.5, 1982: 0.5, 1984: 0.5}
    pv = _make_party_votes(shares)
    dr = build_dems_reps(pv)
    early = dr[(dr["state_po"] == "CA") & (dr["year"] <= 1982)]
    assert early["dem_lean"].isna().all()


def test_dem_lean_trend_slope():
    # Linearly increasing share → trend slope should be 0.1 per election
    shares = {1976: 0.4, 1978: 0.5, 1980: 0.6, 1982: 0.7, 1984: 0.8}
    pv = _make_party_votes(shares)
    dr = build_dems_reps(pv)
    row = dr[(dr["state_po"] == "CA") & (dr["year"] == 1984)]
    assert pytest.approx(row["dem_lean_trend"].values[0], abs=1e-6) == 0.1


def test_dem_lean_trend_flat_is_zero():
    shares = {1976: 0.6, 1978: 0.6, 1980: 0.6, 1982: 0.6, 1984: 0.6}
    pv = _make_party_votes(shares)
    dr = build_dems_reps(pv)
    row = dr[(dr["state_po"] == "CA") & (dr["year"] == 1984)]
    assert pytest.approx(row["dem_lean_trend"].values[0], abs=1e-6) == 0.0


# --- approval sign flip ---

def _write_approval_file(tmp_path, rows: list[tuple]) -> str:
    lines = "\n".join(f"{start}\t{end}\t{approve}\t{disapprove}\t{unsure}"
                      for start, end, approve, disapprove, unsure in rows)
    p = tmp_path / "approval.xls"
    p.write_text(lines)
    return str(p)


def test_approval_dem_president_positive(tmp_path):
    # Carter 1978 (DEMOCRAT): approval should be positive
    path = _write_approval_file(tmp_path, [("1/1/1978", "12/31/1978", 55, 37, 8)])
    result = load_approval(path)
    assert result[result["year"] == 1978]["pres_approval"].values[0] > 0


def test_approval_rep_president_negative(tmp_path):
    # Reagan 1982 (REPUBLICAN): approval should be negative
    path = _write_approval_file(tmp_path, [("1/1/1982", "12/31/1982", 42, 50, 8)])
    result = load_approval(path)
    assert result[result["year"] == 1982]["pres_approval"].values[0] < 0


def test_approval_disapproval_signs_are_opposite(tmp_path):
    # pres_approval and pres_disapproval should have opposite signs for any president
    path = _write_approval_file(tmp_path, [("1/1/1982", "12/31/1982", 42, 50, 8)])
    result = load_approval(path)
    row = result[result["year"] == 1982].iloc[0]
    assert (row["pres_approval"] > 0) != (row["pres_disapproval"] > 0)


# --- integration: feature matrix shape and nulls ---

def test_feature_matrix_no_nan(real_data):
    X = real_data["X"]
    assert X.isnull().sum().sum() == 0


def test_feature_matrix_expected_columns(real_data):
    expected = {
        "dem_lean", "dem_lean_trend", "pres_approval", "pres_disapproval",
        "generic_ballot", "is_incumbent_on_ballot", "redistricting",
        "nation_incumbent_party_REPUBLICAN", "district_incumbent_party_REPUBLICAN",
    }
    assert set(real_data["X"].columns) == expected


def test_feature_matrix_binary_target(real_data):
    assert set(real_data["y"].unique()).issubset({0, 1})


def test_redistricting_flag_on_correct_years(real_data):
    winners = real_data["winners"]
    for year in REDISTRICTING_YEARS:
        rows = winners[winners["year"] == year]
        if len(rows):
            assert (rows["redistricting"] == 1).all()
    non_redist = winners[~winners["year"].isin(REDISTRICTING_YEARS)]
    assert (non_redist["redistricting"] == 0).all()
