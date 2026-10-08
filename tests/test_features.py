import pandas as pd
import pytest

from src.features import (
    GENERIC_BALLOT,
    PRES_PARTY,
    REDISTRICTING_YEARS,
    build_dems_reps,
    build_pred_2026,
    load_approval,
    load_house_data,
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
    dr = build_dems_reps(_make_party_votes(shares))
    row = dr[(dr["state_po"] == "CA") & (dr["year"] == 1984)]
    assert pytest.approx(row["dem_lean"].values[0], abs=1e-6) == 0.55


def test_dem_lean_trend_slope():
    # Linearly increasing share → trend slope should be 0.1 per election
    shares = {1976: 0.4, 1978: 0.5, 1980: 0.6, 1982: 0.7, 1984: 0.8}
    dr = build_dems_reps(_make_party_votes(shares))
    row = dr[(dr["state_po"] == "CA") & (dr["year"] == 1984)]
    assert pytest.approx(row["dem_lean_trend"].values[0], abs=1e-6) == 0.1


def test_dem_lean_trend_flat_is_zero():
    shares = {1976: 0.6, 1978: 0.6, 1980: 0.6, 1982: 0.6, 1984: 0.6}
    dr = build_dems_reps(_make_party_votes(shares))
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
    path = _write_approval_file(tmp_path, [("1/1/1978", "12/31/1978", 55, 37, 8)])
    result = load_approval(path)
    assert result[result["year"] == 1978]["pres_approval"].values[0] > 0


def test_approval_rep_president_negative(tmp_path):
    path = _write_approval_file(tmp_path, [("1/1/1982", "12/31/1982", 42, 50, 8)])
    result = load_approval(path)
    assert result[result["year"] == 1982]["pres_approval"].values[0] < 0


def test_approval_disapproval_signs_are_opposite(tmp_path):
    path = _write_approval_file(tmp_path, [("1/1/1982", "12/31/1982", 42, 50, 8)])
    result = load_approval(path)
    row = result[result["year"] == 1982].iloc[0]
    assert (row["pres_approval"] > 0) != (row["pres_disapproval"] > 0)


# --- load_house_data ---

def _write_house_csv(tmp_path, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "house.csv"
    df.to_csv(path, index=False)
    return str(path)


def _base_row(**overrides) -> dict:
    row = {
        "year": 2020, "state_po": "CA", "district": 1,
        "candidate": "JOHN DOE", "party": "DEMOCRAT",
        "candidatevotes": 100, "totalvotes": 200,
        "stage": "GEN", "special": "FALSE", "runoff": "FALSE",
        "writein": "FALSE", "unofficial": "FALSE", "fusion_ticket": "FALSE",
        "state_fips": 1, "state_cen": 1, "state_ic": 1,
        "office": "US HOUSE", "mode": "TOTAL", "version": 1,
    }
    row.update(overrides)
    return row


def test_load_house_data_filters_non_gen_stage(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(stage="GEN"), _base_row(stage="PRIMARY")])
    assert len(load_house_data(path)) == 1


def test_load_house_data_filters_special_elections(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(special="FALSE"), _base_row(special="TRUE")])
    assert len(load_house_data(path)) == 1


def test_load_house_data_filters_writeins(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(writein="FALSE"), _base_row(writein="TRUE")])
    assert len(load_house_data(path)) == 1


def test_load_house_data_drops_null_party(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(party="DEMOCRAT"), _base_row(party=None)])
    assert len(load_house_data(path)) == 1


def test_load_house_data_drops_negative_votes(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(candidatevotes=100), _base_row(candidatevotes=-1)])
    assert len(load_house_data(path)) == 1


def test_load_house_data_normalizes_dfl_party(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(party="DEMOCRATIC-FARMER-LABOR")])
    assert load_house_data(path)["party"].values[0] == "DEMOCRAT"


def test_load_house_data_normalizes_foglietta(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(party="FOGLIETTA (DEMOCRAT)")])
    assert load_house_data(path)["party"].values[0] == "DEMOCRAT"


def test_load_house_data_normalizes_independent_republican(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row(party="INDEPENDENT-REPUBLICAN")])
    assert load_house_data(path)["party"].values[0] == "REPUBLICAN"


def test_load_house_data_drops_metadata_columns(tmp_path):
    path = _write_house_csv(tmp_path, [_base_row()])
    df = load_house_data(path)
    dropped = {"stage", "special", "runoff", "writein", "unofficial", "fusion_ticket",
               "state_fips", "state_cen", "state_ic", "office", "mode", "version"}
    assert dropped.isdisjoint(df.columns)


# --- integration: feature matrix ---

def test_feature_matrix_no_nan(real_data):
    assert real_data["X"].isnull().sum().sum() == 0


def test_feature_matrix_expected_columns(real_data):
    expected = {
        "dem_lean", "dem_lean_trend", "pres_approval", "pres_disapproval",
        "generic_ballot", "is_incumbent_on_ballot", "redistricting",
        "nation_incumbent_party_REPUBLICAN", "district_incumbent_party_REPUBLICAN",
    }
    assert set(real_data["X"].columns) == expected


def test_feature_matrix_returns_years(real_data):
    assert isinstance(real_data["years"], pd.Series)
    assert set(real_data["years"].unique()).issubset(set(range(1976, 2025, 2)))


def test_feature_matrix_years_aligned_with_X(real_data):
    assert list(real_data["X"].index) == list(real_data["years"].index)


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


def test_build_pred_2026_returns_tuple(real_data):
    result = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    assert isinstance(result, tuple) and len(result) == 2


def test_build_pred_2026_district_ids_aligned(real_data):
    pred, district_ids = build_pred_2026(
        real_data["winners"], real_data["dems_reps"], real_data["X"].columns
    )
    assert len(pred) == len(district_ids)
    assert set(district_ids.columns) == {"state_po", "district"}


def test_build_pred_2026_covers_most_districts(real_data):
    pred, _ = build_pred_2026(
        real_data["winners"], real_data["dems_reps"], real_data["X"].columns
    )
    assert len(pred) > 400
