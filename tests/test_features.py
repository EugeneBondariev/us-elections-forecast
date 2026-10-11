import pandas as pd
import pytest

from src.constants import Col, Val, GENERIC_BALLOT, PRES_PARTY, REDISTRICTING_YEARS
from src.loaders import load_approval, load_clean_house_data
from src.features import build_dems_reps, build_pred_2026

ELECTION_YEARS = set(range(1976, 2025, 2))


class TestConstants:
    def test_generic_ballot_covers_all_election_years(self):
        assert ELECTION_YEARS.issubset(GENERIC_BALLOT.keys())

    def test_generic_ballot_2026_dem_favorable(self):
        assert GENERIC_BALLOT[2026] > 0

    def test_pres_party_covers_all_election_years(self):
        assert ELECTION_YEARS.issubset(PRES_PARTY.keys())

    def test_redistricting_years(self):
        assert REDISTRICTING_YEARS == {1982, 1992, 2002, 2012, 2022}


class TestBuildDemsReps:
    @staticmethod
    def _make_party_votes(shares: dict, state_po: str = "CA", district: int = 1) -> pd.DataFrame:
        rows = []
        for year, dem_share in shares.items():
            dem_v = round(dem_share * 100)
            rep_v = 100 - dem_v
            rows += [
                {Col.YEAR: year, Col.STATE: state_po, Col.DISTRICT: district,
                 Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: dem_v, Col.TOTAL_VOTES: 100},
                {Col.YEAR: year, Col.STATE: state_po, Col.DISTRICT: district,
                 Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: rep_v, Col.TOTAL_VOTES: 100},
            ]
        return pd.DataFrame(rows)

    def test_dem_lean_uses_only_prior_elections(self):
        # Increasing share 0.4→0.8; dem_lean for 1984 must equal mean(0.4,0.5,0.6,0.7)=0.55,
        # not 0.6 which would happen if the current year (0.8) leaked in.
        shares = {1976: 0.4, 1978: 0.5, 1980: 0.6, 1982: 0.7, 1984: 0.8}
        dr = build_dems_reps(self._make_party_votes(shares))
        row = dr[(dr[Col.STATE] == "CA") & (dr[Col.YEAR] == 1984)]
        assert pytest.approx(row[Col.DEM_LEAN].values[0], abs=1e-6) == 0.55

    def test_dem_lean_trend_slope(self):
        # Linearly increasing share → trend slope should be 0.1 per election
        shares = {1976: 0.4, 1978: 0.5, 1980: 0.6, 1982: 0.7, 1984: 0.8}
        dr = build_dems_reps(self._make_party_votes(shares))
        row = dr[(dr[Col.STATE] == "CA") & (dr[Col.YEAR] == 1984)]
        assert pytest.approx(row[Col.DEM_LEAN_TREND].values[0], abs=1e-6) == 0.1

    def test_dem_lean_trend_flat_is_zero(self):
        shares = {1976: 0.6, 1978: 0.6, 1980: 0.6, 1982: 0.6, 1984: 0.6}
        dr = build_dems_reps(self._make_party_votes(shares))
        row = dr[(dr[Col.STATE] == "CA") & (dr[Col.YEAR] == 1984)]
        assert pytest.approx(row[Col.DEM_LEAN_TREND].values[0], abs=1e-6) == 0.0


class TestLoadApproval:
    @staticmethod
    def _write_approval_file(tmp_path, rows: list[tuple]) -> str:
        lines = "\n".join(f"{start}\t{end}\t{approve}\t{disapprove}\t{unsure}"
                          for start, end, approve, disapprove, unsure in rows)
        p = tmp_path / "approval.xls"
        p.write_text(lines)
        return str(p)

    def test_dem_president_positive(self, tmp_path):
        path = self._write_approval_file(tmp_path, [("1/1/1978", "12/31/1978", 55, 37, 8)])
        result = load_approval(path)
        assert result[result[Col.YEAR] == 1978][Col.PRES_APPROVAL].values[0] > 0

    def test_rep_president_negative(self, tmp_path):
        path = self._write_approval_file(tmp_path, [("1/1/1982", "12/31/1982", 42, 50, 8)])
        result = load_approval(path)
        assert result[result[Col.YEAR] == 1982][Col.PRES_APPROVAL].values[0] < 0

    def test_disapproval_signs_are_opposite(self, tmp_path):
        path = self._write_approval_file(tmp_path, [("1/1/1982", "12/31/1982", 42, 50, 8)])
        result = load_approval(path)
        row = result[result[Col.YEAR] == 1982].iloc[0]
        assert (row[Col.PRES_APPROVAL] > 0) != (row[Col.PRES_DISAPPROVAL] > 0)


class TestLoadHouseData:
    @staticmethod
    def _write_house_csv(tmp_path, rows: list[dict]) -> str:
        df = pd.DataFrame(rows)
        path = tmp_path / "house.csv"
        df.to_csv(path, index=False)
        return str(path)

    @staticmethod
    def _base_row(**overrides) -> dict:
        row = {
            Col.YEAR: 2020, Col.STATE: "CA", Col.DISTRICT: 1,
            Col.CANDIDATE: "JOHN DOE", Col.PARTY: Val.DEMOCRAT,
            Col.CANDIDATE_VOTES: 100, Col.TOTAL_VOTES: 200,
            Col.STAGE: Val.GEN, Col.SPECIAL: "FALSE", Col.RUNOFF: "FALSE",
            Col.WRITE_IN: "FALSE", Col.UNOFFICIAL: "FALSE", Col.FUSION_TICKET: "FALSE",
            Col.STATE_FIPS: 1, Col.STATE_CEN: 1, Col.STATE_IC: 1,
            Col.OFFICE: "US HOUSE", Col.MODE: "TOTAL", Col.VERSION: 1,
        }
        row.update(overrides)
        return row

    def test_filters_non_gen_stage(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.STAGE: Val.GEN}), self._base_row(**{Col.STAGE: "PRIMARY"})])
        assert len(load_clean_house_data(path)) == 1

    def test_filters_special_elections(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.SPECIAL: "FALSE"}), self._base_row(**{Col.SPECIAL: "TRUE"})])
        assert len(load_clean_house_data(path)) == 1

    def test_filters_runoffs(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.RUNOFF: "FALSE"}), self._base_row(**{Col.RUNOFF: "TRUE"})])
        assert len(load_clean_house_data(path)) == 1

    def test_filters_unofficial(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.UNOFFICIAL: "FALSE"}), self._base_row(**{Col.UNOFFICIAL: "TRUE"})])
        assert len(load_clean_house_data(path)) == 1

    def test_filters_writeins(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.WRITE_IN: "FALSE"}), self._base_row(**{Col.WRITE_IN: "TRUE"})])
        assert len(load_clean_house_data(path)) == 1

    def test_drops_null_party(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.PARTY: Val.DEMOCRAT}), self._base_row(**{Col.PARTY: None})])
        assert len(load_clean_house_data(path)) == 1

    def test_drops_negative_votes(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.CANDIDATE_VOTES: 100}), self._base_row(**{Col.CANDIDATE_VOTES: -1})])
        assert len(load_clean_house_data(path)) == 1

    def test_normalizes_dfl_party(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.PARTY: Val.DEM_FARMER_LABOR})])
        assert load_clean_house_data(path)[Col.PARTY].values[0] == Val.DEMOCRAT

    def test_normalizes_foglietta(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.PARTY: Val.FOGLIETTA_DEM})])
        assert load_clean_house_data(path)[Col.PARTY].values[0] == Val.DEMOCRAT

    def test_normalizes_independent_republican(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row(**{Col.PARTY: Val.INDEPENDENT_REP})])
        assert load_clean_house_data(path)[Col.PARTY].values[0] == Val.REPUBLICAN

    def test_drops_metadata_columns(self, tmp_path):
        path = self._write_house_csv(tmp_path, [self._base_row()])
        df = load_clean_house_data(path)
        dropped = {Col.STAGE, Col.SPECIAL, Col.RUNOFF, Col.WRITE_IN, Col.UNOFFICIAL,
                   Col.FUSION_TICKET, Col.STATE_FIPS, Col.STATE_CEN, Col.STATE_IC,
                   Col.OFFICE, Col.MODE, Col.VERSION}
        assert dropped.isdisjoint(df.columns)


class TestBuildFeatureMatrix:
    def test_no_nan(self, real_data):
        assert real_data["X"].isnull().sum().sum() == 0

    def test_expected_columns(self, real_data):
        required = {
            Col.DEM_LEAN, Col.DEM_LEAN_TREND, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL,
            Col.GENERIC_BALLOT, Col.IS_INCUMBENT, Col.REDISTRICTING,
            Col.PREV_DEM_SHARE, Col.INC_X_BALLOT,
            Col.ECON_GROWTH, Col.UNEMPLOYMENT,
            "nation_incumbent_party_REPUBLICAN", "district_incumbent_party_REPUBLICAN",
        }
        assert required.issubset(set(real_data["X"].columns))

    def test_returns_years(self, real_data):
        assert isinstance(real_data["years"], pd.Series)
        assert set(real_data["years"].unique()).issubset(set(range(1976, 2025, 2)))

    def test_years_aligned_with_X(self, real_data):
        assert list(real_data["X"].index) == list(real_data["years"].index)

    def test_binary_target(self, real_data):
        assert set(real_data["y"].unique()).issubset({0, 1})

    def test_redistricting_flag_on_correct_years(self, real_data):
        winners = real_data["winners"]
        for year in REDISTRICTING_YEARS:
            rows = winners[winners[Col.YEAR] == year]
            if len(rows):
                assert (rows[Col.REDISTRICTING] == 1).all()
        non_redist = winners[~winners[Col.YEAR].isin(REDISTRICTING_YEARS)]
        assert (non_redist[Col.REDISTRICTING] == 0).all()


class TestBuildPred2026:
    def test_returns_tuple(self, real_data):
        result = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
        assert isinstance(result, tuple) and len(result) == 2

    def test_district_ids_aligned(self, real_data):
        pred, district_ids = build_pred_2026(
            real_data["winners"], real_data["dems_reps"], real_data["X"].columns
        )
        assert len(pred) == len(district_ids)
        assert set(district_ids.columns) == {Col.STATE, Col.DISTRICT}

    def test_covers_most_districts(self, real_data):
        pred, _ = build_pred_2026(
            real_data["winners"], real_data["dems_reps"], real_data["X"].columns
        )
        assert len(pred) > 400
