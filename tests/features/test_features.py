import pandas as pd
import pytest

from src.constants import Col, Val
from src.features.features import build_dems_reps, build_pred_2026


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


class TestBuildFeatureMatrix:
    def test_no_nan(self, real_data):
        assert real_data["X"].isnull().sum().sum() == 0

    def test_expected_columns(self, real_data):
        required = {
            Col.DEM_LEAN, Col.DEM_LEAN_TREND, Col.NET_PRES_APPROVAL,
            Col.GENERIC_BALLOT, Col.IS_INCUMBENT, Col.REDISTRICTING,
            Col.INC_X_BALLOT,
            Col.ECON_GROWTH, Col.UNEMPLOYMENT,
            Col.STATE_PARTISAN_LEAN,
            "nation_incumbent_party_REPUBLICAN", "district_incumbent_party_REPUBLICAN",
        }
        assert required.issubset(set(real_data["X"].columns))

    def test_dropped_collinear_columns(self, real_data):
        """prev_dem_vote_share and spending_dem_share must be excluded (high VIF)."""
        assert Col.PREV_DEM_SHARE not in real_data["X"].columns
        assert Col.SPENDING_DEM_SHARE not in real_data["X"].columns

    def test_y_vote_share_continuous(self, real_data):
        yvs = real_data["y_vote_share"]
        assert yvs.dtype == float or str(yvs.dtype).startswith("float")
        assert yvs.between(0, 1).all()  # uncontested races imputed to 0.85 / 0.15
        assert len(yvs) == len(real_data["y"])

    def test_returns_years(self, real_data):
        assert isinstance(real_data["years"], pd.Series)
        assert set(real_data["years"].unique()).issubset(set(range(1976, 2025, 2)))

    def test_years_aligned_with_X(self, real_data):
        assert list(real_data["X"].index) == list(real_data["years"].index)

    def test_binary_target(self, real_data):
        assert set(real_data["y"].unique()).issubset({0, 1})

    def test_redistricting_flag_on_correct_years(self, real_data):
        from src.constants import REDISTRICTING_YEARS

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
