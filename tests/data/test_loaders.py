from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.constants import Col, Val
from src.data.loaders import load_approval, load_clean_house_data, load_cook_ratings, load_pres_lean


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

    def test_drops_uncontested_zero_total_votes(self, tmp_path):
        """Real MIT data has uncontested races recorded as candidatevotes=totalvotes=0
        (e.g. FL-24 2016, Frederica Wilson ran unopposed). Caught via the actual GitHub
        Release mirror of 1976-2024-house.tab tripping validate_house_data's totalvotes
        gt=0 bound — fixed by dropping these explicitly instead of letting them become
        a silent 0/0 NaN three functions downstream."""
        path = self._write_house_csv(tmp_path, [
            self._base_row(**{Col.CANDIDATE_VOTES: 100, Col.TOTAL_VOTES: 200}),
            self._base_row(**{Col.CANDIDATE: "UNOPPOSED", Col.CANDIDATE_VOTES: 0, Col.TOTAL_VOTES: 0}),
        ])
        result = load_clean_house_data(path)
        assert len(result) == 1
        assert (result[Col.TOTAL_VOTES] > 0).all()

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


class TestLoadCookRatings:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_cook_ratings(tmp_path / "nonexistent.csv")
        assert result.empty
        assert Col.COOK_PROB in result.columns

    def test_loads_correct_columns(self, tmp_path):
        p = tmp_path / "cook.csv"
        p.write_text("state_po,district,rating,cook_prob\nPA,7,lean-d,0.72\nAZ,6,tossup,0.50\n")
        result = load_cook_ratings(p)
        assert len(result) == 2
        assert set(result.columns) == {Col.STATE, Col.DISTRICT, Col.COOK_PROB}

    def test_cook_prob_bounded(self, tmp_path):
        p = tmp_path / "cook.csv"
        p.write_text("state_po,district,rating,cook_prob\nPA,7,lean-d,0.72\nAZ,6,tossup,0.50\n")
        result = load_cook_ratings(p)
        assert result[Col.COOK_PROB].between(0, 1).all()


class TestLoadPresLean:
    @pytest.fixture(autouse=True)
    def _patch_validate(self):
        with patch("src.data.loaders.validate_pres_lean"):
            yield

    @staticmethod
    def _write_2008_2020(tmp_path: Path) -> Path:
        lines = (
            "header row 1\n"
            "header row 2\n"
            "CA-1,incumbent,DEMOCRAT,55,45,52,48,54,46,53,47\n"
        )
        p = tmp_path / "pres_2008_2020.csv"
        p.write_text(lines)
        return p

    @staticmethod
    def _write_2020_2024(tmp_path: Path) -> Path:
        lines = (
            "header row 1\n"
            "header row 2\n"
            "header row 3\n"
            "CA-1,incumbent,DEMOCRAT,51,49,2,55,45,10\n"
        )
        p = tmp_path / "pres_2020_2024.csv"
        p.write_text(lines)
        return p

    def test_required_columns(self, tmp_path):
        result = load_pres_lean(
            self._write_2008_2020(tmp_path),
            self._write_2020_2024(tmp_path),
        )
        for col in (Col.STATE, Col.DISTRICT, "pres_year", Col.PRES_LEAN):
            assert col in result.columns

    def test_pres_lean_bounded(self, tmp_path):
        result = load_pres_lean(
            self._write_2008_2020(tmp_path),
            self._write_2020_2024(tmp_path),
        )
        assert result[Col.PRES_LEAN].between(0, 1).all()

    def test_covers_expected_years(self, tmp_path):
        result = load_pres_lean(
            self._write_2008_2020(tmp_path),
            self._write_2020_2024(tmp_path),
        )
        assert {2008, 2012, 2016, 2020, 2024}.issubset(set(result["pres_year"].unique()))
