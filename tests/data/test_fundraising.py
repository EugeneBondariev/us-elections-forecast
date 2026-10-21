"""Tests for src/fundraising.py aggregation logic. No network access needed —
all tests operate on hand-crafted in-memory records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.fundraising import _RAW_FIELDS, _aggregate, _save_raw, _top_candidate_per_group
from src.types import FECCandidateRecord


def _record(
    party: str,
    candidate_id: str,
    receipts: float,
    disbursements: float,
    year: int = 2024,
    state_po: str = "CA",
    district: int = 1,
) -> FECCandidateRecord:
    return {
        "year": year,
        "state_po": state_po,
        "district": district,
        "party": party,
        "candidate_id": candidate_id,
        "candidate_name": candidate_id,
        "receipts": receipts,
        "individual_receipts": receipts * 0.8,
        "individual_itemized_contributions": receipts * 0.5,
        "individual_unitemized_contributions": 0.0,
        "other_political_committee_contributions": receipts * 0.1,
        "candidate_contribution": 0.0,
        "disbursements": disbursements,
    }


class TestTopCandidatePerGroup:
    def test_picks_row_with_max_receipts(self):
        df = pd.DataFrame(
            [
                _record("DEM", "A", receipts=100.0, disbursements=20.0),
                _record("DEM", "B", receipts=40.0, disbursements=90.0),
            ]
        )
        result = _top_candidate_per_group(df)
        assert len(result) == 1
        row = result.iloc[0]
        # Must come entirely from candidate A (max receipts), not a mix of A and B.
        assert row["receipts"] == 100.0
        assert row["disbursements"] == 20.0

    def test_empty_input_returns_empty_frame_with_columns(self):
        # Realistic empty input: a zero-row slice of the full records frame (as
        # produced by `df[df["party"] == "REP"]` when no REP candidate filed),
        # which always carries every _RAW_FIELDS column even with 0 rows.
        df = pd.DataFrame(columns=["year", "state_po", "district", "party"] + _RAW_FIELDS)
        result = _top_candidate_per_group(df)
        assert result.empty
        assert "receipts" in result.columns


class TestAggregate:
    def test_does_not_mix_fields_across_candidates(self):
        """Regression test: a multi-candidate DEM primary must not stitch
        candidate A's receipts together with candidate B's disbursements."""
        records = [
            _record("DEM", "A", receipts=100.0, disbursements=20.0),  # top DEM by receipts
            _record("DEM", "B", receipts=40.0, disbursements=90.0),  # top DEM by disbursements
            _record("REP", "C", receipts=60.0, disbursements=55.0),
        ]
        result = _aggregate(records)
        row = result.iloc[0]

        assert abs(row["fundraising_dem_share"] - 100 / 160) < 1e-9
        # Bug would have used disbursements=90 (candidate B's) instead of 20 (candidate A's).
        assert abs(row["spending_dem_share"] - 20 / (20 + 55)) < 1e-9

    def test_zero_total_defaults_to_half(self):
        """When neither party reports any receipts for a field, share falls back to 0.5
        rather than dividing by zero."""
        records = [
            _record("DEM", "A", receipts=0.0, disbursements=0.0),
            _record("REP", "C", receipts=0.0, disbursements=0.0),
        ]
        result = _aggregate(records)
        row = result.iloc[0]
        assert row["fundraising_dem_share"] == 0.5
        assert row["spending_dem_share"] == 0.5

    def test_no_opposing_candidate_gives_full_share(self):
        """If only DEM filed in this district-year, the DEM share of receipts is 1.0
        (REP contributes 0 via the outer-merge fillna)."""
        records = [_record("DEM", "A", receipts=100.0, disbursements=20.0)]
        result = _aggregate(records)
        row = result.iloc[0]
        assert row["fundraising_dem_share"] == 1.0


class TestAggregateNegativeClipping:
    def test_negative_receipts_clipped_to_zero(self) -> None:
        """DEM receipts=-50 clips to 0; share is 0/(0+100)=0.0, not negative."""
        records = [
            _record("DEM", "A", receipts=-50.0, disbursements=0.0),
            _record("REP", "B", receipts=100.0, disbursements=0.0),
        ]
        result = _aggregate(records)
        assert result.iloc[0]["fundraising_dem_share"] == 0.0


class TestSaveRaw:
    def test_writes_sortable_csv(self, tmp_path: Path) -> None:
        records: list[FECCandidateRecord] = [
            _record(
                "DEM", "A", receipts=100.0, disbursements=20.0, year=2022, state_po="TX", district=7
            ),
            _record(
                "REP", "B", receipts=80.0, disbursements=15.0, year=2020, state_po="CA", district=1
            ),
            _record(
                "DEM", "C", receipts=60.0, disbursements=10.0, year=2022, state_po="CA", district=3
            ),
        ]
        out = tmp_path / "out.csv"
        _save_raw(records=records, raw_path=out)
        assert out.exists()
        df = pd.read_csv(out)
        assert "year" in df.columns
        assert "state_po" in df.columns
        assert "district" in df.columns
        # Sorted by year then state_po then district: 2020/CA/1, 2022/CA/3, 2022/TX/7
        assert list(df["year"]) == [2020, 2022, 2022]
        assert list(df["state_po"]) == ["CA", "CA", "TX"]


class TestLoadFundraising:
    def test_no_cache_no_key_returns_none(self, tmp_path: Path, monkeypatch: object) -> None:
        from src.data.fundraising import load_fundraising

        monkeypatch.delenv("FEC_API_KEY", raising=False)
        monkeypatch.delenv("DATA_GOV_API_KEY", raising=False)
        result = load_fundraising(
            cache_path=tmp_path / "missing.csv",
            api_key=None,
        )
        assert result is None

    def test_cache_hit_returns_dataframe(self, tmp_path: Path) -> None:
        from src.data.fundraising import load_fundraising

        cache = tmp_path / "fundraising.csv"
        pd.DataFrame(
            {
                "year": [2020, 2022, 2024],
                "state_po": ["CA", "TX", "FL"],
                "district": [1, 2, 3],
                "fundraising_dem_share": [0.5, 0.6, 0.4],
                "large_donor_dem_share": [0.5, 0.6, 0.4],
                "pac_dem_share": [0.5, 0.6, 0.4],
                "spending_dem_share": [0.5, 0.6, 0.4],
                "surplus_dem_share": [0.5, 0.6, 0.4],
            }
        ).to_csv(cache, index=False)
        result = load_fundraising(cache_path=cache)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
