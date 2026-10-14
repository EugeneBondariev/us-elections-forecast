"""Tests for src/fundraising.py aggregation logic. No network access needed —
all tests operate on hand-crafted in-memory records."""
from __future__ import annotations

import pandas as pd

from src.fundraising import _RAW_FIELDS, _aggregate, _top_candidate_per_group


def _record(party: str, candidate_id: str, receipts: float, disbursements: float,
            year: int = 2024, state_po: str = "CA", district: int = 1) -> dict:
    return {
        "year": year, "state_po": state_po, "district": district, "party": party,
        "candidate_id": candidate_id, "candidate_name": candidate_id,
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
        df = pd.DataFrame([
            _record("DEM", "A", receipts=100.0, disbursements=20.0),
            _record("DEM", "B", receipts=40.0, disbursements=90.0),
        ])
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
            _record("DEM", "B", receipts=40.0, disbursements=90.0),   # top DEM by disbursements
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
