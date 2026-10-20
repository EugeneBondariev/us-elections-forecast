"""Unit tests for src.data.candidate_nlp. No network access, no files on disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.candidate_nlp import (
    build_candidate_descriptions,
    build_candidate_quality,
    compute_experience_cycles,
    compute_quality_scores,
    compute_receipts_rank,
    load_candidate_quality,
    select_top_candidates,
    tfidf_components,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def fec_df() -> pd.DataFrame:
    """Synthetic FEC raw data covering 3 cycles, 3 states, 2 districts each,
    both parties, with one DEM candidate appearing in all 3 cycles."""
    rows: list[dict] = []

    def _row(
        year: int,
        state_po: str,
        district: int,
        party: str,
        candidate_id: str,
        candidate_name: str,
        receipts: float,
    ) -> dict:
        return {
            "year": year,
            "state_po": state_po,
            "district": district,
            "party": party,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "receipts": receipts,
            "individual_receipts": receipts * 0.7,
            "individual_itemized_contributions": receipts * 0.5,
            "individual_unitemized_contributions": receipts * 0.2,
            "other_political_committee_contributions": receipts * 0.1,
            "candidate_contribution": 0.0,
            "disbursements": receipts * 0.9,
        }

    # CA-1: DEM candidate H0CA01001 appears in all 3 cycles (experience_cycles=3)
    for year, receipts in [(2018, 500_000), (2020, 600_000), (2022, 700_000)]:
        rows.append(_row(year, "CA", 1, "DEM", "H0CA01001", "Smith J", receipts))
        rows.append(_row(year, "CA", 1, "REP", f"H0CA01002_{year}", f"Jones {year}", 300_000))

    # CA-2: fresh candidates each cycle (experience_cycles=1 each)
    for year, receipts_d, receipts_r in [
        (2018, 200_000, 180_000),
        (2020, 220_000, 210_000),
        (2022, 240_000, 230_000),
    ]:
        rows.append(_row(year, "CA", 2, "DEM", f"H0CA02D_{year}", f"Doe {year}", receipts_d))
        rows.append(_row(year, "CA", 2, "REP", f"H0CA02R_{year}", f"Roe {year}", receipts_r))

    # TX-1 and TX-2: 2-cycle candidates
    for year in [2020, 2022]:
        rows.append(_row(year, "TX", 1, "DEM", "H0TX01001", "Lee A", 150_000))
        rows.append(_row(year, "TX", 1, "REP", "H0TX01002", "Kim B", 160_000))
        rows.append(_row(year, "TX", 2, "DEM", f"H0TX02D_{year}", "Park C", 120_000))
        rows.append(_row(year, "TX", 2, "REP", f"H0TX02R_{year}", "Chen D", 130_000))

    # NY-1 and NY-2: single-cycle 2022 only
    rows.append(_row(2022, "NY", 1, "DEM", "H0NY01D", "Adams E", 80_000))
    rows.append(_row(2022, "NY", 1, "REP", "H0NY01R", "Brown F", 90_000))
    rows.append(_row(2022, "NY", 2, "DEM", "H0NY02D", "Clark G", 70_000))
    rows.append(_row(2022, "NY", 2, "REP", "H0NY02R", "Davis H", 95_000))

    # Add a third-party candidate to verify filtering
    rows.append(_row(2022, "NY", 1, "IND", "H0NY01I", "Green X", 5_000))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_experience_cycles
# ---------------------------------------------------------------------------

class TestComputeExperienceCycles:
    def test_compute_experience_cycles_multi_cycle(self, fec_df: pd.DataFrame) -> None:
        result = compute_experience_cycles(fec_df)
        exp = result.loc[result["candidate_id"] == "H0CA01001", "experience_cycles"].iloc[0]
        assert exp == 3

    def test_compute_experience_cycles_single_cycle(self, fec_df: pd.DataFrame) -> None:
        result = compute_experience_cycles(fec_df)
        single_ids = ["H0NY01D", "H0NY01R", "H0NY02D", "H0NY02R"]
        for cid in single_ids:
            exp = result.loc[result["candidate_id"] == cid, "experience_cycles"].iloc[0]
            assert exp == 1, f"Expected 1 for {cid}, got {exp}"

    def test_compute_experience_cycles_no_mutation(self, fec_df: pd.DataFrame) -> None:
        original_cols = list(fec_df.columns)
        _ = compute_experience_cycles(fec_df)
        assert list(fec_df.columns) == original_cols
        assert "experience_cycles" not in fec_df.columns


# ---------------------------------------------------------------------------
# select_top_candidates
# ---------------------------------------------------------------------------

class TestSelectTopCandidates:
    def test_select_top_candidates_one_per_group(self, fec_df: pd.DataFrame) -> None:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        counts = top.groupby(["year", "state_po", "district", "party"]).size()
        assert (counts == 1).all()

    def test_select_top_candidates_highest_receipts(self, fec_df: pd.DataFrame) -> None:
        # Add a lower-receipts DEM to CA-1 2022 to make the selection non-trivial
        extra_row = {
            "year": 2022, "state_po": "CA", "district": 1, "party": "DEM",
            "candidate_id": "H0CA01999", "candidate_name": "Low Raiser",
            "receipts": 50_000,
            "individual_receipts": 35_000,
            "individual_itemized_contributions": 25_000,
            "individual_unitemized_contributions": 10_000,
            "other_political_committee_contributions": 5_000,
            "candidate_contribution": 0.0,
            "disbursements": 45_000,
        }
        augmented = pd.concat([fec_df, pd.DataFrame([extra_row])], ignore_index=True)
        with_exp = compute_experience_cycles(augmented)
        top = select_top_candidates(with_exp)
        ca1_2022_dem = top[
            (top["year"] == 2022) & (top["state_po"] == "CA")
            & (top["district"] == 1) & (top["party"] == "DEM")
        ]
        assert len(ca1_2022_dem) == 1
        assert ca1_2022_dem.iloc[0]["receipts"] == 700_000

    def test_select_top_candidates_filters_to_dem_rep(self, fec_df: pd.DataFrame) -> None:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        assert set(top["party"].unique()).issubset({"DEM", "REP"})


# ---------------------------------------------------------------------------
# compute_receipts_rank
# ---------------------------------------------------------------------------

class TestComputeReceiptsRank:
    def test_compute_receipts_rank_range(self, fec_df: pd.DataFrame) -> None:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        ranked = compute_receipts_rank(top)
        assert ranked["receipts_rank"].between(0.0, 1.0).all()

    def test_compute_receipts_rank_within_year_party(self, fec_df: pd.DataFrame) -> None:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        ranked = compute_receipts_rank(top)
        # Within each (year, party) group, ranks should span [low, high] independently
        for (year, party), group in ranked.groupby(["year", "party"]):
            ranks = group["receipts_rank"].values
            assert ranks.max() <= 1.0 + 1e-9
            assert ranks.min() >= 0.0 - 1e-9
            # For groups with >1 candidate, ranks must not all be identical
            if len(group) > 1:
                assert not np.allclose(ranks, ranks[0])


# ---------------------------------------------------------------------------
# build_candidate_descriptions
# ---------------------------------------------------------------------------

class TestBuildCandidateDescriptions:
    def _get_candidates(self, fec_df: pd.DataFrame) -> pd.DataFrame:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        return compute_receipts_rank(top)

    def test_build_candidate_descriptions_shape(self, fec_df: pd.DataFrame) -> None:
        candidates = self._get_candidates(fec_df)
        descriptions = build_candidate_descriptions(candidates)
        assert len(descriptions) == len(candidates)

    def test_build_candidate_descriptions_contains_party(self, fec_df: pd.DataFrame) -> None:
        candidates = self._get_candidates(fec_df)
        descriptions = build_candidate_descriptions(candidates)
        for i, (_, row) in enumerate(candidates.iterrows()):
            desc = descriptions.iloc[i]
            if row["party"] == "DEM":
                assert "democrat" in desc, f"Missing 'democrat' in: {desc}"
            else:
                assert "republican" in desc, f"Missing 'republican' in: {desc}"

    def test_build_candidate_descriptions_incumbent_label(self, fec_df: pd.DataFrame) -> None:
        candidates = self._get_candidates(fec_df)
        # H0CA01001 appears in all 3 cycles → experience_cycles=3 → "incumbent"
        multi = candidates[candidates["candidate_id"] == "H0CA01001"]
        assert len(multi) >= 1
        descriptions = build_candidate_descriptions(candidates)
        for idx in multi.index:
            desc = descriptions.loc[idx]
            assert "incumbent" in desc, f"Expected 'incumbent' for multi-cycle candidate: {desc}"


# ---------------------------------------------------------------------------
# tfidf_components
# ---------------------------------------------------------------------------

class TestTfidfComponents:
    def _get_descriptions(self, fec_df: pd.DataFrame) -> pd.Series:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        ranked = compute_receipts_rank(top)
        return build_candidate_descriptions(ranked)

    def test_tfidf_components_shape(self, fec_df: pd.DataFrame) -> None:
        descriptions = self._get_descriptions(fec_df)
        result = tfidf_components(descriptions, n_components=2)
        assert result.shape == (len(descriptions), 2)
        assert list(result.columns) == ["tfidf_1", "tfidf_2"]

    def test_tfidf_components_finite(self, fec_df: pd.DataFrame) -> None:
        descriptions = self._get_descriptions(fec_df)
        result = tfidf_components(descriptions, n_components=2)
        assert not result.isnull().any().any()
        assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# compute_quality_scores
# ---------------------------------------------------------------------------

class TestComputeQualityScores:
    def test_compute_quality_scores_range(self, fec_df: pd.DataFrame) -> None:
        with_exp = compute_experience_cycles(fec_df)
        top = select_top_candidates(with_exp)
        ranked = compute_receipts_rank(top)
        scored = compute_quality_scores(ranked)
        assert scored["quality_score"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# build_candidate_quality (integration over synthetic data)
# ---------------------------------------------------------------------------

class TestBuildCandidateQuality:
    def test_build_candidate_quality_output_columns(
        self, fec_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "fundraising_raw.csv"
        fec_df.to_csv(csv_path, index=False)
        result = build_candidate_quality(fec_raw_path=csv_path)
        expected_cols = {
            "year", "state_po", "district",
            "candidate_quality_dem_share",
            "experience_cycles_dem",
            "experience_cycles_rep",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_build_candidate_quality_dem_share_range(
        self, fec_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "fundraising_raw.csv"
        fec_df.to_csv(csv_path, index=False)
        result = build_candidate_quality(fec_raw_path=csv_path)
        shares = result["candidate_quality_dem_share"]
        assert shares.between(0.0, 1.0).all()

    def test_build_candidate_quality_multi_cycle_candidate_higher_score(
        self, tmp_path: Path
    ) -> None:
        """CA-1 DEM (3 cycles) should have a higher quality_score than a first-cycle
        DEM in another district with identical receipts."""
        rows = []

        def _row(
            year: int, state: str, district: int, party: str,
            cid: str, receipts: float,
        ) -> dict:
            return {
                "year": year, "state_po": state, "district": district,
                "party": party, "candidate_id": cid, "candidate_name": cid,
                "receipts": receipts,
                "individual_receipts": receipts * 0.7,
                "individual_itemized_contributions": receipts * 0.5,
                "individual_unitemized_contributions": receipts * 0.2,
                "other_political_committee_contributions": receipts * 0.1,
                "candidate_contribution": 0.0,
                "disbursements": receipts * 0.9,
            }

        # DEM in CA-1: same candidate_id across 3 cycles, same receipts each time
        for year in [2018, 2020, 2022]:
            rows.append(_row(year, "CA", 1, "DEM", "VETERAN_DEM", 300_000))
            rows.append(_row(year, "CA", 1, "REP", f"CA1REP_{year}", 300_000))

        # DEM in CA-2: first-cycle only, same receipts as VETERAN_DEM
        rows.append(_row(2022, "CA", 2, "DEM", "ROOKIE_DEM", 300_000))
        rows.append(_row(2022, "CA", 2, "REP", "CA2REP", 300_000))

        df = pd.DataFrame(rows)
        csv_path = tmp_path / "fundraising_raw.csv"
        df.to_csv(csv_path, index=False)

        result = build_candidate_quality(fec_raw_path=csv_path)

        veteran_row = result[
            (result["state_po"] == "CA") & (result["district"] == 1) & (result["year"] == 2022)
        ]
        rookie_row = result[
            (result["state_po"] == "CA") & (result["district"] == 2) & (result["year"] == 2022)
        ]
        assert len(veteran_row) == 1
        assert len(rookie_row) == 1
        assert (
            veteran_row["experience_cycles_dem"].iloc[0]
            > rookie_row["experience_cycles_dem"].iloc[0]
        ), "Veteran should have more experience cycles than rookie"
        # Veteran has 3 cycles vs rookie's 1 → veteran dem_share should be higher
        assert (
            veteran_row["candidate_quality_dem_share"].iloc[0]
            > rookie_row["candidate_quality_dem_share"].iloc[0]
        ), (
            f"Veteran dem_share={veteran_row['candidate_quality_dem_share'].iloc[0]:.4f} "
            f"should exceed rookie dem_share={rookie_row['candidate_quality_dem_share'].iloc[0]:.4f}"
        )


# ---------------------------------------------------------------------------
# load_candidate_quality
# ---------------------------------------------------------------------------

class TestLoadCandidateQuality:
    def test_load_candidate_quality_returns_none_if_missing(self) -> None:
        result = load_candidate_quality(Path("nonexistent.csv"))
        assert result is None
