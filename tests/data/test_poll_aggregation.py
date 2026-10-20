"""Unit tests for src.data.poll_aggregation. No network, no disk I/O — all synthetic data."""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.poll_aggregation import (
    ELECTION_DATES,
    aggregate_year,
    build_poll_aggregates,
    estimate_house_effects,
    fetch_generic_ballot_polls,
    normalize_poll_columns,
    time_decay_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ELECTION_2018 = ELECTION_DATES[2018]  # 2018-11-06


def _poll(
    year: int,
    end_date: str,
    pollster: str,
    dem: float,
    rep: float,
) -> dict[str, object]:
    return {
        "year": year,
        "end_date": end_date,
        "pollster": pollster,
        "dem": dem,
        "rep": rep,
        "margin": dem - rep,
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def polls_df() -> pd.DataFrame:
    """Synthetic generic ballot polls across 2018, 2020, 2022.

    PollsterA: Dem-leaning (+2 above cycle average)
    PollsterB: neutral
    PollsterC: Rep-leaning (-2 below cycle average)

    Each pollster has 5 polls per year spread over 90 days before election day.
    dem + rep ≈ 88 (realistic; undecided not included).
    """
    rows: list[dict[str, object]] = []

    # True cycle margins (roughly): 2018 D+7, 2020 D+7, 2022 D-1
    # PollsterA adds +2, PollsterB +0, PollsterC -2
    cycle_configs = [
        # (year, base_dem, base_rep) → base margin close to known environment
        (2018, 47.5, 40.5),  # base D+7
        (2020, 47.5, 40.5),  # base D+7
        (2022, 43.5, 44.5),  # base D-1
    ]
    offsets = {
        "PollsterA": +2.0,
        "PollsterB": 0.0,
        "PollsterC": -2.0,
    }
    # 5 polls spread over 90 days before election, step ~18 days
    day_offsets = [7, 25, 43, 61, 79]

    for year, base_dem, base_rep in cycle_configs:
        election_date = ELECTION_DATES[year]
        for pollster, offset in offsets.items():
            for days_before in day_offsets:
                poll_date = election_date - datetime.timedelta(days=days_before)
                dem = base_dem + offset
                rep = base_rep
                rows.append(_poll(
                    year=year,
                    end_date=poll_date.isoformat(),
                    pollster=pollster,
                    dem=dem,
                    rep=rep,
                ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# normalize_poll_columns
# ---------------------------------------------------------------------------

class TestNormalizePollColumns:
    def test_normalize_keeps_canonical_columns(self):
        df = pd.DataFrame([_poll(2018, "2018-10-01", "PollsterA", 47.0, 40.0)])
        result = normalize_poll_columns(df)
        assert set(result.columns) == {"year", "end_date", "pollster", "dem", "rep", "margin"}

    def test_normalize_maps_adjusted_dem(self):
        df = pd.DataFrame([{
            "cycle": 2018,
            "end_date": "2018-10-01",
            "pollster": "PollsterA",
            "adj_dem": 47.5,
            "adj_rep": 40.5,
        }])
        result = normalize_poll_columns(df)
        assert result["dem"].iloc[0] == pytest.approx(47.5)
        assert result["rep"].iloc[0] == pytest.approx(40.5)

    def test_normalize_drops_missing_dem(self):
        df = pd.DataFrame([
            {"year": 2018, "end_date": "2018-10-01", "pollster": "A", "dem": np.nan, "rep": 40.0},
            {"year": 2018, "end_date": "2018-10-02", "pollster": "A", "dem": 47.0, "rep": 40.0},
        ])
        result = normalize_poll_columns(df)
        assert len(result) == 1
        assert result["dem"].iloc[0] == pytest.approx(47.0)


# ---------------------------------------------------------------------------
# estimate_house_effects
# ---------------------------------------------------------------------------

class TestEstimateHouseEffects:
    def test_estimate_house_effects_known_bias(self, polls_df: pd.DataFrame):
        effects = estimate_house_effects(polls=polls_df, min_polls=3)
        # PollsterA is consistently +2 above cycle average
        assert effects.get("PollsterA", 0.0) == pytest.approx(2.0, abs=1.0)

    def test_estimate_house_effects_min_polls_threshold(self):
        # PollsterSmall has only 2 polls — below min_polls=3
        df = pd.DataFrame([
            _poll(2018, "2018-10-01", "PollsterSmall", 47.0, 40.0),
            _poll(2018, "2018-10-05", "PollsterSmall", 47.0, 40.0),
            _poll(2018, "2018-10-01", "PollsterBig",   47.0, 40.0),
            _poll(2018, "2018-10-05", "PollsterBig",   47.0, 40.0),
            _poll(2018, "2018-10-10", "PollsterBig",   47.0, 40.0),
        ])
        effects = estimate_house_effects(polls=df, min_polls=3)
        assert "PollsterSmall" not in effects
        assert "PollsterBig" in effects


# ---------------------------------------------------------------------------
# time_decay_weights
# ---------------------------------------------------------------------------

class TestTimeDecayWeights:
    def test_time_decay_weights_recent_heavier(self):
        end_dates = pd.Series(["2018-10-30", "2018-09-07"])  # 7 days before, 60 days before
        weights = time_decay_weights(
            end_dates=end_dates,
            election_date=_ELECTION_2018,
            halflife_days=28.0,
        )
        assert weights[0] > weights[1]

    def test_time_decay_weights_shape(self):
        end_dates = pd.Series(["2018-10-01", "2018-10-10", "2018-10-20"])
        weights = time_decay_weights(
            end_dates=end_dates,
            election_date=_ELECTION_2018,
        )
        assert weights.shape == (3,)


# ---------------------------------------------------------------------------
# aggregate_year
# ---------------------------------------------------------------------------

class TestAggregateYear:
    def test_aggregate_year_filters_post_election_polls(self, polls_df: pd.DataFrame):
        """A poll dated after election day must not affect the aggregate."""
        post_election_poll = _poll(
            year=2018,
            end_date=(_ELECTION_2018 + datetime.timedelta(days=3)).isoformat(),
            pollster="PollsterB",
            dem=99.0,  # extreme value — would shift result dramatically if included
            rep=1.0,
        )
        polls_with_post = pd.concat(
            [polls_df, pd.DataFrame([post_election_poll])],
            ignore_index=True,
        )
        margin_clean, _ = aggregate_year(polls=polls_df, year=2018)
        margin_post, _ = aggregate_year(polls=polls_with_post, year=2018)
        # Post-election poll should be excluded, so results must be equal
        assert margin_clean == pytest.approx(margin_post, abs=0.01)

    def test_aggregate_year_output_types(self, polls_df: pd.DataFrame):
        margin, sigma = aggregate_year(polls=polls_df, year=2018)
        assert isinstance(margin, float)
        assert isinstance(sigma, float)

    def test_aggregate_year_sigma_minimum(self, polls_df: pd.DataFrame):
        _, sigma = aggregate_year(polls=polls_df, year=2018)
        assert sigma >= 1.0

    def test_aggregate_year_weighted_mean_direction(self):
        """When all polls show D+5 the weighted mean should be near 5.0."""
        election_date = ELECTION_DATES[2018]
        rows = [
            _poll(2018, (election_date - datetime.timedelta(days=d)).isoformat(), "P", 47.5, 42.5)
            for d in [7, 14, 21, 28, 35]
        ]
        df = pd.DataFrame(rows)
        margin, _ = aggregate_year(polls=df, year=2018)
        assert margin == pytest.approx(5.0, rel=0.1)


# ---------------------------------------------------------------------------
# build_poll_aggregates
# ---------------------------------------------------------------------------

class TestBuildPollAggregates:
    def test_build_poll_aggregates_columns(self, polls_df: pd.DataFrame):
        result = build_poll_aggregates(polls=polls_df)
        assert set(result.columns) == {"year", "generic_ballot", "generic_ballot_sigma"}

    def test_build_poll_aggregates_sorted_by_year(self, polls_df: pd.DataFrame):
        result = build_poll_aggregates(polls=polls_df)
        years = result["year"].tolist()
        assert years == sorted(years)
        # strictly increasing
        assert years == list(dict.fromkeys(years))

    def test_build_poll_aggregates_skips_missing_year(self, polls_df: pd.DataFrame):
        """Requesting a year with no polls should be silently skipped."""
        result = build_poll_aggregates(polls=polls_df, years=[2018, 2024])
        assert 2018 in result["year"].values
        assert 2024 not in result["year"].values


# ---------------------------------------------------------------------------
# house_effect_correction
# ---------------------------------------------------------------------------

class TestHouseEffectCorrection:
    def test_house_effect_correction_moves_mean(self):
        """When one Dem-leaning pollster dominates, correction lowers the raw mean."""
        election_date = ELECTION_DATES[2018]
        # 8 polls from Dem-leaning pollster (+3 each), 1 from neutral
        rows = [
            _poll(2018, (election_date - datetime.timedelta(days=d)).isoformat(), "BiasedPoll", 50.0, 42.0)
            for d in range(7, 80, 9)  # 8 polls
        ] + [
            _poll(2018, (election_date - datetime.timedelta(days=10)).isoformat(), "NeutralPoll", 47.0, 42.0),
            _poll(2018, (election_date - datetime.timedelta(days=20)).isoformat(), "NeutralPoll", 47.0, 42.0),
            _poll(2018, (election_date - datetime.timedelta(days=30)).isoformat(), "NeutralPoll", 47.0, 42.0),
        ]
        df = pd.DataFrame(rows)
        # Naive unweighted mean is dominated by BiasedPoll's D+8
        naive_mean = float(df[df["year"] == 2018]["margin"].mean())
        corrected_margin, _ = aggregate_year(polls=df, year=2018)
        # House-effect correction should reduce the margin (bias is positive → subtracted)
        assert corrected_margin < naive_mean


# ---------------------------------------------------------------------------
# normalize_poll_columns — topline format
# ---------------------------------------------------------------------------

class TestNormalizePollColumnsToplineFormat:
    def test_topline_modeldate_maps_to_end_date(self):
        df = pd.DataFrame([{
            "modeldate": "11/6/2018",
            "subgroup": "All polls",
            "dem_estimate": 48.5,
            "rep_estimate": 41.5,
        }])
        result = normalize_poll_columns(df)
        assert result["end_date"].iloc[0] == "11/6/2018"
        assert result["pollster"].iloc[0] == "All polls"

    def test_topline_year_derived_from_modeldate(self):
        df = pd.DataFrame([{
            "modeldate": "11/6/2018",
            "subgroup": "All polls",
            "dem_estimate": 48.5,
            "rep_estimate": 41.5,
        }])
        result = normalize_poll_columns(df)
        assert result["year"].iloc[0] == 2018

    def test_topline_dem_estimate_maps_to_dem(self):
        df = pd.DataFrame([{
            "modeldate": "11/6/2018",
            "subgroup": "All polls",
            "dem_estimate": 48.5,
            "rep_estimate": 41.5,
        }])
        result = normalize_poll_columns(df)
        assert result["dem"].iloc[0] == pytest.approx(48.5)


# ---------------------------------------------------------------------------
# normalize_poll_columns — edge cases
# ---------------------------------------------------------------------------

class TestNormalizePollColumnsEdgeCases:
    def test_cycle_column_used_for_year(self):
        df = pd.DataFrame([{
            "cycle": 2018,
            "end_date": "2018-10-01",
            "pollster": "PollsterA",
            "dem": 47.0,
            "rep": 40.0,
        }])
        result = normalize_poll_columns(df)
        assert result["year"].iloc[0] == 2018

    def test_no_recognized_dem_column_drops_rows(self):
        """A DataFrame with no dem/adjusted_dem/adj_dem/dem_estimate column produces an empty result."""
        df = pd.DataFrame([{
            "end_date": "2018-10-01",
            "pollster": "PollsterA",
            "unknown_dem_col": 47.0,
            "rep": 40.0,
        }])
        result = normalize_poll_columns(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# fetch_generic_ballot_polls — cache hit
# ---------------------------------------------------------------------------

class TestFetchGenericBallotPollsCacheHit:
    def test_returns_cached_csv_without_network(self, tmp_path: Path):
        cache_file = tmp_path / "polls.csv"
        expected = pd.DataFrame({
            "year": [2018, 2020],
            "end_date": ["2018-10-01", "2020-10-01"],
            "pollster": ["PollsterA", "PollsterB"],
            "dem": [47.0, 48.0],
            "rep": [40.0, 41.0],
            "margin": [7.0, 7.0],
        })
        expected.to_csv(cache_file, index=False)

        with patch("requests.get", side_effect=RuntimeError("should not reach network")):
            result = fetch_generic_ballot_polls(cache_path=cache_file)

        pd.testing.assert_frame_equal(result, expected)
