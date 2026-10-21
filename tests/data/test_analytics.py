"""Unit tests for src/data/analytics.py. No network access, no disk I/O."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.analytics import (
    build_analytics_db,
    build_district_analytics,
    compute_swing_volatility,
    compute_win_streak,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_YEARS = list(range(1998, 2018, 2))  # 1998, 2000, ..., 2016 — 10 cycles


@pytest.fixture
def dems_reps() -> pd.DataFrame:
    """3 districts × 10 years synthetic data.

    District 1 (CA): always Democrat (0.60–0.70)
    District 2 (CA): always Republican (0.30–0.40)
    District 3 (CA): alternating (0.52, 0.48, 0.52, …)
    """
    rows = []
    for i, year in enumerate(_YEARS):
        # District 1: firmly Dem, nearly flat — low volatility
        rows.append(
            {
                "year": year,
                "state_po": "CA",
                "district": 1,
                "dem_vote_share": 0.65,
            }
        )
        # District 2: firmly Rep, nearly flat
        rows.append(
            {
                "year": year,
                "state_po": "CA",
                "district": 2,
                "dem_vote_share": 0.35,
            }
        )
        # District 3: alternating with wide swing — high volatility
        rows.append(
            {
                "year": year,
                "state_po": "CA",
                "district": 3,
                "dem_vote_share": 0.55 if i % 2 == 0 else 0.45,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestBuildAnalyticsDb
# ---------------------------------------------------------------------------


class TestBuildAnalyticsDb:
    def test_creates_results_table(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        (count,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        assert count == len(dems_reps)

    def test_columns_present(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        info = conn.execute("PRAGMA table_info(results)").fetchall()
        col_names = {row[1] for row in info}
        assert {"year", "state_po", "district", "dem_vote_share"}.issubset(col_names)


# ---------------------------------------------------------------------------
# TestComputeWinStreak
# ---------------------------------------------------------------------------


class TestComputeWinStreak:
    def test_always_dem_gets_streak(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        df = compute_win_streak(conn)
        # After 5+ Dem wins district 1 should reach the max streak of 4
        late_d1 = df[(df["district"] == 1) & (df["year"] >= 2008)]
        assert (late_d1["win_streak"] == 4).any()

    def test_first_cycle_streak_zero(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        df = compute_win_streak(conn)
        first_year = df[df["year"] == 1998]
        assert (first_year["win_streak"] == 0).all()

    def test_alternating_district_streak_one(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        df = compute_win_streak(conn)
        # District 3 alternates D/R so p1 != p2 — streak should be 1 in later years
        # (after the second cycle, when p1 is available but p1 != p2)
        d3_later = df[(df["district"] == 3) & (df["year"] >= 2002)]
        assert (d3_later["win_streak"] == 1).any()


# ---------------------------------------------------------------------------
# TestComputeSwingVolatility
# ---------------------------------------------------------------------------


class TestComputeSwingVolatility:
    def test_stable_district_low_volatility(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        df = compute_swing_volatility(conn)
        # In years where 4 prior cycles are available (2006+)
        late = df[df["year"] >= 2006]
        d1_vol = late[late["district"] == 1]["swing_volatility"].dropna().mean()
        d3_vol = late[late["district"] == 3]["swing_volatility"].dropna().mean()
        # District 1 varies by at most 0.02 per step; district 3 alternates by 0.04 each cycle
        assert d1_vol < d3_vol

    def test_null_before_four_cycles(self, dems_reps: pd.DataFrame) -> None:
        conn = build_analytics_db(dems_reps)
        df = compute_swing_volatility(conn)
        # Years 1998–2004 have fewer than 4 prior cycles: volatility must be NULL
        early = df[df["year"].isin([1998, 2000, 2002, 2004])]
        assert early["swing_volatility"].isna().all()


# ---------------------------------------------------------------------------
# TestBuildDistrictAnalytics
# ---------------------------------------------------------------------------


class TestBuildDistrictAnalytics:
    def test_output_shape(self, dems_reps: pd.DataFrame) -> None:
        result = build_district_analytics(dems_reps)
        assert len(result) == len(dems_reps)
        assert {"year", "state_po", "district", "win_streak", "swing_volatility"}.issubset(
            set(result.columns)
        )

    def test_win_streak_dtype(self, dems_reps: pd.DataFrame) -> None:
        result = build_district_analytics(dems_reps)
        assert pd.api.types.is_integer_dtype(result["win_streak"])

    def test_swing_vol_has_nulls_early(self, dems_reps: pd.DataFrame) -> None:
        result = build_district_analytics(dems_reps)
        assert result["swing_volatility"].isna().any()

    def test_year_and_district_int(self, dems_reps: pd.DataFrame) -> None:
        result = build_district_analytics(dems_reps)
        assert pd.api.types.is_integer_dtype(result["year"])
        assert pd.api.types.is_integer_dtype(result["district"])
