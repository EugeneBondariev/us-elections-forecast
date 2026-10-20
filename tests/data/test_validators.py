"""
Unit tests for src.data.validators. All fixtures are synthetic hand-crafted
DataFrames — no files in data/ needed.
"""
from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# validate_feature_matrix: valid DataFrame passes without raising
# ---------------------------------------------------------------------------

def test_validate_feature_matrix_valid():
    from src.data.validators import validate_feature_matrix

    X = pd.DataFrame({
        "dem_lean":            [0.5, 0.6],
        "prev_dem_vote_share": [0.45, 0.55],
        "net_pres_approval":   [10.0, -5.0],
        "state_partisan_lean": [0.52, 0.48],
        "extra_col":           [1, 2],
    })
    result = validate_feature_matrix(X)
    assert result is X


def test_validate_feature_matrix_invalid_raises():
    """Out-of-range values must raise — hardened so a bad input can't silently train."""
    import pandera.pandas as pa

    from src.data.validators import validate_feature_matrix

    X = pd.DataFrame({
        "dem_lean":            [0.5],
        "prev_dem_vote_share": [1.8],   # > 1.0, violates schema
        "net_pres_approval":   [10.0],
    })
    with pytest.raises(pa.errors.SchemaErrors):
        validate_feature_matrix(X)


# ---------------------------------------------------------------------------
# validate_approval_data: plausibility checks on raw (pre sign-flip) ratings
# ---------------------------------------------------------------------------

def test_validate_approval_data_valid_passes():
    from src.constants import Col
    from src.data.validators import validate_approval_data

    df = pd.DataFrame({
        Col.YEAR: [1978, 1982],
        Col.PRES_APPROVAL: [55.0, 42.0],
        Col.PRES_DISAPPROVAL: [37.0, 50.0],
        Col.UNSURE: [8.0, 8.0],
    })
    result = validate_approval_data(df)
    assert result is df


def test_validate_approval_data_out_of_range_raises():
    import pandera.pandas as pa

    from src.constants import Col
    from src.data.validators import validate_approval_data

    df = pd.DataFrame({
        Col.YEAR: [1978],
        Col.PRES_APPROVAL: [155.0],   # > 100, impossible
        Col.PRES_DISAPPROVAL: [37.0],
        Col.UNSURE: [8.0],
    })
    with pytest.raises(pa.errors.SchemaErrors):
        validate_approval_data(df)


def test_validate_approval_data_bad_total_raises():
    """approve+disapprove+unsure far from 100% signals a wrong-row/column copy."""
    from src.constants import Col
    from src.data.validators import validate_approval_data

    df = pd.DataFrame({
        Col.YEAR: [1978],
        Col.PRES_APPROVAL: [55.0],
        Col.PRES_DISAPPROVAL: [50.0],   # 55+50+8 = 113, way off 100
        Col.UNSURE: [8.0],
    })
    with pytest.raises(ValueError, match="approve\\+disapprove\\+unsure"):
        validate_approval_data(df)


def test_validate_approval_data_small_rounding_slack_is_tolerated():
    from src.constants import Col
    from src.data.validators import validate_approval_data

    df = pd.DataFrame({
        Col.YEAR: [1978],
        Col.PRES_APPROVAL: [55.0],
        Col.PRES_DISAPPROVAL: [38.0],
        Col.UNSURE: [8.0],   # sums to 101 — within default tolerance of 5
    })
    result = validate_approval_data(df)
    assert result is df


# ---------------------------------------------------------------------------
# validate_approval_coverage: duplicate / missing year detection
# ---------------------------------------------------------------------------

def test_validate_approval_coverage_valid_passes():
    from src.constants import Col
    from src.data.validators import validate_approval_coverage

    df = pd.DataFrame({Col.YEAR: [1976, 1978, 1980]})
    result = validate_approval_coverage(df, required_years={1976, 1978, 1980})
    assert result is df


def test_validate_approval_coverage_missing_year_raises():
    from src.constants import Col
    from src.data.validators import validate_approval_coverage

    df = pd.DataFrame({Col.YEAR: [1976, 1980]})   # 1978 missing
    with pytest.raises(ValueError, match="missing year"):
        validate_approval_coverage(df, required_years={1976, 1978, 1980})


def test_validate_approval_coverage_duplicate_year_raises():
    from src.constants import Col
    from src.data.validators import validate_approval_coverage

    df = pd.DataFrame({Col.YEAR: [1976, 1976, 1980]})   # 1976 twice
    with pytest.raises(ValueError, match="duplicate"):
        validate_approval_coverage(df, required_years={1976, 1980})


# ---------------------------------------------------------------------------
# validate_census
# ---------------------------------------------------------------------------

class TestValidateCensus:
    def test_valid_passes(self) -> None:
        from src.constants import Col
        from src.data.validators import validate_census

        n = 450
        df = pd.DataFrame({
            Col.STATE:            ["CA"] * n,
            Col.DISTRICT:         (list(range(54)) * (n // 54 + 1))[:n],
            Col.PCT_BACHELOR_PLUS: [0.3] * n,
            Col.PCT_NON_WHITE:     [0.4] * n,
            Col.MEDIAN_INCOME:     [60_000.0] * n,
            Col.MEDIAN_AGE:        [38.0] * n,
        })
        result = validate_census(df)
        assert result is df

    def test_too_few_rows_raises_ValueError(self) -> None:
        from src.constants import Col
        from src.data.validators import validate_census

        df = pd.DataFrame({
            Col.STATE:            ["CA"] * 5,
            Col.DISTRICT:         list(range(5)),
            Col.PCT_BACHELOR_PLUS: [0.3] * 5,
            Col.PCT_NON_WHITE:     [0.4] * 5,
            Col.MEDIAN_INCOME:     [60_000.0] * 5,
            Col.MEDIAN_AGE:        [38.0] * 5,
        })
        with pytest.raises(ValueError, match="too small"):
            validate_census(df)

    def test_pct_out_of_range_raises_SchemaErrors(self) -> None:
        import pandera.pandas as pa

        from src.constants import Col
        from src.data.validators import validate_census

        n = 450
        pct_values = [0.3] * n
        pct_values[0] = 1.5   # out of [0, 1]
        df = pd.DataFrame({
            Col.STATE:            ["CA"] * n,
            Col.DISTRICT:         (list(range(54)) * (n // 54 + 1))[:n],
            Col.PCT_BACHELOR_PLUS: pct_values,
            Col.PCT_NON_WHITE:     [0.4] * n,
            Col.MEDIAN_INCOME:     [60_000.0] * n,
            Col.MEDIAN_AGE:        [38.0] * n,
        })
        with pytest.raises(pa.errors.SchemaErrors):
            validate_census(df)


# ---------------------------------------------------------------------------
# validate_candidate_quality
# ---------------------------------------------------------------------------

class TestValidateCandidateQuality:
    def test_valid_passes(self) -> None:
        from src.constants import Col
        from src.data.validators import validate_candidate_quality

        df = pd.DataFrame({
            Col.YEAR:                  [2018, 2020, 2022],
            Col.STATE:                 ["CA", "TX", "NY"],
            Col.DISTRICT:              [1, 2, 3],
            Col.CANDIDATE_QUALITY:     [0.55, 0.48, 0.60],
            Col.EXPERIENCE_CYCLES_DEM: [2, 0, 3],
            Col.EXPERIENCE_CYCLES_REP: [1, 4, 0],
        })
        result = validate_candidate_quality(df)
        assert result is df

    def test_invalid_quality_share_raises(self) -> None:
        import pandera.pandas as pa

        from src.constants import Col
        from src.data.validators import validate_candidate_quality

        df = pd.DataFrame({
            Col.YEAR:                  [2018],
            Col.STATE:                 ["CA"],
            Col.DISTRICT:              [1],
            Col.CANDIDATE_QUALITY:     [1.5],   # > 1, violates [0, 1]
            Col.EXPERIENCE_CYCLES_DEM: [2],
            Col.EXPERIENCE_CYCLES_REP: [1],
        })
        with pytest.raises(pa.errors.SchemaErrors):
            validate_candidate_quality(df)


# ---------------------------------------------------------------------------
# validate_poll_aggregates
# ---------------------------------------------------------------------------

class TestValidatePollAggregates:
    def test_valid_passes(self) -> None:
        from src.constants import Col
        from src.data.validators import validate_poll_aggregates

        df = pd.DataFrame({
            Col.YEAR:                 [2006, 2010, 2018, 2022],
            Col.GENERIC_BALLOT:       [7.0, -9.0, 7.0, -1.0],
            Col.GENERIC_BALLOT_SIGMA: [1.5, 2.0, 1.2, 1.8],
        })
        result = validate_poll_aggregates(df)
        assert result is df

    def test_out_of_range_margin_raises(self) -> None:
        import pandera.pandas as pa

        from src.constants import Col
        from src.data.validators import validate_poll_aggregates

        df = pd.DataFrame({
            Col.YEAR:                 [2018],
            Col.GENERIC_BALLOT:       [50.0],   # > 30, violates [-30, 30]
            Col.GENERIC_BALLOT_SIGMA: [1.5],
        })
        with pytest.raises(pa.errors.SchemaErrors):
            validate_poll_aggregates(df)
