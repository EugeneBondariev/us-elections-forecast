"""
Unit tests for src.data.validators. All fixtures are synthetic hand-crafted
DataFrames — no files in data/ needed.
"""
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
