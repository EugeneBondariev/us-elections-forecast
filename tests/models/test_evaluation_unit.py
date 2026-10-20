"""Unit tests for src.models.evaluation — covers cross_validate,
walk_forward_feature_importance, and walk_forward_tossup_skew.
All fixtures are synthetic; no real data files required.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_fixture() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, years) with 4 groups × 8 rows each (years 1994-2000).

    Enough history so 2000 is a valid walk-forward test year.
    """
    rng = np.random.RandomState(0)
    group_years = [1994, 1996, 1998, 2000]
    n_per_year = 8
    rows_X: list[pd.DataFrame] = []
    rows_y: list[int] = []
    rows_years: list[int] = []
    for yr in group_years:
        rows_X.append(pd.DataFrame(
            rng.randn(n_per_year, 4),
            columns=["a", "b", "c", "d"],
        ))
        rows_y.extend([i % 2 for i in range(n_per_year)])
        rows_years.extend([yr] * n_per_year)
    X = pd.concat(rows_X, ignore_index=True)
    y = pd.Series(rows_y)
    years = pd.Series(rows_years)
    return X, y, years


class TestCrossValidate:
    def test_returns_float_between_zero_and_one(self) -> None:
        from src.models.evaluation import cross_validate

        X, y, _ = _make_fixture()
        score = cross_validate(X, y, cv=2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestWalkForwardFeatureImportance:
    def test_returns_dataframe_with_year_index(self) -> None:
        from src.models.evaluation import walk_forward_feature_importance

        X, y, years = _make_fixture()
        result = walk_forward_feature_importance(X, y, years, start_test_year=2000)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == X.shape[1]
        assert 2000 in result.index

    def test_all_weights_nonnegative(self) -> None:
        from src.models.evaluation import walk_forward_feature_importance

        X, y, years = _make_fixture()
        result = walk_forward_feature_importance(X, y, years, start_test_year=2000)
        assert (result.values >= 0).all()


class TestWalkForwardTossupSkew:
    def test_no_tossups_returns_empty(self) -> None:
        """All probabilities are far from 0.5, so no district qualifies as a toss-up."""
        from src.models.evaluation import walk_forward_tossup_skew

        # Inject probabilities that are clearly 0.8 or 0.2 — none within ±0.15 of 0.5.
        fake_folds = {
            2000: (
                np.array([0.8, 0.8, 0.2, 0.2]),
                np.array([1, 1, 0, 0], dtype=float),
            )
        }
        X, y, years = _make_fixture()
        with patch("src.models.evaluation.walk_forward_probas", return_value=fake_folds):
            result = walk_forward_tossup_skew(X, y, years, tossup_threshold=0.15)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_competitive_districts_appear(self) -> None:
        """Mock walk_forward_probas to inject a fold with proba=0.5 rows; result must be non-empty."""
        from src.models.evaluation import walk_forward_tossup_skew

        fake_folds = {
            2000: (
                np.array([0.5, 0.5, 0.9, 0.1]),
                np.array([1, 0, 1, 0], dtype=float),
            )
        }
        X, y, years = _make_fixture()
        with patch("src.models.evaluation.walk_forward_probas", return_value=fake_folds):
            result = walk_forward_tossup_skew(X, y, years, tossup_threshold=0.15)
        assert len(result) >= 1

    def test_output_columns(self) -> None:
        """Result DataFrame must always carry exactly the five expected columns."""
        from src.models.evaluation import walk_forward_tossup_skew

        fake_folds = {
            2000: (
                np.array([0.5, 0.5, 0.9, 0.1]),
                np.array([1, 0, 1, 0], dtype=float),
            )
        }
        X, y, years = _make_fixture()
        with patch("src.models.evaluation.walk_forward_probas", return_value=fake_folds):
            result = walk_forward_tossup_skew(X, y, years, tossup_threshold=0.15)
        assert set(result.columns) == {"year", "n_tossup", "dem_wins", "dem_pct", "skew"}
