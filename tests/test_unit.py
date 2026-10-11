"""
Unit tests that run without any files in data/.
All fixtures are synthetic hand-crafted DataFrames / arrays.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. WeightedPipeline routes sample_weight correctly
# ---------------------------------------------------------------------------

def test_weighted_pipeline_accepts_sample_weight():
    from src.model import build_pipeline

    X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]}, dtype=float)
    y = pd.Series([0, 0, 1, 1, 1])
    w = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    pipe = build_pipeline()
    pipe.fit(X, y, sample_weight=w)  # must not raise
    assert pipe.predict(X).shape == (5,)


# ---------------------------------------------------------------------------
# 2. compute_sample_weights gives more weight to recent years
# ---------------------------------------------------------------------------

def test_sample_weights_recency():
    from src.model import compute_sample_weights

    years = pd.Series([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
    w = compute_sample_weights(years)
    assert w[-1] > w[0]          # most recent > oldest
    assert len(w) == len(years)
    assert (w > 0).all()


# ---------------------------------------------------------------------------
# 3. simulate_election returns correct shape and range
# ---------------------------------------------------------------------------

def test_simulate_election_shape():
    from src.model import simulate_election

    proba = np.array([0.3, 0.5, 0.7, 0.6, 0.4])
    result = simulate_election(proba, n=500, seed=0)
    assert result.shape == (500,)
    assert result.min() >= 0
    assert result.max() <= len(proba)


# ---------------------------------------------------------------------------
# 4. bayesian_simulate returns correct shape
# ---------------------------------------------------------------------------

def test_bayesian_simulate_shape():
    from src.model import bayesian_simulate

    proba = np.full(10, 0.5)
    result = bayesian_simulate(proba, n=200, seed=1)
    assert result.shape == (200,)
    assert result.min() >= 0
    assert result.max() <= 10


# ---------------------------------------------------------------------------
# 5. feature_weights returns sorted dict summing to ~100
# ---------------------------------------------------------------------------

def test_feature_weights_sum():
    from src.model import build_pipeline, feature_weights

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + np.random.randn(50) * 0.1 > 0).astype(int))
    pipe = build_pipeline()
    pipe.fit(X, y)
    fw = feature_weights(pipe, X.columns)
    assert set(fw.keys()) == {"a", "b", "c"}
    assert abs(sum(fw.values()) - 100) < 0.01
    vals = list(fw.values())
    assert vals == sorted(vals, reverse=True)  # sorted descending


# ---------------------------------------------------------------------------
# 6. bootstrap_ci returns expected keys and valid ranges
# ---------------------------------------------------------------------------

def test_bootstrap_ci_keys():
    from src.model import bootstrap_ci

    np.random.seed(0)
    n = 100
    X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
    y = pd.Series(np.random.randint(0, 2, n))
    # Use years 2000 (train) and 2002 (test) so bootstrap_ci has one valid split
    years = pd.Series([2000] * 50 + [2002] * 50)
    result = bootstrap_ci(X, y, years, n_bootstrap=10)
    expected_keys = {
        "accuracy_mean", "accuracy_low", "accuracy_high",
        "seat_error_mean", "seat_error_low", "seat_error_high",
    }
    assert set(result.keys()) == expected_keys
    assert 0 <= result["accuracy_mean"] <= 1
    assert result["accuracy_low"] <= result["accuracy_mean"] <= result["accuracy_high"]


# ---------------------------------------------------------------------------
# 7. build_dems_reps computes dem_vote_share correctly
# ---------------------------------------------------------------------------

def test_build_dems_reps_vote_share():
    from src.features import build_dems_reps
    from src.constants import Col, Val

    top = pd.DataFrame([
        {Col.YEAR: 2020, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 60},
        {Col.YEAR: 2020, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 40},
        {Col.YEAR: 2022, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 55},
        {Col.YEAR: 2022, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 45},
    ])
    result = build_dems_reps(top)
    row_2020 = result[result[Col.YEAR] == 2020].iloc[0]
    assert abs(row_2020[Col.DEM_VOTE_SHARE] - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# 8. walk_forward_cv produces reasonable output on toy data
# ---------------------------------------------------------------------------

def test_walk_forward_cv_toy():
    from src.model import walk_forward_cv

    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int))
    years = pd.Series([2000] * 50 + [2002] * 50 + [2004] * 50 + [2006] * 50)
    result = walk_forward_cv(X, y, years, start_test_year=2002)
    assert len(result) == 3  # 2002, 2004, 2006
    assert "district_accuracy" in result.columns
    assert (result["district_accuracy"] >= 0).all()
    assert (result["district_accuracy"] <= 1).all()


# ---------------------------------------------------------------------------
# 9. stacking ensemble trains and predicts
# ---------------------------------------------------------------------------

def test_stacking_ensemble_trains():
    from src.model import build_stacking_ensemble

    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    model = build_stacking_ensemble()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
