"""
Unit tests for src.models.model that run without any files in data/.
All fixtures are synthetic hand-crafted DataFrames / arrays.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. WeightedPipeline routes sample_weight correctly
# ---------------------------------------------------------------------------

def test_weighted_pipeline_accepts_sample_weight():
    from src.models.model import build_pipeline

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
    from src.models.model import compute_sample_weights

    years = pd.Series([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
    w = compute_sample_weights(years)
    assert w[-1] > w[0]          # most recent > oldest
    assert len(w) == len(years)
    assert (w > 0).all()


# ---------------------------------------------------------------------------
# 3. simulate_election returns correct shape and range
# ---------------------------------------------------------------------------

def test_simulate_election_shape():
    from src.models.model import simulate_election

    proba = np.array([0.3, 0.5, 0.7, 0.6, 0.4])
    result = simulate_election(proba, n=500, seed=0)
    assert result.shape == (500,)
    assert result.min() >= 0
    assert result.max() <= len(proba)


# ---------------------------------------------------------------------------
# 4. bayesian_simulate returns correct shape
# ---------------------------------------------------------------------------

def test_bayesian_simulate_shape():
    from src.models.model import bayesian_simulate

    proba = np.full(10, 0.5)
    result = bayesian_simulate(proba, n=200, seed=1)
    assert result.shape == (200,)
    assert result.min() >= 0
    assert result.max() <= 10


# ---------------------------------------------------------------------------
# 5. feature_weights returns sorted dict summing to ~100
# ---------------------------------------------------------------------------

def test_feature_weights_sum():
    from src.models.model import build_pipeline, feature_weights

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
    from src.models.model import bootstrap_ci

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
# 7. walk_forward_cv produces reasonable output on toy data
# ---------------------------------------------------------------------------

def test_walk_forward_cv_toy():
    from src.models.model import walk_forward_cv

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
# 8. stacking ensemble trains and predicts
# ---------------------------------------------------------------------------

def test_stacking_ensemble_trains():
    from src.models.model import build_stacking_ensemble

    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    model = build_stacking_ensemble()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 9. feature_weights works with StackingClassifier (GBM branch)
# ---------------------------------------------------------------------------

def test_feature_weights_stacking_ensemble():
    from src.models.model import build_stacking_ensemble, feature_weights

    np.random.seed(7)
    X = pd.DataFrame(np.random.randn(80, 3), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int))
    # Tuned params (min_child_samples=112) prevent splits on 80 samples — override for this test.
    _small = {"min_child_samples": 1, "n_estimators": 50, "verbose": -1}
    with patch("src.models.model._gbm_params", return_value=_small):
        model = build_stacking_ensemble()
        model.fit(X, y)
    fw = feature_weights(model, X.columns)
    assert set(fw.keys()) == {"a", "b", "c"}
    assert abs(sum(fw.values()) - 100) < 0.01
    vals = list(fw.values())
    assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# 10. simulate_election is deterministic with a fixed seed
# ---------------------------------------------------------------------------

def test_simulate_election_deterministic():
    from src.models.model import simulate_election

    proba = np.array([0.3, 0.5, 0.7, 0.6, 0.4])
    r1 = simulate_election(proba, n=1000, seed=99)
    r2 = simulate_election(proba, n=1000, seed=99)
    np.testing.assert_array_equal(r1, r2)


def _make_walk_forward_frame(n_years=6, n_per_year=60, seed=0):
    rng = np.random.RandomState(seed)
    years = sorted(2000 + 2 * i for i in range(n_years))
    rows_X, rows_y, rows_years = [], [], []
    for yr in years:
        a = rng.randn(n_per_year)
        b = rng.randn(n_per_year)
        y_year = (a + 0.1 * rng.randn(n_per_year) > 0).astype(int)
        rows_X.append(pd.DataFrame({"a": a, "b": b}))
        rows_y.extend(y_year)
        rows_years.extend([yr] * n_per_year)
    X = pd.concat(rows_X, ignore_index=True)
    y = pd.Series(rows_y)
    years_s = pd.Series(rows_years)
    return X, y, years_s


# ---------------------------------------------------------------------------
# 11. walk_forward_probas returns one (proba, actual) pair per test year
# ---------------------------------------------------------------------------

def test_walk_forward_probas_shapes():
    from src.models.model import walk_forward_probas

    X, y, years = _make_walk_forward_frame()
    folds = walk_forward_probas(X, y, years, start_test_year=2004)
    assert set(folds.keys()) == {2004, 2006, 2008, 2010}
    for proba, actual in folds.values():
        assert proba.shape == actual.shape
        assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------------
# 12. walk_forward_calibration returns Brier score / log loss per year
# ---------------------------------------------------------------------------

def test_walk_forward_calibration_columns_and_bounds():
    from src.models.model import walk_forward_calibration

    X, y, years = _make_walk_forward_frame()
    calib = walk_forward_calibration(X, y, years, start_test_year=2004)
    assert set(calib.columns) == {"year", "brier_score", "log_loss", "n_districts"}
    assert (calib["brier_score"] >= 0).all() and (calib["brier_score"] <= 1).all()
    assert (calib["log_loss"] >= 0).all()


# ---------------------------------------------------------------------------
# 13. calibrate_national_sigma: too few years falls back to the documented default
# ---------------------------------------------------------------------------

def test_calibrate_national_sigma_falls_back_with_too_few_years():
    from src.models.model import calibrate_national_sigma

    X, y, years = _make_walk_forward_frame(n_years=3)
    result = calibrate_national_sigma(X, y, years, start_test_year=2004, min_years=5)
    assert result["sigma_national"] == 0.25
    assert np.isnan(result["target_std"])


# ---------------------------------------------------------------------------
# 14. calibrate_national_sigma: with enough years, returns a positive, bounded sigma
# ---------------------------------------------------------------------------

def test_calibrate_national_sigma_returns_bounded_value_with_enough_years():
    from src.models.model import calibrate_national_sigma

    X, y, years = _make_walk_forward_frame(n_years=8)
    result = calibrate_national_sigma(X, y, years, start_test_year=2002, n=200, min_years=3)
    assert 0.0 < result["sigma_national"] < 1.0
    assert result["sigma_district"] == 0.15
    assert result["n_years"] > 0


# ---------------------------------------------------------------------------
# 15. _robust_std: a single extreme outlier shouldn't dominate it the way
# plain std does - this is what real 2000-2024 walk-forward data actually hit:
# a -51.7 seat miss in 2002 (post-redistricting, model trained on too little
# history) inflated plain std enough to pin calibrate_national_sigma at its
# search ceiling every time, confirmed by running it against real data.
# ---------------------------------------------------------------------------

def test_robust_std_resists_single_outlier():
    from src.models.model import _robust_std

    typical = [2.7, -16.4, 15.9, -9.9, -22.4, 11.8, -14.8, -1.1, 32.8, 2.1, 36.9, 9.2]
    with_outlier = typical + [-51.7]

    plain_std_typical = np.std(typical)
    plain_std_with_outlier = np.std(with_outlier)
    robust_typical = _robust_std(typical)
    robust_with_outlier = _robust_std(with_outlier)

    # Plain std jumps a lot when the outlier is added; the robust version barely moves.
    assert plain_std_with_outlier > plain_std_typical * 1.2
    assert robust_with_outlier < robust_typical * 1.2


def test_robust_std_matches_plain_std_for_symmetric_normal_data():
    """Sanity check the 1.4826 scaling constant: for data with no outliers, the
    two measures should be in the same ballpark, not wildly different."""
    from src.models.model import _robust_std

    rng = np.random.RandomState(0)
    values = rng.normal(0, 10, 500)
    assert abs(_robust_std(values) - np.std(values)) < 1.5


# ---------------------------------------------------------------------------
# 16. load_simulation_params: defaults when no calibrated file exists, else reads it
# ---------------------------------------------------------------------------

def test_load_simulation_params_defaults_when_file_absent(tmp_path, monkeypatch):
    from src.models.model import load_simulation_params
    from src.paths import DataFile

    monkeypatch.setattr(DataFile, "CALIBRATED_SIGMA", tmp_path / "nonexistent.json")
    result = load_simulation_params()
    assert result == {"sigma_national": 0.25, "sigma_district": 0.15, "blend_weight": 0.5, "concentration": 15.0}


def test_load_simulation_params_reads_calibrated_file(tmp_path, monkeypatch):
    import json

    from src.models.model import load_simulation_params
    from src.paths import DataFile

    path = tmp_path / "calibrated_sigma.json"
    path.write_text(json.dumps({"sigma_national": 0.4, "sigma_district": 0.15, "blend_weight": 0.75, "concentration": 30.0}))
    monkeypatch.setattr(DataFile, "CALIBRATED_SIGMA", path)

    result = load_simulation_params()
    assert result == {"sigma_national": 0.4, "sigma_district": 0.15, "blend_weight": 0.75, "concentration": 30.0}
