"""Unit tests for src.models.bayesian — no real election data required.

All fixtures use synthetic data. PyMC is skipped at module level if not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pymc = pytest.importorskip("pymc")

from src.models.bayesian import BayesianElectionModel, make_district_labels  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Module-scoped fixture: tiny fitted model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    rng = np.random.default_rng(42)
    n_obs = 80
    n_features = 3
    n_districts = 4

    X = pd.DataFrame(
        rng.standard_normal(size=(n_obs, n_features)),
        columns=["f0", "f1", "f2"],
    )
    y = pd.Series(rng.integers(low=0, high=2, size=n_obs).astype(int))

    # 4 districts: CA-1 through CA-4, 20 obs each
    states = np.repeat(["CA"] * n_districts, n_obs // n_districts)
    districts = np.repeat(np.arange(1, n_districts + 1), n_obs // n_districts)
    district_ids = pd.DataFrame({"state_po": states, "district": districts})

    model = BayesianElectionModel(
        n_draws=100, n_tune=100, n_chains=1, random_seed=42
    )
    model.fit(X=X, y=y, district_ids=district_ids)
    return model, X, y, district_ids


# ---------------------------------------------------------------------------
# 1. fit() returns self
# ---------------------------------------------------------------------------

def test_fit_returns_self(tiny_model):
    model, _X, _y, _district_ids = tiny_model
    assert isinstance(model, BayesianElectionModel)


# ---------------------------------------------------------------------------
# 2. predict_proba shape
# ---------------------------------------------------------------------------

def test_predict_proba_shape(tiny_model):
    model, X, _y, district_ids = tiny_model
    proba = model.predict_proba(X=X, district_ids=district_ids)
    assert proba.shape == (80,)


# ---------------------------------------------------------------------------
# 3. predict_proba values in [0, 1]
# ---------------------------------------------------------------------------

def test_predict_proba_in_unit_interval(tiny_model):
    model, X, _y, district_ids = tiny_model
    proba = model.predict_proba(X=X, district_ids=district_ids)
    assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------------
# 4. credible_interval: lo ≤ hi everywhere
# ---------------------------------------------------------------------------

def test_credible_interval_ordered(tiny_model):
    model, X, _, district_ids = tiny_model
    lo, hi = model.credible_interval(X=X, district_ids=district_ids)
    assert (lo <= hi).all()


# ---------------------------------------------------------------------------
# 5. credible_interval contains the mean
# ---------------------------------------------------------------------------

def test_credible_interval_contains_mean(tiny_model):
    model, X, _, district_ids = tiny_model
    lo, hi = model.credible_interval(X=X, district_ids=district_ids)
    mean = model.predict_proba(X=X, district_ids=district_ids)
    assert (lo <= mean).all()
    assert (mean <= hi).all()


# ---------------------------------------------------------------------------
# 6. diagnostics() returns the correct keys
# ---------------------------------------------------------------------------

def test_diagnostics_keys(tiny_model):
    model, _, __, district_ids = tiny_model
    result = model.diagnostics()
    expected = {"r_hat_max", "r_hat_mean", "ess_bulk_min", "ess_tail_min", "n_divergences"}
    assert set(result.keys()) == expected


# ---------------------------------------------------------------------------
# 7. n_divergences is a plain Python int, not a numpy integer
# ---------------------------------------------------------------------------

def test_diagnostics_n_divergences_int(tiny_model):
    model, _, __, district_ids = tiny_model
    result = model.diagnostics()
    assert type(result["n_divergences"]) is int


# ---------------------------------------------------------------------------
# 8. summary() returns a DataFrame
# ---------------------------------------------------------------------------

def test_summary_is_dataframe(tiny_model):
    model, _, __, ___ = tiny_model
    assert isinstance(model.summary(), pd.DataFrame)


# ---------------------------------------------------------------------------
# 9. Unknown district does not raise; returns value in [0, 1]
# ---------------------------------------------------------------------------

def test_unknown_district_handled(tiny_model):
    model, X, _, __ = tiny_model
    X_single = X.iloc[:1]
    unknown = pd.DataFrame({"state_po": ["ZZ"], "district": [99]})
    proba = model.predict_proba(X=X_single, district_ids=unknown)
    assert proba.shape == (1,)
    assert 0.0 <= float(proba[0]) <= 1.0


# ---------------------------------------------------------------------------
# 10. save / load round-trip produces identical predictions
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tiny_model, tmp_path):
    model, X, _, district_ids = tiny_model
    save_path = tmp_path / "bayes"
    model.save(path=save_path)

    loaded = BayesianElectionModel.load(path=save_path)
    proba_orig = model.predict_proba(X=X, district_ids=district_ids)
    proba_loaded = loaded.predict_proba(X=X, district_ids=district_ids)
    np.testing.assert_allclose(proba_orig, proba_loaded, rtol=1e-5)


# ---------------------------------------------------------------------------
# 11. make_district_labels shape and columns (standalone — no fixture)
# ---------------------------------------------------------------------------

def test_make_district_labels_shape():
    winners = pd.DataFrame({
        "state_po": ["CA", "TX", "NY"],
        "district": [1, 5, 3],
    })
    idx = pd.Index([0, 2])
    result = make_district_labels(winners=winners, X_index=idx)
    assert result.shape == (2, 2)
    assert list(result.columns) == ["state_po", "district"]
    assert list(result["state_po"]) == ["CA", "NY"]
    assert list(result["district"]) == [1, 3]


# ---------------------------------------------------------------------------
# 12. _district_map has exactly 4 entries after fit
# ---------------------------------------------------------------------------

def test_district_map_populated(tiny_model):
    model, _, __, ___ = tiny_model
    assert len(model._district_map) == 4


# ---------------------------------------------------------------------------
# 13. credible_interval with alpha=0.1 gives strictly positive width
# ---------------------------------------------------------------------------

def test_partial_pooling_intervals_nonempty(tiny_model):
    model, X, _, district_ids = tiny_model
    lo, hi = model.credible_interval(X=X, district_ids=district_ids, alpha=0.1)
    assert (hi - lo > 0).all()
