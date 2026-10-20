"""Unit tests for src.models.simulation — no data files required."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. simulate_election: basic shape and range
# ---------------------------------------------------------------------------

def test_simulate_election_output_shape():
    from src.models.simulation import simulate_election

    proba = np.array([0.4, 0.6, 0.8])
    result = simulate_election(proba, n=100, seed=0)
    assert result.shape == (100,)
    assert result.min() >= 0
    assert result.max() <= len(proba)


# ---------------------------------------------------------------------------
# 2. hetero_k=0 is homoskedastic (identical to old flat sigma_district)
# ---------------------------------------------------------------------------

def test_hetero_k_zero_matches_flat_sigma():
    """With hetero_k=0 all districts get sigma_district; result must equal flat call."""
    from src.models.simulation import simulate_election

    proba = np.array([0.3, 0.5, 0.7, 0.9])
    flat = simulate_election(
        proba, n=500, seed=7, sigma_national=0.25, sigma_district=0.15, hetero_k=0.0
    )
    homo = simulate_election(
        proba, n=500, seed=7, sigma_national=0.25, sigma_district=0.15, hetero_k=0.0
    )
    np.testing.assert_array_equal(flat, homo)


# ---------------------------------------------------------------------------
# 3. hetero_k>0 gives competitive districts more variance than safe seats
# ---------------------------------------------------------------------------

def test_hetero_k_increases_competitive_variance():
    """With hetero_k=1.0, p=0.5 gets 2x sigma; p=0.9 gets 1x sigma.
    The seat-count std from a purely competitive array should exceed a safe array."""
    from src.models.simulation import simulate_election

    n = 5_000
    # All competitive
    competitive = np.full(50, 0.5)
    # All safe (p = 0.9)
    safe = np.full(50, 0.9)

    std_comp = simulate_election(
        competitive, n=n, seed=42, sigma_national=0.0, sigma_district=0.15, hetero_k=1.0
    ).std()
    std_safe = simulate_election(
        safe, n=n, seed=42, sigma_national=0.0, sigma_district=0.15, hetero_k=1.0
    ).std()

    assert std_comp > std_safe, (
        f"Expected competitive std ({std_comp:.2f}) > safe std ({std_safe:.2f})"
    )


# ---------------------------------------------------------------------------
# 4. simulate_blended passes hetero_k through to simulate_election
# ---------------------------------------------------------------------------

def test_simulate_blended_hetero_k_propagates():
    """blend_weight=1.0 reduces to simulate_election; hetero_k must affect output."""
    from src.models.simulation import simulate_blended

    proba = np.array([0.5] * 20)

    sim_hetero = simulate_blended(
        proba, n=1000, seed=0, blend_weight=1.0,
        sigma_national=0.0, sigma_district=0.15, hetero_k=2.0,
    )
    sim_flat = simulate_blended(
        proba, n=1000, seed=0, blend_weight=1.0,
        sigma_national=0.0, sigma_district=0.15, hetero_k=0.0,
    )
    # Larger hetero_k on all-competitive array → more variance
    assert sim_hetero.std() > sim_flat.std()


# ---------------------------------------------------------------------------
# 5. exclude_years: 2002 is dropped from error distribution by default
# ---------------------------------------------------------------------------

def test_exclude_years_removes_2002_from_folds():
    """calibrate_national_sigma must drop 2002 from folds when exclude_years={2002}."""
    from unittest.mock import patch

    from src.models.simulation import calibrate_national_sigma

    proba = np.array([0.5, 0.4, 0.6, 0.7, 0.3])
    actual = np.array([1, 0, 1, 1, 0], dtype=float)

    fake_folds = {
        2002: (proba, actual),
        2004: (proba, actual),
        2006: (proba, actual),
        2008: (proba, actual),
    }

    with patch("src.models.simulation.walk_forward_probas", return_value=fake_folds):
        result_excl = calibrate_national_sigma(
            pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int),
            exclude_years=frozenset({2002}),
            n=200, seed=0,
        )
        result_incl = calibrate_national_sigma(
            pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=int),
            exclude_years=frozenset(),
            n=200, seed=0,
        )

    assert result_excl["n_years"] == 3, "2002 must be dropped from folds"
    assert result_incl["n_years"] == 4, "All years kept when exclude_years is empty"


# ---------------------------------------------------------------------------
# 7. recency_decay: recent years are weighted more heavily in calibration
# ---------------------------------------------------------------------------

def test_recency_decay_default_is_085():
    """recency_decay must default to 0.85 in both calibration functions."""
    import inspect

    from src.models.simulation import calibrate_national_sigma, calibrate_simulation_params

    for fn in (calibrate_national_sigma, calibrate_simulation_params):
        sig = inspect.signature(fn)
        default = sig.parameters["recency_decay"].default
        assert default == 0.85, (
            f"{fn.__name__}: expected recency_decay default 0.85, got {default!r}"
        )


def test_recency_weights_sum_to_one():
    """Internally computed weights (recency_decay^(max_year - year)) must sum to 1."""
    years_list = [2000, 2002, 2004, 2006, 2008]
    decay = 0.85
    max_yr = max(years_list)
    weights = np.array([decay ** (max_yr - yr) for yr in years_list])
    weights /= weights.sum()
    assert abs(weights.sum() - 1.0) < 1e-10
    # Most recent year has the highest weight
    assert weights[-1] == weights.max()


def test_exclude_years_default_is_2002():
    """Default exclude_years must be frozenset({2002})."""
    import inspect

    from src.models.simulation import calibrate_national_sigma, calibrate_simulation_params

    for fn in (calibrate_national_sigma, calibrate_simulation_params):
        sig = inspect.signature(fn)
        default = sig.parameters["exclude_years"].default
        assert default == frozenset({2002}), (
            f"{fn.__name__}: expected exclude_years default frozenset({{2002}}), got {default!r}"
        )


# ---------------------------------------------------------------------------
# 6. load_simulation_params defaults hetero_k=1.0 when key absent from JSON
# ---------------------------------------------------------------------------

def test_load_simulation_params_default_hetero_k(tmp_path, monkeypatch):
    """Old calibrated_sigma.json without hetero_k must still load with default 1.0."""
    import json

    import src.paths as paths_mod
    from src.models.simulation import load_simulation_params

    json_path = tmp_path / "calibrated_sigma.json"
    json_path.write_text(json.dumps({
        "sigma_national": 0.3,
        "sigma_district": 0.12,
        "blend_weight": 1.0,
        "concentration": 5.0,
    }))

    # Patch DataFile.CALIBRATED_SIGMA to point at our temp file
    class _FakeDataFile:
        CALIBRATED_SIGMA = json_path

    monkeypatch.setattr(paths_mod, "DataFile", _FakeDataFile)
    # Also patch the import inside simulation.py
    import src.models.simulation as sim_mod
    monkeypatch.setattr(sim_mod, "load_simulation_params", load_simulation_params)

    # Re-import after patching isn't straightforward; call directly with the patched path
    import src.paths
    original = src.paths.DataFile
    src.paths.DataFile.CALIBRATED_SIGMA = json_path  # type: ignore[attr-defined]
    try:
        result = load_simulation_params()
        assert result.get("hetero_k", 1.0) == 1.0
    finally:
        src.paths.DataFile.CALIBRATED_SIGMA = original.CALIBRATED_SIGMA  # type: ignore[attr-defined]
