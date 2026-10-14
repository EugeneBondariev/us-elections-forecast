"""Round-trip tests for src/database.py forecast persistence. Uses a temp SQLite
file so no real data files or shared state are needed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.database import load_latest_forecast, save_forecast

_DISTRICTS = pd.DataFrame([
    {"label": "PA-7", "state_po": "PA", "district": 7, "proba_dem": 0.55},
    {"label": "AZ-6", "state_po": "AZ", "district": 6, "proba_dem": 0.42},
])

_RESULT = {
    "n_districts": 2,
    "seats_sim": np.array([210, 215, 220]),
    "districts": _DISTRICTS,
}


def test_load_returns_none_when_db_absent(tmp_path):
    assert load_latest_forecast(tmp_path / "missing.db") is None


def test_save_then_load_round_trip(tmp_path):
    db_path = tmp_path / "elections.db"
    run_id = save_forecast(_RESULT, db_path)
    assert isinstance(run_id, int)

    loaded = load_latest_forecast(db_path)
    assert loaded is not None
    assert loaded["n_districts"] == 2
    np.testing.assert_array_equal(loaded["seats_sim"], _RESULT["seats_sim"])
    pd.testing.assert_frame_equal(
        loaded["districts"].reset_index(drop=True),
        _DISTRICTS.reset_index(drop=True),
        check_dtype=False,
    )


def test_load_returns_most_recent_of_multiple_saves(tmp_path):
    db_path = tmp_path / "elections.db"
    save_forecast(_RESULT, db_path)
    newer = {**_RESULT, "n_districts": 99}
    save_forecast(newer, db_path)

    loaded = load_latest_forecast(db_path)
    assert loaded["n_districts"] == 99
