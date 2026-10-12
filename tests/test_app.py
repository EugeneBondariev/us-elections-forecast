"""Smoke tests for app.py — streamlit is stubbed so tests run outside a Streamlit context."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Replace streamlit with a lightweight mock BEFORE app.py is imported.
# cache_data is a passthrough so `load` stays a real callable.
# tabs/columns return correctly-sized lists so tuple unpacking in app.py succeeds.
_st_mock = MagicMock()
_st_mock.cache_data = lambda f: f
_st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
_st_mock.columns.side_effect = lambda spec: [MagicMock() for _ in (range(spec) if isinstance(spec, int) else spec)]
# Empty selection prevents the district detail panel from executing at import time.
_mock_event = MagicMock()
_mock_event.selection.rows = []
_st_mock.dataframe.return_value = _mock_event
sys.modules["streamlit"] = _st_mock

from src.constants import Col, DEM_MAJORITY_SEATS  # noqa: E402

_DISTRICTS = pd.DataFrame([
    {"label": "PA-7", Col.STATE: "PA", Col.DISTRICT: 7, "proba_dem": 0.60,
     "district_incumbent": "DEMOCRAT", "is_open": False, "dem_lean": 0.57},
    {"label": "AZ-6", Col.STATE: "AZ", Col.DISTRICT: 6, "proba_dem": 0.48,
     "district_incumbent": "REPUBLICAN", "is_open": True,  "dem_lean": 0.46},
    {"label": "TX-3", Col.STATE: "TX", Col.DISTRICT: 3, "proba_dem": 0.20,
     "district_incumbent": "REPUBLICAN", "is_open": False, "dem_lean": 0.22},
])

_FORECAST = {
    "seats_sim": np.array([210, 215, 218, 216, 220]),
    "n_districts": 3,
    "districts": _DISTRICTS,
}

# Import app with run_forecast mocked so module-level `data = load()` does not hit disk.
sys.modules.pop("app", None)
with patch("src.forecast.run_forecast", return_value=_FORECAST):
    import app  # noqa: E402


class TestAppImport:
    def test_median_dem_is_int(self):
        assert isinstance(app.median_dem, int)

    def test_p_majority_bounded(self):
        assert 0.0 <= app.p_majority <= 100.0

    def test_n_competitive_non_negative(self):
        assert app.n_competitive >= 0

    def test_n_districts_matches_mock(self):
        assert app.n_districts == 3

    def test_majority_vline_uses_constant(self):
        # DEM_MAJORITY_SEATS - 0.5 must have been passed; check the mock recorded it
        calls = str(_st_mock.mock_calls)
        assert str(DEM_MAJORITY_SEATS - 0.5) in calls


class TestLoadFunction:
    def test_returns_required_keys(self):
        with patch("app.run_forecast", return_value=_FORECAST):
            result = app.load()
        assert {"seats_sim", "districts", "n_districts"}.issubset(result)

    def test_n_districts_value(self):
        with patch("app.run_forecast", return_value=_FORECAST):
            assert app.load()["n_districts"] == 3
