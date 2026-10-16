"""Smoke tests for app.py — streamlit is stubbed so tests run outside a Streamlit context."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.constants import DEM_MAJORITY_SEATS, Col


def _make_column_mock() -> MagicMock:
    # A bare MagicMock's .selectbox()/.checkbox() return an auto-generated child
    # MagicMock, which app.py then compares against real DataFrame columns
    # (e.g. `df["state_po"] == state_sel`). Newer pandas raises on that instead of
    # just evaluating to all-False, so give these controls realistic no-op defaults
    # ("All" / unchecked) matching their actual Streamlit widget defaults.
    col = MagicMock()
    col.selectbox.return_value = "All"
    col.checkbox.return_value = False
    return col


_st_mock = MagicMock()
_st_mock.cache_data = lambda f: f
_st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
_st_mock.columns.side_effect = lambda spec: [
    _make_column_mock() for _ in (range(spec) if isinstance(spec, int) else spec)
]
# Empty string keeps the Ask-tab `if query:` branch untaken (real Groq API is not called).
_st_mock.text_input.return_value = ""
# False prevents the tab_tools `if st.button(...):` branch from launching subprocesses.
_st_mock.button.return_value = False
# Empty selection prevents the district detail panel from executing during app import.
_mock_event = MagicMock()
_mock_event.selection.rows = []
_st_mock.dataframe.return_value = _mock_event

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


@pytest.fixture(scope="session")
def app():
    """Import app.py with all external I/O mocked.

    Deferred to test-execution time (not collection) so that the heavy import chain
    (lightgbm → shap → sklearn, ~5-8 s) doesn't inflate pytest --collect-only time.
    """
    sys.modules["streamlit"] = _st_mock
    sys.modules.pop("app", None)
    with (
        patch("src.data.database.load_latest_forecast", return_value=_FORECAST),
        patch("src.data.database.save_forecast"),
        patch("src.models.forecast.run_forecast", return_value=_FORECAST),
        patch("src.visualization.narrative.generate_narrative", return_value=""),
    ):
        import app as _app
    return _app


class TestAppImport:
    def test_median_dem_is_int(self, app) -> None:
        assert isinstance(app.median_dem, int)

    def test_p_majority_bounded(self, app) -> None:
        assert 0.0 <= app.p_majority <= 100.0

    def test_n_competitive_non_negative(self, app) -> None:
        assert app.n_competitive >= 0

    def test_n_districts_matches_mock(self, app) -> None:
        assert app.n_districts == 3

    def test_majority_vline_uses_constant(self, app) -> None:
        # DEM_MAJORITY_SEATS - 0.5 must have been passed to add_vline; verify via mock
        assert hasattr(app, "median_dem")  # confirms fixture ran and app was imported
        calls = str(_st_mock.mock_calls)
        assert str(DEM_MAJORITY_SEATS - 0.5) in calls


class TestLoadFunction:
    def test_returns_required_keys(self, app) -> None:
        with patch("app.run_forecast", return_value=_FORECAST):
            result = app.load()
        assert {"seats_sim", "districts", "n_districts"}.issubset(result)

    def test_n_districts_value(self, app) -> None:
        with patch("app.run_forecast", return_value=_FORECAST):
            assert app.load()["n_districts"] == 3
