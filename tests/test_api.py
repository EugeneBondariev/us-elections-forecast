"""FastAPI endpoint tests — forecast is mocked so no real data files are needed."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from src.constants import Col

_MOCK_DISTRICTS = pd.DataFrame(
    [
        {
            "label": "PA-7",
            Col.STATE: "PA",
            Col.DISTRICT: 7,
            "proba_dem": 0.55,
            "district_incumbent": "Democrat",
        },
        {
            "label": "AZ-6",
            Col.STATE: "AZ",
            Col.DISTRICT: 6,
            "proba_dem": 0.42,
            "district_incumbent": "Republican",
        },
    ]
)

_MOCK_FORECAST = {
    "seats_sim": np.full(10_000, 215),
    "n_districts": 2,
    "districts": _MOCK_DISTRICTS,
}


@pytest.fixture()
def client():
    with patch("api.load_latest_forecast", return_value=_MOCK_FORECAST):
        with patch("api.run_forecast", return_value=_MOCK_FORECAST):
            with TestClient(app) as c:
                yield c


class TestHealth:
    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_cache_ready_after_startup(self, client):
        assert client.get("/health").json()["cache_ready"] is True


class TestForecast:
    def test_returns_200(self, client):
        assert client.get("/forecast").status_code == 200

    def test_required_keys_present(self, client):
        data = client.get("/forecast").json()
        assert {"median_dem_seats", "p_dem_majority", "ci_68", "ci_95", "n_districts"}.issubset(
            data
        )

    def test_n_districts_matches_mock(self, client):
        assert client.get("/forecast").json()["n_districts"] == 2

    def test_majority_probability_bounded(self, client):
        p = client.get("/forecast").json()["p_dem_majority"]
        assert 0.0 <= p <= 1.0


class TestDistricts:
    def test_returns_list(self, client):
        resp = client.get("/districts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_length_matches_mock(self, client):
        assert len(client.get("/districts").json()) == 2


class TestDistrictDetail:
    def test_known_district_200(self, client):
        resp = client.get("/districts/PA/7")
        assert resp.status_code == 200

    def test_known_district_payload(self, client):
        data = client.get("/districts/PA/7").json()
        assert data["label"] == "PA-7"
        assert data["proba_dem"] == pytest.approx(0.55, abs=1e-4)

    def test_state_case_insensitive(self, client):
        assert client.get("/districts/pa/7").status_code == 200

    def test_unknown_district_404(self, client):
        assert client.get("/districts/ZZ/99").status_code == 404


class TestRefreshAuth:
    """/refresh triggers a full retrain — must be gated once REFRESH_API_KEY is set."""

    @pytest.fixture()
    def mocked_refresh(self):
        with patch("api.run_forecast", return_value=_MOCK_FORECAST), patch("api.save_forecast"):
            yield

    def test_open_when_no_key_configured(self, client, monkeypatch, mocked_refresh):
        monkeypatch.delenv("REFRESH_API_KEY", raising=False)
        assert client.get("/refresh").status_code == 200

    def test_rejects_missing_key_when_configured(self, client, monkeypatch, mocked_refresh):
        monkeypatch.setenv("REFRESH_API_KEY", "secret123")
        assert client.get("/refresh").status_code == 401

    def test_rejects_wrong_key(self, client, monkeypatch, mocked_refresh):
        monkeypatch.setenv("REFRESH_API_KEY", "secret123")
        resp = client.get("/refresh", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_accepts_correct_key(self, client, monkeypatch, mocked_refresh):
        monkeypatch.setenv("REFRESH_API_KEY", "secret123")
        resp = client.get("/refresh", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "refreshed"
