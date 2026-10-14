"""Tests for src/narrative.py — Groq client is mocked, no real API calls."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.visualization.narrative import _build_prompt, generate_narrative

_SIM = np.array([210, 215, 218, 220, 225])
_DISTRICTS = pd.DataFrame([
    {"label": "PA-7", "proba_dem": 0.51, "district_incumbent": "DEMOCRAT", "is_open": False},
    {"label": "AZ-6", "proba_dem": 0.49, "district_incumbent": "REPUBLICAN", "is_open": True},
])


def test_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert generate_narrative(_SIM, _DISTRICTS, len(_DISTRICTS)) == ""


def test_build_prompt_includes_headline_numbers_and_districts():
    prompt = _build_prompt(_SIM, _DISTRICTS, len(_DISTRICTS))
    assert "Median Democratic seats" in prompt
    assert "PA-7" in prompt


def test_calls_groq_and_returns_text(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="The forecast favors a narrow Dem majority."))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    monkeypatch.setitem(sys.modules, "groq", mock_groq_module)

    result = generate_narrative(_SIM, _DISTRICTS, len(_DISTRICTS))

    assert result == "The forecast favors a narrow Dem majority."
    mock_groq_module.Groq.assert_called_once_with(api_key="gsk-test")


def test_returns_empty_on_api_failure(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")
    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    monkeypatch.setitem(sys.modules, "groq", mock_groq_module)

    assert generate_narrative(_SIM, _DISTRICTS, len(_DISTRICTS)) == ""
