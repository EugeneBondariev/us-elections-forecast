"""Tests for src/visualization/forecast_agent.py — Groq client is mocked, no real API calls."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.constants import Col
from src.visualization.forecast_agent import _run_tool, run_agent

_DISTRICTS = pd.DataFrame(
    [
        {
            "label": "PA-7",
            Col.STATE: "PA",
            Col.DISTRICT: 7,
            "proba_dem": 0.55,
            "district_incumbent": "DEMOCRAT",
            "is_open": False,
        },
        {
            "label": "AZ-6",
            Col.STATE: "AZ",
            Col.DISTRICT: 6,
            "proba_dem": 0.42,
            "district_incumbent": "REPUBLICAN",
            "is_open": True,
        },
    ]
)

_FORECAST = {
    "seats_sim": np.full(10_000, 215),
    "n_districts": 2,
    "districts": _DISTRICTS,
}


class TestRunTool:
    def test_forecast_summary_has_expected_keys(self):
        result = json.loads(_run_tool("get_forecast_summary", {}, _FORECAST))
        assert result["median_dem_seats"] == 215
        assert result["n_districts"] == 2

    def test_district_detail_known_district(self):
        result = json.loads(
            _run_tool("get_district_detail", {"state": "pa", "district": 7}, _FORECAST)
        )
        assert result["label"] == "PA-7"

    def test_district_detail_unknown_returns_error(self):
        result = json.loads(
            _run_tool("get_district_detail", {"state": "ZZ", "district": 99}, _FORECAST)
        )
        assert "error" in result

    def test_unknown_tool_returns_error(self):
        result = json.loads(_run_tool("not_a_real_tool", {}, _FORECAST))
        assert "error" in result


def _mock_tool_call(call_id: str, name: str, arguments: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def test_run_agent_executes_tool_call_then_returns_final_answer(monkeypatch):
    tool_call_response = MagicMock()
    tool_call_response.choices = [
        MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[_mock_tool_call("call_1", "get_forecast_summary", {})],
            )
        )
    ]
    final_response = MagicMock()
    final_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Democrats are projected to win 215 seats.",
                tool_calls=None,
            )
        )
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [tool_call_response, final_response]
    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    monkeypatch.setitem(sys.modules, "groq", mock_groq_module)

    answer = run_agent("How many seats do Democrats win?", _FORECAST, api_key="gsk-test")

    assert answer == "Democrats are projected to win 215 seats."
    assert mock_client.chat.completions.create.call_count == 2

    # The second call's messages must include the tool's result fed back in.
    second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"])["median_dem_seats"] == 215


def test_run_agent_returns_directly_when_no_tool_call_needed(monkeypatch):
    final_response = MagicMock()
    final_response.choices = [
        MagicMock(
            message=MagicMock(content="I need more specifics to answer that.", tool_calls=None)
        )
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = final_response
    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_client
    monkeypatch.setitem(sys.modules, "groq", mock_groq_module)

    answer = run_agent("hello", _FORECAST, api_key="gsk-test")

    assert answer == "I need more specifics to answer that."
    assert mock_client.chat.completions.create.call_count == 1
