"""Tests for src/rag.py — chromadb/groq clients are mocked, no network calls."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from src.visualization.rag import _district_to_text, rag_answer, retrieve


def test_district_to_text_open_seat():
    row = {"label": "PA-7", "proba_dem": 0.65, "is_open": True, "district_incumbent": "unknown"}
    text = _district_to_text(row)
    assert "open seat" in text
    assert "65.0%" in text


def test_district_to_text_closed_seat_names_incumbent():
    row = {"label": "AZ-6", "proba_dem": 0.3, "is_open": False, "district_incumbent": "REPUBLICAN"}
    text = _district_to_text(row)
    assert "Republican" in text
    assert "open seat" not in text


def test_retrieve_uses_passed_collection():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["District PA-7: Democrats favored."]],
        "metadatas": [[{"state": "PA", "district": 7, "proba_dem": 0.55}]],
    }

    result = retrieve("competitive districts in PA", collection=mock_collection)

    assert result == [
        {"text": "District PA-7: Democrats favored.", "state": "PA", "district": 7, "proba_dem": 0.55}
    ]
    mock_collection.query.assert_called_once_with(
        query_texts=["competitive districts in PA"], n_results=5
    )


def test_rag_answer_without_key_returns_message(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert rag_answer("anything", api_key=None) == "GROQ_API_KEY not set."


def test_rag_answer_calls_groq_with_retrieved_context(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["District PA-7: Democrats favored."]],
        "metadatas": [[{"state": "PA", "district": 7, "proba_dem": 0.55}]],
    }

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="PA-7 leans Democratic."))]
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.return_value = mock_response

    mock_groq_module = MagicMock()
    mock_groq_module.Groq.return_value = mock_groq_client
    monkeypatch.setitem(sys.modules, "groq", mock_groq_module)

    answer = rag_answer(
        "Which PA districts are competitive?", collection=mock_collection, api_key="gsk-test"
    )

    assert answer == "PA-7 leans Democratic."
    mock_groq_module.Groq.assert_called_once_with(api_key="gsk-test")
