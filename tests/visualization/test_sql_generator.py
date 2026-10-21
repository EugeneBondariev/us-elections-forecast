"""Tests for src/visualization/sql_generator.py. No Groq API or real DB calls."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.visualization.sql_generator import (
    execute_query,
    generate_sql,
    get_db_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> Path:
    """Create a minimal SQLite DB with one table for testing."""
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE forecasts (state_po TEXT, district INTEGER, win_prob REAL)")
        conn.execute("INSERT INTO forecasts VALUES ('CA', 1, 0.75)")
        conn.execute("INSERT INTO forecasts VALUES ('NY', 5, 0.48)")
    return db


# ---------------------------------------------------------------------------
# get_db_schema
# ---------------------------------------------------------------------------


class TestGetDbSchema:
    def test_returns_create_statement(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        schema = get_db_schema(db_path=db)
        assert "forecasts" in schema
        assert "state_po" in schema

    def test_missing_db_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_db_schema(db_path=tmp_path / "no_such.db")

    def test_empty_db_returns_no_tables_message(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.db"
        with sqlite3.connect(db) as conn:
            conn.execute("SELECT 1")
        schema = get_db_schema(db_path=db)
        assert "no tables" in schema

    def test_multiple_tables_all_included(self, tmp_path: Path) -> None:
        db = tmp_path / "multi.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE alpha (x INTEGER)")
            conn.execute("CREATE TABLE beta (y TEXT)")
        schema = get_db_schema(db_path=db)
        assert "alpha" in schema
        assert "beta" in schema


# ---------------------------------------------------------------------------
# generate_sql
# ---------------------------------------------------------------------------


class TestGenerateSql:
    def _mock_groq(self, content: str) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def test_returns_stripped_sql(self) -> None:
        client = self._mock_groq("SELECT * FROM forecasts")
        with patch("src.visualization.sql_generator.Groq", return_value=client):
            result = generate_sql(
                question="Show all forecasts",
                schema="CREATE TABLE forecasts (state_po TEXT)",
                api_key="key",
            )
        assert result == "SELECT * FROM forecasts"

    def test_strips_markdown_fences(self) -> None:
        client = self._mock_groq("```sql\nSELECT * FROM forecasts\n```")
        with patch("src.visualization.sql_generator.Groq", return_value=client):
            result = generate_sql(
                question="Show all",
                schema="CREATE TABLE forecasts (x INTEGER)",
                api_key="key",
            )
        assert "```" not in result
        assert "SELECT" in result

    def test_strips_plain_fences(self) -> None:
        client = self._mock_groq("```\nSELECT 1\n```")
        with patch("src.visualization.sql_generator.Groq", return_value=client):
            result = generate_sql(question="q", schema="s", api_key="k")
        assert result == "SELECT 1"

    def test_passes_model_to_groq(self) -> None:
        client = self._mock_groq("SELECT 1")
        with patch("src.visualization.sql_generator.Groq", return_value=client):
            generate_sql(
                question="q",
                schema="s",
                api_key="k",
                model="llama-3.3-70b-versatile",
            )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_schema_included_in_prompt(self) -> None:
        client = self._mock_groq("SELECT 1")
        with patch("src.visualization.sql_generator.Groq", return_value=client):
            generate_sql(
                question="how many?",
                schema="CREATE TABLE votes (n INTEGER)",
                api_key="k",
            )
        prompt_content = client.chat.completions.create.call_args[1]["messages"][0]["content"]
        assert "CREATE TABLE votes" in prompt_content


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


class TestExecuteQuery:
    def test_select_returns_dataframe(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        df = execute_query(sql="SELECT * FROM forecasts", db_path=db)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_where_clause_filters(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        df = execute_query(sql="SELECT * FROM forecasts WHERE state_po = 'CA'", db_path=db)
        assert len(df) == 1
        assert df.iloc[0]["state_po"] == "CA"

    def test_insert_blocked(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(ValueError, match="mutating"):
            execute_query(sql="INSERT INTO forecasts VALUES ('TX', 2, 0.5)", db_path=db)

    def test_update_blocked(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(ValueError, match="mutating"):
            execute_query(sql="UPDATE forecasts SET win_prob=0.9", db_path=db)

    def test_delete_blocked(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(ValueError, match="mutating"):
            execute_query(sql="DELETE FROM forecasts", db_path=db)

    def test_drop_blocked(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(ValueError, match="mutating"):
            execute_query(sql="DROP TABLE forecasts", db_path=db)

    def test_non_select_without_mutating_keyword_blocked(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(ValueError, match="not a SELECT"):
            execute_query(sql="EXPLAIN SELECT * FROM forecasts", db_path=db)

    def test_invalid_sql_raises_operational_error(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(Exception):
            execute_query(sql="SELECT * FROM nonexistent_table", db_path=db)

    def test_case_insensitive_mutating_check(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(ValueError):
            execute_query(sql="insert into forecasts values ('TX', 3, 0.6)", db_path=db)

    def test_aggregate_query(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        df = execute_query(sql="SELECT COUNT(*) AS n FROM forecasts", db_path=db)
        assert df.iloc[0]["n"] == 2
