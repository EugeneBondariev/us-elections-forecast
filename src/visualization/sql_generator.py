"""Natural-language to SQL query generator for the elections database.

Accepts a plain-English question about the forecast data, sends the elections.db
schema and question to Groq, executes the returned SQL read-only, and returns the
result as a pandas DataFrame. Only SELECT statements are permitted; any attempt by
the model to generate mutating SQL is blocked before execution.

Used by:
    - `make sql-query` — interactive CLI prompt
    - app.py Streamlit page — "Ask a question about the data"

Run:
    python -m src.visualization.sql_generator
or:
    make sql-query
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd
from groq import Groq

from src.paths import DataFile

logger = logging.getLogger(__name__)

__all__ = [
    "get_db_schema",
    "generate_sql",
    "execute_query",
    "ask",
]

_GROQ_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 256
# Patterns that indicate mutating SQL — block before execution
_MUTATING_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def get_db_schema(db_path: Path = DataFile.DB) -> str:
    """Return the CREATE TABLE statements for all non-system tables in the database.

    The schema string is included verbatim in the LLM prompt so the model knows
    exactly what columns are available and can write correct SQL.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found — run `make db` to build the forecast database"
        )
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()
    if not tables:
        return "(no tables found)"
    return "\n\n".join(f"-- {name}\n{sql};" for name, sql in tables if sql)


def generate_sql(
    question: str,
    schema: str,
    api_key: str,
    model: str = _GROQ_MODEL,
) -> str:
    """Use Groq to translate a natural-language question into a SELECT statement.

    The prompt instructs the model to return ONLY the SQL with no surrounding prose.
    Any non-SELECT result triggers a ValueError before execution.
    """
    prompt = (
        "You are an expert SQL assistant. Given the SQLite schema below and a question, "
        "write a single valid SELECT statement that answers the question. "
        "Return ONLY the SQL — no explanation, no markdown code fences, no leading text.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def execute_query(sql: str, db_path: Path = DataFile.DB) -> pd.DataFrame:
    """Execute a SQL statement read-only and return results as a DataFrame.

    Raises ValueError if the statement is not a SELECT or contains mutating keywords.
    Raises sqlite3.OperationalError if the SQL is invalid.
    """
    clean = sql.strip()
    if _MUTATING_PATTERN.search(clean):
        raise ValueError(
            f"Refusing to execute potentially mutating SQL — only SELECT is allowed.\n{clean}"
        )
    if not re.match(r"^\s*SELECT\b", clean, re.IGNORECASE):
        raise ValueError(
            f"Generated SQL is not a SELECT statement — cannot execute safely.\n{clean}"
        )
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(clean, conn)


def ask(
    question: str,
    api_key: str,
    db_path: Path = DataFile.DB,
    model: str = _GROQ_MODEL,
) -> tuple[str, pd.DataFrame]:
    """Translate a question to SQL, execute it, and return (sql, result_df).

    This is the main entry point used by both the Streamlit app and the CLI.
    Raises on generation failure or unsafe/invalid SQL.
    """
    schema = get_db_schema(db_path=db_path)
    sql = generate_sql(question=question, schema=schema, api_key=api_key, model=model)
    logger.info("Generated SQL: %s", sql)
    result = execute_query(sql=sql, db_path=db_path)
    logger.info("Query returned %d rows", len(result))
    return sql, result


def main() -> None:
    import logging as _logging
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.error(
            "GROQ_API_KEY not set — cannot generate SQL. Register free at https://console.groq.com"
        )
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()

    if not question:
        logger.error("No question provided")
        sys.exit(1)

    model = os.getenv("GROQ_MODEL", _GROQ_MODEL)
    try:
        sql, result = ask(question=question, api_key=api_key, model=model)
        print(f"\nSQL:\n{sql}\n")
        print(result.to_string(index=False))
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
