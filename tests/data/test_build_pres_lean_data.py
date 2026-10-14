"""Tests for scripts/build_pres_lean_data.py. Network calls mocked, no real fetches."""
from __future__ import annotations

from unittest.mock import MagicMock

from scripts.build_pres_lean_data import SOURCES, fetch_and_save
from src.paths import DataFile


def test_sources_map_to_the_expected_destinations():
    assert set(SOURCES.values()) == {DataFile.PRES_08_20, DataFile.PRES_20_24}
    assert all(isinstance(sheet_id, str) and sheet_id for sheet_id in SOURCES)


def test_fetch_and_save_writes_bytes_unchanged(tmp_path, monkeypatch):
    """Regression guard for the \\r\\n -> \\r\\r\\n bug: writing must be byte-for-byte,
    with no newline translation at all - caught by running this against the real
    download and finding doubled CRs, not by a test written in advance."""
    raw_bytes = b"District,Incumbent,Party\r\nAK-AL,Mary Peltola,(D)\r\n"

    mock_response = MagicMock()
    mock_response.content = raw_bytes
    mock_response.raise_for_status = MagicMock()

    mock_get = MagicMock(return_value=mock_response)
    monkeypatch.setattr("scripts.build_pres_lean_data.requests.get", mock_get)

    dest = tmp_path / "out.csv"
    n_lines = fetch_and_save("fake-sheet-id", dest)

    written = dest.read_bytes()
    assert written == raw_bytes
    assert b"\r\r\n" not in written
    assert n_lines == raw_bytes.count(b"\n")


def test_fetch_and_save_creates_parent_directory(tmp_path, monkeypatch):
    mock_response = MagicMock()
    mock_response.content = b"a,b,c\r\n"
    mock_response.raise_for_status = MagicMock()
    monkeypatch.setattr(
        "scripts.build_pres_lean_data.requests.get", MagicMock(return_value=mock_response)
    )

    dest = tmp_path / "nested" / "dir" / "out.csv"
    fetch_and_save("fake-sheet-id", dest)

    assert dest.exists()
    assert dest.read_bytes() == b"a,b,c\r\n"


def test_fetch_and_save_requests_csv_export_url(tmp_path, monkeypatch):
    mock_response = MagicMock()
    mock_response.content = b"x\r\n"
    mock_response.raise_for_status = MagicMock()
    mock_get = MagicMock(return_value=mock_response)
    monkeypatch.setattr("scripts.build_pres_lean_data.requests.get", mock_get)

    fetch_and_save("some-sheet-id", tmp_path / "out.csv")

    called_url = mock_get.call_args[0][0]
    assert called_url == "https://docs.google.com/spreadsheets/d/some-sheet-id/export?format=csv"
