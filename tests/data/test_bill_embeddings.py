"""Tests for src/data/bill_embeddings.py. No network or sentence-transformers calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.bill_embeddings import (
    _build_bio_to_district,
    build_all_embeddings,
    build_district_embeddings,
    embed_titles,
    fetch_sponsored_bills_for_congress,
)

# Skip the whole module if sentence_transformers is not installed —
# placed after src imports (which don't need it at module level) so ruff E402 is satisfied.
pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_member(bio: str, state: str, district: int) -> dict:
    return {"bioguideId": bio, "state": state, "district": district}


def _fake_bills(n: int = 3) -> list[dict]:
    return [
        {"title": f"Bill {i}", "number": str(i), "type": "HR", "congress": 119} for i in range(n)
    ]


# ---------------------------------------------------------------------------
# embed_titles
# ---------------------------------------------------------------------------


class TestEmbedTitles:
    def test_empty_returns_none(self) -> None:
        assert embed_titles([]) is None

    def test_returns_ndarray(self) -> None:
        fake_vec = np.ones((2, 384), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vec
        with patch("src.data.bill_embeddings.SentenceTransformer", return_value=mock_model):
            result = embed_titles(["a bill", "another bill"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_mean_of_multiple_titles(self) -> None:
        vecs = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = vecs
        with patch("src.data.bill_embeddings.SentenceTransformer", return_value=mock_model):
            result = embed_titles(["a", "b"])
        assert result is not None
        np.testing.assert_allclose(result, [2.0, 0.0])


# ---------------------------------------------------------------------------
# _build_bio_to_district
# ---------------------------------------------------------------------------


class TestBuildBioToDistrict:
    def test_maps_correctly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        members = [
            _make_member("A000001", "California", 1),
            _make_member("B000002", "New York", 5),
        ]
        state_map = {"California": "CA", "New York": "NY"}
        cache = tmp_path / "congress_members_119.json"
        cache.write_text(json.dumps(members))

        from src.paths import DataFile

        monkeypatch.setattr(DataFile, "congress_members_cache", staticmethod(lambda _: cache))
        result = _build_bio_to_district(congress=119, state_name_map=state_map)
        assert result["A000001"] == ("CA", 1)
        assert result["B000002"] == ("NY", 5)

    def test_skips_delegate_states(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        members = [
            _make_member("A000001", "California", 1),
            _make_member("C000003", "Puerto Rico", 0),
        ]
        state_map = {"California": "CA"}  # Puerto Rico excluded
        cache = tmp_path / "congress_members_119.json"
        cache.write_text(json.dumps(members))

        from src.paths import DataFile

        monkeypatch.setattr(DataFile, "congress_members_cache", staticmethod(lambda _: cache))
        result = _build_bio_to_district(congress=119, state_name_map=state_map)
        assert "C000003" not in result
        assert "A000001" in result

    def test_missing_cache_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.paths import DataFile

        missing = tmp_path / "no_such_file.json"
        monkeypatch.setattr(DataFile, "congress_members_cache", staticmethod(lambda _: missing))
        assert _build_bio_to_district(congress=119, state_name_map=None) == {}


# ---------------------------------------------------------------------------
# fetch_sponsored_bills_for_congress
# ---------------------------------------------------------------------------


class TestFetchSponsoredBills:
    def test_cache_hit_skips_network(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cached = {"A000001": _fake_bills(2)}
        cache_file = tmp_path / "sponsored_bills_119.json"
        cache_file.write_text(json.dumps(cached))

        from src.paths import DataFile

        monkeypatch.setattr(
            DataFile, "congress_sponsored_bills_cache", staticmethod(lambda _: cache_file)
        )
        with patch("requests.get", side_effect=RuntimeError("should not be called")):
            result = fetch_sponsored_bills_for_congress(congress=119)
        assert result == cached

    def test_missing_cache_fetches_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_file = tmp_path / "sponsored_bills_119.json"
        member_cache = tmp_path / "congress_members_119.json"
        members = [_make_member("A000001", "California", 1)]
        member_cache.write_text(json.dumps(members))

        from src.paths import DataFile

        monkeypatch.setattr(
            DataFile, "congress_sponsored_bills_cache", staticmethod(lambda _: cache_file)
        )
        monkeypatch.setattr(
            DataFile, "congress_members_cache", staticmethod(lambda _: member_cache)
        )

        def mock_get(url: str, **_: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "sponsored-legislation" in url:
                resp.json.return_value = {"sponsoredLegislation": _fake_bills(2)}
            else:
                resp.json.return_value = {"members": members}
            return resp

        with patch("src.data.bill_embeddings.fetch_house_members", return_value=members):
            with patch("requests.get", side_effect=mock_get):
                result = fetch_sponsored_bills_for_congress(
                    congress=119,
                    state_name_map={"California": "CA"},
                )
        assert "A000001" in result
        assert len(result["A000001"]) == 2


# ---------------------------------------------------------------------------
# build_district_embeddings
# ---------------------------------------------------------------------------


class TestBuildDistrictEmbeddings:
    def test_returns_correct_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_file = tmp_path / "sponsored_bills_119.json"
        member_cache = tmp_path / "congress_members_119.json"
        members = [_make_member("A000001", "California", 1)]
        member_cache.write_text(json.dumps(members))
        bills = {"A000001": _fake_bills(3)}
        cache_file.write_text(json.dumps(bills))

        from src.paths import DataFile

        monkeypatch.setattr(
            DataFile, "congress_sponsored_bills_cache", staticmethod(lambda _: cache_file)
        )
        monkeypatch.setattr(
            DataFile, "congress_members_cache", staticmethod(lambda _: member_cache)
        )

        fake_vec = np.ones((3, 4), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vec

        with patch("src.data.bill_embeddings.SentenceTransformer", return_value=mock_model):
            with patch(
                "src.data.bill_embeddings.fetch_sponsored_bills_for_congress", return_value=bills
            ):
                df = build_district_embeddings(congress=119, state_name_map={"California": "CA"})

        assert set(df.columns) == {"state_po", "district", "_embedding"}
        assert len(df) == 1
        assert df.iloc[0]["state_po"] == "CA"
        assert df.iloc[0]["district"] == 1


# ---------------------------------------------------------------------------
# build_all_embeddings
# ---------------------------------------------------------------------------


class TestBuildAllEmbeddings:
    def test_output_shape_and_cluster_column(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        congress_frames = {
            113: pd.DataFrame(
                {
                    "state_po": ["CA", "NY"],
                    "district": [1, 5],
                    "_embedding": [np.ones(4), np.zeros(4)],
                }
            ),
            119: pd.DataFrame(
                {
                    "state_po": ["CA", "NY"],
                    "district": [1, 5],
                    "_embedding": [np.ones(4) * 2, np.zeros(4)],
                }
            ),
        }

        def fake_build_district_embeddings(congress: int, **_: object) -> pd.DataFrame:
            return congress_frames.get(congress, pd.DataFrame())

        from src.paths import DataFile

        monkeypatch.setattr(DataFile, "BILL_EMBEDDINGS_MODEL", tmp_path / "km.pkl")

        with patch(
            "src.data.bill_embeddings.build_district_embeddings",
            side_effect=fake_build_district_embeddings,
        ):
            result = build_all_embeddings(start_congress=113, end_congress=119, state_name_map=None)

        assert set(result.columns) == {"year", "state_po", "district", "bill_topic_cluster"}
        assert len(result) == 4  # 2 congresses × 2 districts
        assert result["bill_topic_cluster"].dtype == int
        assert (tmp_path / "km.pkl").exists()
