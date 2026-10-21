"""Tests for src/data/cosponsorship.py. No network access — all mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.cosponsorship import (
    build_all_cosponsorship,
    build_cosponsorship_for_congress,
    fetch_bill_cosponsors,
    fetch_cosponsored_count,
)

# ---------------------------------------------------------------------------
# fetch_cosponsored_count
# ---------------------------------------------------------------------------


class TestFetchCosponsoredCount:
    def test_returns_pagination_count(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"pagination": {"count": 42}, "cosponsoredLegislation": []}

        with patch("requests.get", return_value=mock_resp):
            result = fetch_cosponsored_count(bioguide_id="A000001")

        assert result == 42

    def test_missing_pagination_returns_zero(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {}

        with patch("requests.get", return_value=mock_resp):
            result = fetch_cosponsored_count(bioguide_id="A000001")

        assert result == 0

    def test_http_error_returns_zero(self) -> None:
        with patch(
            "requests.get",
            side_effect=requests.RequestException("timeout"),
        ):
            result = fetch_cosponsored_count(bioguide_id="A000001")

        assert result == 0

    def test_passes_api_key(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"pagination": {"count": 5}}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            fetch_cosponsored_count(bioguide_id="A000001", api_key="testkey")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["api_key"] == "testkey"


# ---------------------------------------------------------------------------
# fetch_bill_cosponsors
# ---------------------------------------------------------------------------


class TestFetchBillCosponsors:
    def _make_cosponsor(self, bio: str, party: str) -> dict:
        return {"bioguideId": bio, "party": party, "state": "CA"}

    def test_returns_cosponsor_list(self) -> None:
        cosponsors = [self._make_cosponsor("B000002", "D"), self._make_cosponsor("C000003", "R")]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"cosponsors": cosponsors}

        with patch("requests.get", return_value=mock_resp):
            result = fetch_bill_cosponsors(congress=119, bill_type="HR", bill_number="123")

        assert len(result) == 2
        assert result[0]["party"] == "D"

    def test_http_error_returns_empty(self) -> None:
        with patch("requests.get", side_effect=requests.RequestException("err")):
            result = fetch_bill_cosponsors(congress=119, bill_type="HR", bill_number="123")

        assert result == []

    def test_bill_type_lowercased_in_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"cosponsors": []}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            fetch_bill_cosponsors(congress=119, bill_type="HR", bill_number="1")

        url = mock_get.call_args[0][0]
        assert "/hr/" in url


# ---------------------------------------------------------------------------
# build_cosponsorship_for_congress
# ---------------------------------------------------------------------------


class TestBuildCosponsorshipForCongress:
    def _member_cache_data(self) -> list[dict]:
        return [
            {
                "bioguideId": "A000001",
                "state": "California",
                "district": 1,
                "partyName": "Democratic",
            },
            {
                "bioguideId": "B000002",
                "state": "New York",
                "district": 5,
                "partyName": "Republican",
            },
        ]

    def _bills_by_bio(self) -> dict:
        return {
            "A000001": [{"title": "T1", "number": "1", "type": "HR", "congress": 119}],
            "B000002": [{"title": "T2", "number": "2", "type": "HR", "congress": 119}],
        }

    def test_output_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        members = self._member_cache_data()
        bills = self._bills_by_bio()
        bio_map = {"A000001": ("CA", 1), "B000002": ("NY", 5)}
        member_cache = tmp_path / "cm_119.json"
        member_cache.write_text(json.dumps(members))
        feat_cache = tmp_path / "cs_119.json"

        from src.paths import DataFile

        monkeypatch.setattr(
            DataFile, "congress_members_cache", staticmethod(lambda _: member_cache)
        )
        with (
            patch("src.data.cosponsorship._cosponsor_features_cache", return_value=str(feat_cache)),
            patch("src.data.cosponsorship.fetch_sponsored_bills_for_congress", return_value=bills),
            patch("src.data.cosponsorship._build_bio_to_district", return_value=bio_map),
            patch(
                "src.data.cosponsorship._compute_cosponsor_features",
                return_value={"cosponsor_reach": 3, "bipartisan_cosponsor_pct": 0.5},
            ),
            patch("src.data.cosponsorship.fetch_cosponsored_count", return_value=10),
            patch("src.data.cosponsorship.time.sleep"),
        ):
            df = build_cosponsorship_for_congress(congress=119)

        assert set(
            [
                "state_po",
                "district",
                "cosponsor_reach",
                "bipartisan_cosponsor_pct",
                "cosponsored_count",
            ]
        ).issubset(set(df.columns))
        assert len(df) == 2

    def test_bipartisan_pct_in_range(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        members = [
            {"bioguideId": "A000001", "state": "CA", "district": 1, "partyName": "Democratic"}
        ]
        member_cache = tmp_path / "cm.json"
        member_cache.write_text(json.dumps(members))

        from src.paths import DataFile

        monkeypatch.setattr(
            DataFile, "congress_members_cache", staticmethod(lambda _: member_cache)
        )

        feat_cache = tmp_path / "cs.json"
        bills = {"A000001": [{"title": "T", "number": "1", "type": "HR", "congress": 119}]}
        bio_map = {"A000001": ("CA", 1)}

        with (
            patch("src.data.cosponsorship._cosponsor_features_cache", return_value=str(feat_cache)),
            patch("src.data.cosponsorship.fetch_sponsored_bills_for_congress", return_value=bills),
            patch("src.data.cosponsorship._build_bio_to_district", return_value=bio_map),
            patch(
                "src.data.cosponsorship._compute_cosponsor_features",
                return_value={"cosponsor_reach": 5, "bipartisan_cosponsor_pct": 0.4},
            ),
            patch("src.data.cosponsorship.fetch_cosponsored_count", return_value=8),
            patch("src.data.cosponsorship.time.sleep"),
        ):
            df = build_cosponsorship_for_congress(congress=119)

        assert 0.0 <= df.iloc[0]["bipartisan_cosponsor_pct"] <= 1.0

    def test_loads_from_feature_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        members = [
            {"bioguideId": "A000001", "state": "CA", "district": 1, "partyName": "Democratic"}
        ]
        member_cache = tmp_path / "cm.json"
        member_cache.write_text(json.dumps(members))
        bio_map = {"A000001": ("CA", 1)}
        # Pre-populate feature cache so no API calls should be made
        feat_cache = tmp_path / "cs.json"
        feat_cache.write_text(
            json.dumps(
                {
                    "A000001": {
                        "cosponsor_reach": 7,
                        "bipartisan_cosponsor_pct": 0.3,
                        "cosponsored_count": 15,
                    }
                }
            )
        )

        from src.paths import DataFile

        monkeypatch.setattr(
            DataFile, "congress_members_cache", staticmethod(lambda _: member_cache)
        )

        with (
            patch("src.data.cosponsorship._cosponsor_features_cache", return_value=str(feat_cache)),
            patch("src.data.cosponsorship.fetch_sponsored_bills_for_congress", return_value={}),
            patch("src.data.cosponsorship._build_bio_to_district", return_value=bio_map),
            patch(
                "src.data.cosponsorship._compute_cosponsor_features",
                side_effect=RuntimeError("should not be called"),
            ),
            patch(
                "src.data.cosponsorship.fetch_cosponsored_count",
                side_effect=RuntimeError("should not be called"),
            ),
        ):
            df = build_cosponsorship_for_congress(congress=119)

        assert len(df) == 1
        assert df.iloc[0]["cosponsor_reach"] == 7


# ---------------------------------------------------------------------------
# build_all_cosponsorship
# ---------------------------------------------------------------------------


class TestBuildAllCosponsorship:
    def test_year_column_from_congress_number(self) -> None:
        dummy = pd.DataFrame(
            {
                "state_po": ["CA"],
                "district": [1],
                "cosponsor_reach": [3],
                "bipartisan_cosponsor_pct": [0.5],
                "cosponsored_count": [10],
            }
        )

        with patch("src.data.cosponsorship.build_cosponsorship_for_congress", return_value=dummy):
            result = build_all_cosponsorship(start_congress=118, end_congress=119)

        assert set(result["year"].unique()) == {1788 + 2 * 118, 1788 + 2 * 119}

    def test_empty_when_no_congress_data(self) -> None:
        with patch(
            "src.data.cosponsorship.build_cosponsorship_for_congress",
            return_value=pd.DataFrame(),
        ):
            result = build_all_cosponsorship(start_congress=119, end_congress=119)

        assert result.empty
