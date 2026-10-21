"""Tests for src/data/speech_sentiment.py. No network access — all mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.speech_sentiment import (
    build_speech_features,
    fetch_house_members,
    fetch_member_bills,
    score_sentiment,
)


def _make_member(
    state: str = "CA",
    district: int = 1,
    bioguide: str = "A000001",
) -> dict:
    return {
        "bioguideId": bioguide,
        "state": state,
        "district": district,
        "name": "Test Member",
        "partyName": "Democratic",
    }


vader = pytest.importorskip("vaderSentiment", reason="vaderSentiment not installed")


class TestScoreSentiment:
    def test_empty_returns_zero(self) -> None:
        assert score_sentiment([]) == 0.0

    def test_positive_text(self) -> None:
        assert score_sentiment(["great success amazing achievement"]) > 0

    def test_negative_text(self) -> None:
        assert score_sentiment(["terrible disaster fail corrupt"]) < 0

    def test_mean_of_two(self) -> None:
        titles = ["wonderful excellent bill", "horrible failure disaster"]
        result = score_sentiment(titles)
        # Both individual scores are non-zero; the mean must sit between the two
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        s1 = sia.polarity_scores(titles[0])["compound"]
        s2 = sia.polarity_scores(titles[1])["compound"]
        expected = (s1 + s2) / 2
        assert abs(result - expected) < 1e-9


class TestFetchHouseMembers:
    def test_cache_hit_skips_network(self, tmp_path: Path) -> None:
        cached = [_make_member("CA", 1), _make_member("TX", 5, "B000002")]
        cache_file = tmp_path / "members.json"
        cache_file.write_text(json.dumps(cached))

        with patch("requests.get", side_effect=RuntimeError("should not be called")):
            result = fetch_house_members(cache_path=cache_file)

        assert result == cached

    def test_single_page_fetch(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"members": [_make_member("CA", i) for i in range(1, 4)]}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = fetch_house_members(congress=119, api_key="key123")

        assert len(result) == 3


class TestFetchMemberBills:
    def test_returns_titles(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "sponsoredLegislation": [{"title": "A Bill"}, {"title": "Another Bill"}]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = fetch_member_bills(bioguide_id="A000001")

        assert result == ["A Bill", "Another Bill"]

    def test_http_error_returns_empty(self) -> None:
        with patch("requests.get", side_effect=requests.RequestException("timeout")):
            result = fetch_member_bills(bioguide_id="A000001")

        assert result == []


# Import needed for the side_effect reference above to resolve at module load time.
import requests  # noqa: E402


class TestBuildSpeechFeatures:
    def test_output_columns(self) -> None:
        members = [_make_member("CA", 1), _make_member("NY", 5, "B000002")]
        with patch("src.data.speech_sentiment.fetch_member_bills", return_value=[]):
            result = build_speech_features(members=members)
        assert list(result.columns) == ["state_po", "district", "speech_sentiment", "speech_count"]

    def test_skips_member_without_state(self) -> None:
        members = [{"bioguideId": "X000001", "state": None, "district": 1}]
        with patch("src.data.speech_sentiment.fetch_member_bills", return_value=[]):
            result = build_speech_features(members=members)
        assert result.empty

    def test_district_dtype(self) -> None:
        members = [_make_member("CA", 1), _make_member("NY", 5, "B000002")]
        with patch("src.data.speech_sentiment.fetch_member_bills", return_value=[]):
            result = build_speech_features(members=members)
        assert result["district"].dtype == int
