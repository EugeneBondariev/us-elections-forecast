"""Tests for src/data/policy_topics.py. No network or Groq API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.policy_topics import (
    _EQUAL_FRACTIONS,
    TOPIC_KEYS,
    build_all_policy_topics,
    build_policy_topics_for_congress,
    classify_titles_with_groq,
)

# ---------------------------------------------------------------------------
# classify_titles_with_groq
# ---------------------------------------------------------------------------


class TestClassifyTitlesWithGroq:
    def test_empty_titles_returns_equal_fractions(self) -> None:
        result = classify_titles_with_groq(titles=[], api_key="dummy")
        assert result == _EQUAL_FRACTIONS

    def test_valid_json_response_normalized(self) -> None:
        raw_json = json.dumps(
            {
                "healthcare": 2,
                "economy": 2,
                "defense": 2,
                "education": 2,
                "environment": 1,
                "social": 1,
            }
        )
        mock_choice = MagicMock()
        mock_choice.message.content = raw_json
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("src.data.policy_topics.Groq", return_value=mock_client):
            result = classify_titles_with_groq(["A Bill", "Another Bill"], api_key="key")

        assert set(result.keys()) == set(TOPIC_KEYS)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_bad_response_falls_back_to_equal(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "oops no json here"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("src.data.policy_topics.Groq", return_value=mock_client):
            result = classify_titles_with_groq(["A Bill"], api_key="key")

        assert result == _EQUAL_FRACTIONS

    def test_api_exception_falls_back_to_equal(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")

        with patch("src.data.policy_topics.Groq", return_value=mock_client):
            result = classify_titles_with_groq(["Some Bill"], api_key="key")

        assert result == _EQUAL_FRACTIONS

    def test_negative_values_zeroed(self) -> None:
        raw_json = json.dumps(
            {
                "healthcare": -0.5,
                "economy": 1.0,
                "defense": 0.5,
                "education": 0.5,
                "environment": 0.5,
                "social": 0.5,
            }
        )
        mock_choice = MagicMock()
        mock_choice.message.content = raw_json
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("src.data.policy_topics.Groq", return_value=mock_client):
            result = classify_titles_with_groq(["Bill"], api_key="key")

        assert result["healthcare"] == 0.0
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_all_zero_values_falls_back_to_equal(self) -> None:
        raw_json = json.dumps(
            {
                "healthcare": 0,
                "economy": 0,
                "defense": 0,
                "education": 0,
                "environment": 0,
                "social": 0,
            }
        )
        mock_choice = MagicMock()
        mock_choice.message.content = raw_json
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("src.data.policy_topics.Groq", return_value=mock_client):
            result = classify_titles_with_groq(["Bill"], api_key="key")

        assert result == _EQUAL_FRACTIONS


# ---------------------------------------------------------------------------
# build_policy_topics_for_congress
# ---------------------------------------------------------------------------


class TestBuildPolicyTopicsForCongress:
    def _make_bills_by_bio(self) -> dict:
        return {
            "A000001": [{"title": "Healthcare Act", "number": "1", "type": "HR", "congress": 119}],
            "B000002": [{"title": "Defense Bill", "number": "2", "type": "HR", "congress": 119}],
        }

    def _make_bio_to_district(self) -> dict:
        return {
            "A000001": ("CA", 1),
            "B000002": ("NY", 5),
        }

    def test_returns_correct_columns(self, tmp_path: Path) -> None:
        bills = self._make_bills_by_bio()
        bio_to_district = self._make_bio_to_district()
        equal = {k: 1.0 / 6 for k in TOPIC_KEYS}

        with (
            patch(
                "src.data.policy_topics.DataFile.congress_policy_topics_cache",
                staticmethod(lambda n: tmp_path / f"pt_{n}.json"),
            ),
            patch("src.data.policy_topics.fetch_sponsored_bills_for_congress", return_value=bills),
            patch("src.data.policy_topics._build_bio_to_district", return_value=bio_to_district),
            patch("src.data.policy_topics.classify_titles_with_groq", return_value=equal),
            patch("src.data.policy_topics.time.sleep"),
        ):
            df = build_policy_topics_for_congress(congress=119, groq_api_key="key")

        assert set(df.columns) == {"state_po", "district"} | {f"topic_{k}" for k in TOPIC_KEYS}
        assert len(df) == 2

    def test_uses_cache_when_available(self, tmp_path: Path) -> None:
        bills = self._make_bills_by_bio()
        bio_to_district = self._make_bio_to_district()
        cached = {
            "CA_1": {k: 1.0 / 6 for k in TOPIC_KEYS},
            "NY_5": {k: 1.0 / 6 for k in TOPIC_KEYS},
        }
        cache_file = tmp_path / "pt_119.json"
        cache_file.write_text(json.dumps(cached))

        with (
            patch(
                "src.data.policy_topics.DataFile.congress_policy_topics_cache",
                staticmethod(lambda n: cache_file),
            ),
            patch("src.data.policy_topics.fetch_sponsored_bills_for_congress", return_value=bills),
            patch("src.data.policy_topics._build_bio_to_district", return_value=bio_to_district),
            patch(
                "src.data.policy_topics.classify_titles_with_groq",
                side_effect=RuntimeError("should not be called"),
            ),
        ):
            df = build_policy_topics_for_congress(congress=119, groq_api_key="key")

        assert len(df) == 2  # loaded from cache, no Groq calls

    def test_topic_fractions_sum_to_one(self, tmp_path: Path) -> None:
        bills = {"A000001": [{"title": "T", "number": "1", "type": "HR", "congress": 119}]}
        bio_map = {"A000001": ("CA", 1)}
        uneven = {
            "healthcare": 0.3,
            "economy": 0.2,
            "defense": 0.1,
            "education": 0.2,
            "environment": 0.1,
            "social": 0.1,
        }

        with (
            patch(
                "src.data.policy_topics.DataFile.congress_policy_topics_cache",
                staticmethod(lambda n: tmp_path / f"pt_{n}.json"),
            ),
            patch("src.data.policy_topics.fetch_sponsored_bills_for_congress", return_value=bills),
            patch("src.data.policy_topics._build_bio_to_district", return_value=bio_map),
            patch("src.data.policy_topics.classify_titles_with_groq", return_value=uneven),
            patch("src.data.policy_topics.time.sleep"),
        ):
            df = build_policy_topics_for_congress(congress=119, groq_api_key="key")

        row = df.iloc[0]
        total = sum(row[f"topic_{k}"] for k in TOPIC_KEYS)
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# build_all_policy_topics
# ---------------------------------------------------------------------------


class TestBuildAllPolicyTopics:
    def test_year_column_correct(self, tmp_path: Path) -> None:
        dummy_df = pd.DataFrame(
            {"state_po": ["CA"], "district": [1], **{f"topic_{k}": [1.0 / 6] for k in TOPIC_KEYS}}
        )
        call_count = 0

        def fake_build(congress: int, **_: object) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return dummy_df.copy()

        with patch(
            "src.data.policy_topics.build_policy_topics_for_congress", side_effect=fake_build
        ):
            result = build_all_policy_topics(
                start_congress=118, end_congress=119, groq_api_key="key"
            )

        assert call_count == 2
        assert set(result["year"].unique()) == {1788 + 2 * 118, 1788 + 2 * 119}

    def test_empty_when_no_data(self) -> None:
        with patch(
            "src.data.policy_topics.build_policy_topics_for_congress",
            return_value=pd.DataFrame(),
        ):
            result = build_all_policy_topics(start_congress=119, end_congress=119, groq_api_key="k")

        assert result.empty
        assert "year" in result.columns
