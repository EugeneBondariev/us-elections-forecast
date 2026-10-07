import pytest

from src.features import build_pred_2026
from src.model import cross_validate, feature_weights, train


def test_cv_accuracy_above_threshold(real_data):
    acc = cross_validate(real_data["X"], real_data["y"])
    assert acc > 0.88, f"CV accuracy {acc:.4f} fell below 0.88 — model may have regressed"


def test_predictions_are_binary(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    pred = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    assert set(pipeline.predict(pred)).issubset({0, 1})


def test_predictions_cover_most_districts(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    pred = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    assert len(pipeline.predict(pred)) > 400


def test_feature_weights_sum_to_100(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    weights = feature_weights(pipeline, real_data["X"].columns)
    assert pytest.approx(sum(weights.values()), abs=0.1) == 100.0


def test_feature_weights_keys_match_columns(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    weights = feature_weights(pipeline, real_data["X"].columns)
    assert set(weights.keys()) == set(real_data["X"].columns)
