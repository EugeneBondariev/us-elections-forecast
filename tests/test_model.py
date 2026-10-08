from src.features import build_pred_2026
from src.model import (
    build_calibrated_pipeline,
    feature_weights,
    train,
    walk_forward_cv,
)


# --- existing ---

def test_predictions_are_binary(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    pred, _ = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    assert set(pipeline.predict(pred)).issubset({0, 1})


def test_predictions_cover_most_districts(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    pred, _ = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    assert len(pipeline.predict(pred)) > 400


def test_feature_weights_keys_match_columns(real_data):
    pipeline = train(real_data["X"], real_data["y"])
    weights = feature_weights(pipeline, real_data["X"].columns)
    assert set(weights.keys()) == set(real_data["X"].columns)


# --- walk_forward_cv ---

def test_walk_forward_cv_returns_dataframe(real_data):
    result = walk_forward_cv(real_data["X"], real_data["y"], real_data["years"])
    assert isinstance(result, __import__("pandas").DataFrame)


def test_walk_forward_cv_expected_columns(real_data):
    result = walk_forward_cv(real_data["X"], real_data["y"], real_data["years"])
    assert set(result.columns) == {"year", "actual_dem", "predicted_dem", "seat_error", "district_accuracy"}


def test_walk_forward_cv_no_test_before_start(real_data):
    result = walk_forward_cv(real_data["X"], real_data["y"], real_data["years"], start_test_year=2010)
    assert (result["year"] >= 2010).all()


def test_walk_forward_cv_seat_error_reasonable(real_data):
    result = walk_forward_cv(real_data["X"], real_data["y"], real_data["years"])
    mae = result["seat_error"].abs().mean()
    assert mae < 30, f"Mean seat error {mae:.1f} is unreasonably large"


def test_walk_forward_cv_district_accuracy_reasonable(real_data):
    result = walk_forward_cv(real_data["X"], real_data["y"], real_data["years"])
    assert (result["district_accuracy"] > 0.80).all()


# --- calibrated pipeline ---

def test_calibrated_pipeline_probabilities_in_range(real_data):
    pipeline = build_calibrated_pipeline()
    pipeline.fit(real_data["X"], real_data["y"])
    pred, _ = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    proba = pipeline.predict_proba(pred)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_calibrated_pipeline_predictions_binary(real_data):
    pipeline = build_calibrated_pipeline()
    pipeline.fit(real_data["X"], real_data["y"])
    pred, _ = build_pred_2026(real_data["winners"], real_data["dems_reps"], real_data["X"].columns)
    assert set(pipeline.predict(pred)).issubset({0, 1})
