from __future__ import annotations

import logging
import os
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.constants import Col, DEM_MAJORITY_SEATS
from src.features.features import build_dataset, build_pred_2026
from src.logging_config import setup_logging
from src.models.model import (
    bootstrap_ci,
    compute_sample_weights,
    cross_validate,
    feature_weights,
    load_simulation_params,
    simulate_election,
    train,
    walk_forward_calibration,
    walk_forward_cv,
    walk_forward_feature_importance,
)
from src.paths import MODELS_DIR, DataFile
from src.visualization.narrative import generate_narrative

logger = logging.getLogger(__name__)


def _label(state_po: str, district: int) -> str:
    return f"{state_po}-{'AL' if district == 0 else district}"


def _log_walk_forward(wf: pd.DataFrame) -> None:
    logger.info("Walk-forward validation (2000-2024):")
    logger.info("  %4s  %6s  %6s  %6s  %8s", "year", "actual", "pred", "error", "accuracy")
    for row in wf.itertuples(index=False):
        logger.info("  %4d  %6d  %6d  %+6d  %7.1f%%",
                    row.year, row.actual_dem, row.predicted_dem,
                    row.seat_error, float(row.district_accuracy) * 100)  # type: ignore[arg-type]
    logger.info("  Mean absolute seat error: %.1f", wf["seat_error"].abs().mean())


def _log_cv_summary(wf: pd.DataFrame, cv_accuracy: float) -> None:
    logger.info("Random CV accuracy:       %.4f", cv_accuracy)
    logger.info("Walk-forward accuracy:    %.4f", wf["district_accuracy"].mean())


def _log_simulation(sim: np.ndarray, n_districts: int) -> None:
    logger.info("2026 seat projection (10,000 simulations):")
    logger.info("  Median:           %d Dem / %d Rep",
                int(np.median(sim)), n_districts - int(np.median(sim)))
    logger.info("  68%% CI:           %d-%d Dem",
                int(np.percentile(sim, 16)), int(np.percentile(sim, 84)))
    logger.info("  95%% CI:           %d-%d Dem",
                int(np.percentile(sim, 2.5)), int(np.percentile(sim, 97.5)))
    logger.info("  P(Dem majority):  %.1f%%", (sim >= DEM_MAJORITY_SEATS).mean() * 100)


def _log_competitive_districts(district_ids: pd.DataFrame) -> None:
    competitive = (
        district_ids
        .assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
        .sort_values("closeness")
        .head(15)
    )
    logger.info("Most competitive districts:")
    for row in competitive.itertuples(index=False):
        lean = "Dem" if float(row.proba_dem) >= 0.5 else "Rep"  # type: ignore[arg-type]
        logger.info("  %-8s  %.1f%% %s", row.label, float(row.proba_dem) * 100, lean)  # type: ignore[arg-type]


def _log_feature_weights(pipeline: object, X_columns: pd.Index) -> None:
    logger.info("Feature weights:")
    for feat, w in feature_weights(pipeline, X_columns).items():
        logger.info("  %s: %.1f%%", feat, w)


def _log_feature_stability(fi: pd.DataFrame) -> None:
    logger.info("Feature importance stability (walk-forward 2000-2024):")
    logger.info("  %-45s  %5s  %5s  %5s", "feature", "min%", "max%", "std")
    for col in fi.columns:
        logger.info("  %-45s  %5.1f  %5.1f  %5.1f",
                    col, fi[col].min(), fi[col].max(), fi[col].std())


def _log_probabilistic_calibration(calib) -> None:
    logger.info("Walk-forward probabilistic calibration (Brier score / log loss):")
    logger.info("  %4s  %6s  %8s", "year", "brier", "log_loss")
    for row in calib.itertuples(index=False):
        logger.info("  %4d  %6.4f  %8.4f", row.year, row.brier_score, row.log_loss)
    logger.info("  Mean Brier score: %.4f  |  Mean log loss: %.4f",
                calib["brier_score"].mean(), calib["log_loss"].mean())


def main() -> None:
    log_path = setup_logging()
    logger.info("Run log: %s", log_path)

    ds = build_dataset()
    X, y, years, winners = ds.X, ds.y, ds.years, ds.winners
    pres_lean, fundraising, dems_reps = ds.pres_lean, ds.fundraising, ds.dems_reps

    cook_alpha = 0.35 if DataFile.COOK.exists() else 0.0

    # Without an explicit tracking URI, mlflow defaults to a sqlite db wherever the
    # cwd happens to be when this runs - which left a stray mlflow.db at the repo
    # root. models/ is the right home: this is training-run metadata (params,
    # metrics), not "data" in the DS sense, same reasoning as calibrated_sigma.json
    # already living there (best_params_lgbm.json moved to config/ instead, since
    # unlike these it's committed rather than gitignored-and-regenerable).
    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("us-house-forecast")
    with mlflow.start_run():
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_train_samples", len(X))
        mlflow.log_param("cook_alpha", cook_alpha)

        wf = walk_forward_cv(X, y, years)
        _log_walk_forward(wf)
        _log_cv_summary(wf, cross_validate(X, y))

        ci_results = bootstrap_ci(X, y, years)
        logger.info("Bootstrap 95%% CI (200 draws):")
        logger.info("  Accuracy: %.4f [%.4f, %.4f]",
                    ci_results["accuracy_mean"], ci_results["accuracy_low"], ci_results["accuracy_high"])
        logger.info("  Seat error: %.1f [%.1f, %.1f]",
                    ci_results["seat_error_mean"], ci_results["seat_error_low"], ci_results["seat_error_high"])
        mlflow.log_metric("bootstrap_accuracy_mean", ci_results["accuracy_mean"])
        mlflow.log_metric("bootstrap_accuracy_low", ci_results["accuracy_low"])
        mlflow.log_metric("bootstrap_accuracy_high", ci_results["accuracy_high"])

        _log_feature_stability(walk_forward_feature_importance(X, y, years))

        calib = walk_forward_calibration(X, y, years)
        _log_probabilistic_calibration(calib)
        mlflow.log_metric("mean_brier_score", calib["brier_score"].mean())
        mlflow.log_metric("mean_log_loss", calib["log_loss"].mean())
        for row in calib.itertuples(index=False):
            mlflow.log_metric(f"brier_{int(row.year)}", float(row.brier_score))  # type: ignore[arg-type]
            mlflow.log_metric(f"log_loss_{int(row.year)}", float(row.log_loss))  # type: ignore[arg-type]

        for row in wf.itertuples(index=False):
            mlflow.log_metric(f"accuracy_{int(row.year)}", float(row.district_accuracy))  # type: ignore[arg-type]
            mlflow.log_metric(f"seat_error_{int(row.year)}", float(row.seat_error))  # type: ignore[arg-type]
        mlflow.log_metric("mean_wf_accuracy", wf["district_accuracy"].mean())
        mlflow.log_metric("mean_abs_seat_error", wf["seat_error"].abs().mean())

        pipeline = train(X, y, sample_weight=compute_sample_weights(years))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="us-house-stacking-ensemble",
        )
        logger.info("Model logged to MLflow registry as 'us-house-stacking-ensemble'")
        pred_2026, district_ids = build_pred_2026(
            winners, dems_reps, X.columns, pres_lean=pres_lean, fundraising=fundraising
        )
        proba = pipeline.predict_proba(pred_2026)[:, 1]

        sim_params = load_simulation_params()
        mlflow.log_param("sigma_national", sim_params["sigma_national"])
        mlflow.log_param("sigma_district", sim_params["sigma_district"])
        sim = simulate_election(
            proba,
            sigma_national=sim_params["sigma_national"],
            sigma_district=sim_params["sigma_district"],
        )
        _log_simulation(sim, len(proba))
        narrative = generate_narrative(sim, district_ids, len(proba))
        if narrative:
            logger.info("Forecast narrative:\n%s", narrative)
        mlflow.log_metric("median_dem_seats", int(np.median(sim)))
        mlflow.log_metric("p_dem_majority", float((sim >= DEM_MAJORITY_SEATS).mean()))

        for feat, w in list(feature_weights(pipeline, X.columns).items())[:15]:
            mlflow.log_metric(f"fi_{feat[:40]}", round(w, 2))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            wf.to_csv(f, index=False)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="walk_forward")
        os.unlink(tmp)

        logger.info("MLflow run logged (experiment: us-house-forecast)")

        district_ids = district_ids.copy()
        district_ids["proba_dem"] = proba
        district_ids["label"] = district_ids.apply(
            lambda r: _label(r[Col.STATE], r[Col.DISTRICT]), axis=1
        )
        _log_competitive_districts(district_ids)
        _log_feature_weights(pipeline, X.columns)


if __name__ == "__main__":
    main()
