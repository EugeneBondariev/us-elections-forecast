import logging
import os
from pathlib import Path

import numpy as np

from src.logging_config import setup_logging
from src.fundraising import load_fundraising
from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    get_top_candidate_per_party,
    build_pred_2026,
    build_winners,
    load_approval,
    load_clean_house_data,
    load_pres_lean,
)
from src.model import (
    cross_validate,
    feature_weights,
    simulate_election,
    train,
    walk_forward_cv,
    walk_forward_feature_importance,
)

DATA_DIR = Path("data")
logger = logging.getLogger(__name__)


def _label(state_po: str, district: int) -> str:
    return f"{state_po}-{'AL' if district == 0 else district}"


def _log_walk_forward(wf) -> None:
    logger.info("Walk-forward validation (2000-2024):")
    logger.info("  %4s  %6s  %6s  %6s  %8s", "year", "actual", "pred", "error", "accuracy")
    for _, row in wf.iterrows():
        logger.info("  %4d  %6d  %6d  %+6d  %7.1f%%",
                    row["year"], row["actual_dem"], row["predicted_dem"],
                    row["seat_error"], row["district_accuracy"] * 100)
    logger.info("  Mean absolute seat error: %.1f", wf["seat_error"].abs().mean())


def _log_cv_summary(wf, cv_accuracy: float) -> None:
    logger.info("Random CV accuracy:       %.4f", cv_accuracy)
    logger.info("Walk-forward accuracy:    %.4f", wf["district_accuracy"].mean())


def _log_simulation(sim, n_districts: int) -> None:
    logger.info("2026 seat projection (10,000 simulations):")
    logger.info("  Median:           %d Dem / %d Rep",
                int(np.median(sim)), n_districts - int(np.median(sim)))
    logger.info("  68%% CI:           %d-%d Dem",
                int(np.percentile(sim, 16)), int(np.percentile(sim, 84)))
    logger.info("  95%% CI:           %d-%d Dem",
                int(np.percentile(sim, 2.5)), int(np.percentile(sim, 97.5)))
    logger.info("  P(Dem majority):  %.1f%%", (sim >= 218).mean() * 100)


def _log_competitive_districts(district_ids) -> None:
    competitive = (
        district_ids
        .assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
        .sort_values("closeness")
        .head(15)
    )
    logger.info("Most competitive districts:")
    for _, row in competitive.iterrows():
        lean = "Dem" if row["proba_dem"] >= 0.5 else "Rep"
        logger.info("  %-8s  %.1f%% %s", row["label"], row["proba_dem"] * 100, lean)


def _log_feature_weights(pipeline, X_columns) -> None:
    logger.info("Feature weights:")
    for feat, w in feature_weights(pipeline, X_columns).items():
        logger.info("  %s: %.1f%%", feat, w)


def _log_feature_stability(fi) -> None:
    logger.info("Feature importance stability (walk-forward 2000-2024):")
    logger.info("  %-45s  %5s  %5s  %5s", "feature", "min%", "max%", "std")
    for col in fi.columns:
        logger.info("  %-45s  %5.1f  %5.1f  %5.1f",
                    col, fi[col].min(), fi[col].max(), fi[col].std())


def main() -> None:
    log_path = setup_logging()
    logger.info("Run log: %s", log_path)

    df = load_clean_house_data(DATA_DIR / "1976-2024-house.tab")
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls")
    pres_lean = load_pres_lean(
        DATA_DIR / "president-2008-2020-by-district.csv",
        DATA_DIR / "president-2020-2024-by-district.csv",
    )
    fundraising = load_fundraising(
        cache_path=DATA_DIR / "fec_fundraising.csv",
        api_key=os.getenv("FEC_API_KEY"),
    )
    winners = build_winners(top_candidates, dems_reps, approval, pres_lean, fundraising)
    winners = add_incumbency_features(winners, df)
    X, y, years = build_feature_matrix(winners)

    wf = walk_forward_cv(X, y, years)
    _log_walk_forward(wf)
    _log_cv_summary(wf, cross_validate(X, y))
    _log_feature_stability(walk_forward_feature_importance(X, y, years))

    pipeline = train(X, y)
    pred_2026, district_ids = build_pred_2026(
        winners, dems_reps, X.columns, pres_lean=pres_lean, fundraising=fundraising
    )
    proba = pipeline.predict_proba(pred_2026)[:, 1]

    _log_simulation(simulate_election(proba), len(proba))

    district_ids = district_ids.copy()
    district_ids["proba_dem"] = proba
    district_ids["label"] = district_ids.apply(
        lambda r: _label(r["state_po"], r["district"]), axis=1
    )
    _log_competitive_districts(district_ids)
    _log_feature_weights(pipeline, X.columns)


if __name__ == "__main__":
    main()
