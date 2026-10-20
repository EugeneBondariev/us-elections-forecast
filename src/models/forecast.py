from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import COOK_ALPHA, OPEN_SEATS_2026, Col
from src.data.loaders import load_cook_ratings
from src.features.features import build_dataset, build_pred_2026
from src.models.model import VoteShareRegressor, compute_sample_weights, explain_districts, train
from src.models.simulation import load_simulation_params, simulate_blended
from src.paths import DataFile
from src.types import ForecastResult

__all__ = ["run_forecast"]


def run_forecast() -> ForecastResult:
    """Run the full 2026 House forecast pipeline.

    Returns a dict with:
      seats_sim   – np.ndarray of 10k simulated total Dem seat counts
      n_districts – number of contested districts
      districts   – DataFrame with per-district probabilities and metadata
    """
    ds = build_dataset()
    X, y, years, winners = ds.X, ds.y, ds.years, ds.winners
    pres_lean, fundraising, dems_reps = ds.pres_lean, ds.fundraising, ds.dems_reps

    weights = compute_sample_weights(years)
    pipeline = train(X, y, sample_weight=weights, y_vote_share=ds.y_vote_share)
    pred_2026, district_ids = build_pred_2026(
        winners, dems_reps, X.columns,
        pres_lean=pres_lean, fundraising=fundraising,
        census=ds.census, candidate_quality=ds.candidate_quality,
        district_analytics=ds.district_analytics,
        gerrymandering=ds.gerrymandering,
        speech_sentiment=ds.speech_sentiment,
    )
    proba = pipeline.predict_proba(pred_2026)[:, 1]
    shap_df = explain_districts(pipeline, pred_2026)

    districts = district_ids.copy()
    districts["proba_dem"] = proba

    if isinstance(pipeline, VoteShareRegressor):
        pipeline.fit_bootstrap(X, ds.y_vote_share, years, sample_weight=weights)
        ci_lo, ci_hi = pipeline.predict_ci(pred_2026)
        districts["ci_lo"] = ci_lo
        districts["ci_hi"] = ci_hi

    cook = load_cook_ratings(DataFile.COOK)
    if not cook.empty:
        districts = districts.merge(
            cook[[Col.STATE, Col.DISTRICT, Col.COOK_PROB]],
            on=[Col.STATE, Col.DISTRICT],
            how="left",
        )
        has_cook = districts[Col.COOK_PROB].notna()
        districts["proba_dem"] = np.where(
            has_cook,
            (1 - COOK_ALPHA) * districts["proba_dem"]
            + COOK_ALPHA * districts[Col.COOK_PROB].fillna(districts["proba_dem"]),
            districts["proba_dem"],
        )
        districts["cook_rating"] = (
            cook.set_index([Col.STATE, Col.DISTRICT])[Col.COOK_PROB]
            .reindex(pd.MultiIndex.from_arrays([districts[Col.STATE], districts[Col.DISTRICT]]))
            .values
        )

    sim_params = load_simulation_params()
    sim = simulate_blended(
        districts["proba_dem"].to_numpy(),
        sigma_national=sim_params["sigma_national"],
        sigma_district=sim_params["sigma_district"],
        blend_weight=sim_params["blend_weight"],
        concentration=sim_params["concentration"],
    )
    districts["label"] = np.where(
        districts[Col.DISTRICT] == 0,
        districts[Col.STATE].astype(str) + "-AL",
        districts[Col.STATE].astype(str) + "-" + districts[Col.DISTRICT].astype(str),
    )
    incumbents_2024 = winners[winners[Col.YEAR] == 2024][
        [Col.STATE, Col.DISTRICT, Col.PARTY]
    ].rename(columns={Col.PARTY: "district_incumbent"})
    districts = districts.merge(incumbents_2024, on=[Col.STATE, Col.DISTRICT], how="left")
    districts["is_open"] = [
        (state, dist) in OPEN_SEATS_2026
        for state, dist in zip(districts[Col.STATE], districts[Col.DISTRICT])
    ]
    districts["dem_lean"] = pred_2026[Col.DEM_LEAN].values
    if Col.PRES_LEAN in pred_2026.columns:
        districts["pres_lean"] = pred_2026[Col.PRES_LEAN].values

    return {
        "seats_sim": sim,
        "n_districts": len(proba),
        "districts": districts,
        "shap_values": shap_df,
    }
