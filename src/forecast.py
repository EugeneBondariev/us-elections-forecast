import numpy as np
import pandas as pd

from src.constants import COOK_ALPHA, OPEN_SEATS_2026, Col
from src.features import build_dataset, build_pred_2026
from src.loaders import load_cook_ratings
from src.model import (
    compute_sample_weights,
    explain_districts,
    load_simulation_params,
    simulate_election,
    train,
)
from src.paths import DataFile

__all__ = ["run_forecast"]


def run_forecast() -> dict:
    """Run the full 2026 House forecast pipeline.

    Returns a dict with:
      seats_sim   – np.ndarray of 10k simulated total Dem seat counts
      n_districts – number of contested districts
      districts   – DataFrame with per-district probabilities and metadata
    """
    ds = build_dataset()
    X, y, years, winners = ds.X, ds.y, ds.years, ds.winners
    pres_lean, fundraising, dems_reps = ds.pres_lean, ds.fundraising, ds.dems_reps

    pipeline = train(X, y, sample_weight=compute_sample_weights(years))
    pred_2026, district_ids = build_pred_2026(
        winners, dems_reps, X.columns, pres_lean=pres_lean, fundraising=fundraising
    )
    proba = pipeline.predict_proba(pred_2026)[:, 1]
    shap_df = explain_districts(pipeline, pred_2026)

    districts = district_ids.copy()
    districts["proba_dem"] = proba

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
    sim = simulate_election(
        districts["proba_dem"].values,
        sigma_national=sim_params["sigma_national"],
        sigma_district=sim_params["sigma_district"],
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
