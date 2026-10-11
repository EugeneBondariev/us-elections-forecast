import os
from pathlib import Path

from src.features import (
    Col,
    OPEN_SEATS_2026,
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    build_pred_2026,
    build_winners,
    get_top_candidate_per_party,
    load_approval,
    load_clean_house_data,
    load_pres_lean,
)
from src.fundraising import load_fundraising
from src.model import simulate_election, train


def run_forecast(data_dir: Path = Path("data")) -> dict:
    """Run the full 2026 House forecast pipeline.

    Returns a dict with:
      seats_sim   – np.ndarray of 10k simulated total Dem seat counts
      n_districts – number of contested districts
      districts   – DataFrame with per-district probabilities and metadata
    """
    data_dir = Path(data_dir)
    df = load_clean_house_data(data_dir / "1976-2024-house.tab")
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(data_dir / "Approval-Ratings-for-POTUS-raw.xls")
    pres_lean = load_pres_lean(
        data_dir / "president-2008-2020-by-district.csv",
        data_dir / "president-2020-2024-by-district.csv",
    )
    fundraising = load_fundraising(
        cache_path=data_dir / "fec_fundraising.csv",
        api_key=os.getenv("FEC_API_KEY"),
    )
    winners = build_winners(top_candidates, dems_reps, approval, pres_lean, fundraising)
    winners = add_incumbency_features(winners, df)
    X, y, _ = build_feature_matrix(winners)

    pipeline = train(X, y)
    pred_2026, district_ids = build_pred_2026(
        winners, dems_reps, X.columns, pres_lean=pres_lean, fundraising=fundraising
    )
    proba = pipeline.predict_proba(pred_2026)[:, 1]
    sim = simulate_election(proba)

    districts = district_ids.copy()
    districts["proba_dem"] = proba
    districts["label"] = districts.apply(
        lambda r: f"{r[Col.STATE]}-{'AL' if r[Col.DISTRICT] == 0 else r[Col.DISTRICT]}",
        axis=1,
    )
    incumbents_2024 = winners[winners[Col.YEAR] == 2024][
        [Col.STATE, Col.DISTRICT, Col.PARTY]
    ].rename(columns={Col.PARTY: "district_incumbent"})
    districts = districts.merge(
        incumbents_2024, on=[Col.STATE, Col.DISTRICT], how="left"
    )
    districts["is_open"] = districts.apply(
        lambda r: (r[Col.STATE], r[Col.DISTRICT]) in OPEN_SEATS_2026, axis=1
    )
    districts["dem_lean"] = pred_2026[Col.DEM_LEAN].values
    if Col.PRES_LEAN in pred_2026.columns:
        districts["pres_lean"] = pred_2026[Col.PRES_LEAN].values

    return {
        "seats_sim": sim,
        "n_districts": len(proba),
        "districts": districts,
    }
