from pathlib import Path

import pytest

from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    get_top_candidate_per_party,
    build_winners,
)
from src.loaders import load_approval, load_clean_house_data

DATA_DIR = Path("data")


@pytest.fixture(scope="session")
def real_data():
    if not (DATA_DIR / "1976-2024-house.tab").exists():
        pytest.skip("data files not present")
    df = load_clean_house_data(DATA_DIR / "1976-2024-house.tab")
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls")
    winners = build_winners(top_candidates, dems_reps, approval)
    winners = add_incumbency_features(winners, df)
    X, y, years = build_feature_matrix(winners)
    return {
        "df": df,
        "top_candidates": top_candidates,
        "dems_reps": dems_reps,
        "winners": winners,
        "X": X,
        "y": y,
        "years": years,
    }
