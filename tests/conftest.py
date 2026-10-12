import pytest

from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    build_winners,
    get_top_candidate_per_party,
)
from src.loaders import load_approval, load_clean_house_data
from src.paths import DataFile


@pytest.fixture(scope="session")
def real_data():
    if not DataFile.HOUSE.exists():
        pytest.skip("data files not present")
    df = load_clean_house_data(DataFile.HOUSE)
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(DataFile.APPROVAL)
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
