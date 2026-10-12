import pytest

from src.features import (
    build_dataset,
    build_dems_reps,
    build_feature_matrix,
    get_top_candidate_per_party,
)
from src.loaders import load_clean_house_data
from src.paths import DataFile


@pytest.fixture(scope="session")
def real_data():
    if not DataFile.HOUSE.exists():
        pytest.skip("data files not present")
    X, y, winners = build_dataset()
    _, _, years = build_feature_matrix(winners)
    df = load_clean_house_data(DataFile.HOUSE)
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    return {
        "winners": winners,
        "dems_reps": dems_reps,
        "X": X,
        "y": y,
        "years": years,
    }
