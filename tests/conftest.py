from pathlib import Path

import pytest

from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    build_party_votes,
    build_winners,
    load_approval,
    load_house_data,
)

DATA_DIR = Path("data")


@pytest.fixture(scope="session")
def real_data():
    if not (DATA_DIR / "1976-2024-house.tab").exists():
        pytest.skip("data files not present")
    df = load_house_data(DATA_DIR / "1976-2024-house.tab")
    party_votes = build_party_votes(df)
    dems_reps = build_dems_reps(party_votes)
    approval = load_approval(DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls")
    winners = build_winners(party_votes, dems_reps, approval)
    winners = add_incumbency_features(winners, df)
    X, y = build_feature_matrix(winners)
    return {
        "df": df,
        "party_votes": party_votes,
        "dems_reps": dems_reps,
        "winners": winners,
        "X": X,
        "y": y,
    }
