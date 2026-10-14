import pytest

from src.features import build_dataset
from src.paths import DataFile

_REQUIRED_FOR_BUILD_DATASET = [
    DataFile.HOUSE, DataFile.APPROVAL, DataFile.PRES_08_20, DataFile.PRES_20_24,
]


@pytest.fixture(scope="session")
def real_data():
    # Check all 4 — not just HOUSE — since they're independently obtained (one may now
    # auto-download via `make data`'s GitHub Release mirror while the other 3 stay
    # manual), so "only some of them present" is a real, not just theoretical, state.
    missing = [p.name for p in _REQUIRED_FOR_BUILD_DATASET if not p.exists()]
    if missing:
        pytest.skip(f"data files not present: {missing}")
    ds = build_dataset()
    return {
        "winners": ds.winners,
        "dems_reps": ds.dems_reps,
        "X": ds.X,
        "y": ds.y,
        "years": ds.years,
    }
