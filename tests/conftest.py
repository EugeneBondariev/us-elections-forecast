import pytest

from src.features import build_dataset
from src.paths import DataFile


@pytest.fixture(scope="session")
def real_data():
    if not DataFile.HOUSE.exists():
        pytest.skip("data files not present")
    ds = build_dataset()
    return {
        "winners": ds.winners,
        "dems_reps": ds.dems_reps,
        "X": ds.X,
        "y": ds.y,
        "years": ds.years,
    }
