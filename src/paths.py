"""Central registry of data file paths.

All paths are absolute (derived from this file's location) so scripts work
regardless of the working directory they are launched from.
"""
from pathlib import Path

__all__ = ["DATA_DIR", "OUTPUTS_DIR", "DataFile"]

_ROOT     = Path(__file__).parent.parent
DATA_DIR  = _ROOT / "data"
OUTPUTS_DIR = _ROOT / "outputs"


class DataFile:
    HOUSE       = DATA_DIR / "1976-2024-house.tab"
    APPROVAL    = DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls"
    PRES_08_20  = DATA_DIR / "president-2008-2020-by-district.csv"
    PRES_20_24  = DATA_DIR / "president-2020-2024-by-district.csv"
    COOK        = DATA_DIR / "cook.csv"
    FUNDRAISING = DATA_DIR / "fundraising.csv"
    DB          = DATA_DIR / "elections.db"
    CHROMA      = DATA_DIR / "chroma_db"
    BEST_PARAMS = DATA_DIR / "best_params_lgbm.json"
