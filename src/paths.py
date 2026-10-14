"""Central registry of data file paths.

All paths are absolute (derived from this file's location) so scripts work
regardless of the working directory they are launched from.

Follows the Cookiecutter Data Science data/ convention, so "is it safe to hand-edit
this file" is a property of which directory it lives in, not something you have to
remember:

    data/raw/        Primary source files, exactly as downloaded/manually assembled.
                      Immutable — never edit in place; fix a mistake by re-fetching
                      or re-copying, not by hand-editing the file.
    data/external/   Third-party supplementary data (Cook ratings page, raw FEC API
                      records) — same immutability rule, just not the core dataset.
    data/interim/    First-pass transformations of raw/external data (parsed,
                      aggregated) that aren't yet final model-ready features.
    data/processed/  Derived, regenerable artifacts the app/API read at runtime
                      (forecast cache, RAG index) — safe to delete and rebuild.
    models/          Trained/tuned model artifacts — not "data" in the DS sense.
"""
from pathlib import Path

__all__ = [
    "DATA_DIR", "RAW_DIR", "EXTERNAL_DIR", "INTERIM_DIR", "PROCESSED_DIR",
    "MODELS_DIR", "OUTPUTS_DIR", "DataFile",
]

_ROOT = Path(__file__).parent.parent

DATA_DIR      = _ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
EXTERNAL_DIR  = DATA_DIR / "external"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = _ROOT / "models"
OUTPUTS_DIR   = _ROOT / "outputs"

# Created on import, not left to .gitkeep files or any one entry point, because
# every script/test/the app/the API all import this module — so the directories
# exist before anything else runs, regardless of whether `make` is installed or
# scripts/fetch_data.py has been run yet. Cheap and idempotent (exist_ok=True).
for _d in (RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class DataFile:
    # raw/ — primary source files, manually downloaded/assembled. Immutable.
    HOUSE      = RAW_DIR / "1976-2024-house.tab"
    APPROVAL   = RAW_DIR / "Approval-Ratings-for-POTUS-raw.xls"
    PRES_08_20 = RAW_DIR / "president-2008-2020-by-district.csv"
    PRES_20_24 = RAW_DIR / "president-2020-2024-by-district.csv"

    # external/ — third-party supplementary data. Same immutability rule as raw/.
    COOK_HTML       = EXTERNAL_DIR / "cook.html"
    FUNDRAISING_RAW = EXTERNAL_DIR / "fundraising_raw.csv"

    # interim/ — first-pass transformations of raw/external data.
    COOK        = INTERIM_DIR / "cook.csv"
    FUNDRAISING = INTERIM_DIR / "fundraising.csv"

    # processed/ — derived artifacts the app/API read at runtime. Safe to delete.
    DB     = PROCESSED_DIR / "elections.db"
    CHROMA = PROCESSED_DIR / "chroma_db"

    # models/ — trained/tuned model artifacts.
    BEST_PARAMS      = MODELS_DIR / "best_params_lgbm.json"
    CALIBRATED_SIGMA = MODELS_DIR / "calibrated_sigma.json"
