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
    reports/         Generated analysis output (text/csv) — safe to delete and rebuild.
    reports/figures/ Generated plots/images — safe to delete and rebuild.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "DATA_DIR", "RAW_DIR", "EXTERNAL_DIR", "INTERIM_DIR", "PROCESSED_DIR",
    "MODELS_DIR", "REPORTS_DIR", "FIGURES_DIR", "DataFile",
]

_ROOT = Path(__file__).parent.parent

DATA_DIR      = _ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
EXTERNAL_DIR  = DATA_DIR / "external"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = _ROOT / "models"
REPORTS_DIR   = _ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"

# Created on import, not left to .gitkeep files or any one entry point, because
# every script/test/the app/the API all import this module — so the directories
# exist before anything else runs, regardless of whether `make` is installed or
# src/data/fetch_data.py has been run yet. Cheap and idempotent (exist_ok=True).
for _d in (RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class DataFile:
    # raw/ — primary source files, exactly as downloaded, zero transformation
    # applied. Immutable.
    HOUSE      = RAW_DIR / "1976-2024-house.tab"
    PRES_08_20 = RAW_DIR / "president-2008-2020-by-district.csv"
    PRES_20_24 = RAW_DIR / "president-2020-2024-by-district.csv"

    # external/ — third-party supplementary data. Same immutability rule as raw/.
    FUNDRAISING_RAW         = EXTERNAL_DIR / "fundraising_raw.csv"
    GENERIC_BALLOT_RAW      = EXTERNAL_DIR / "generic_ballot_averages_raw.csv"

    # interim/ — first-pass transformations of raw/external data. APPROVAL belongs
    # here, not raw/, despite the filename: src/data/build_approval_history.py pools
    # 9 external per-president pages and *selects* (closest-before-election,
    # staleness-filtered) rather than passing anything through untouched - the
    # same shape of transformation as cook.html -> cook.csv, just with a more
    # involved selection step.
    APPROVAL      = INTERIM_DIR / "Approval-Ratings-for-POTUS-raw.xls"
    COOK          = INTERIM_DIR / "cook.csv"
    FUNDRAISING   = INTERIM_DIR / "fundraising.csv"
    INDICATORS       = INTERIM_DIR / "indicators.csv"
    GENERIC_BALLOT_CSV = INTERIM_DIR / "generic_ballot.csv"
    OPEN_SEATS_CSV   = INTERIM_DIR / "open_seats_2026.csv"

    # processed/ — derived artifacts the app/API read at runtime. Safe to delete.
    DB     = PROCESSED_DIR / "elections.db"
    CHROMA = PROCESSED_DIR / "chroma_db"

    # models/ — trained model artifacts (gitignored, regenerable).
    CALIBRATED_SIGMA = MODELS_DIR / "calibrated_sigma.json"

    # config/ — committed tuned hyperparameters (not gitignored, reproducible).
    BEST_PARAMS = _ROOT / "config" / "best_params_lgbm.json"
