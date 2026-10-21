"""Standalone data quality validator.

Loads every downloaded file through its schema and reports [OK]/[SKIP]/[FAIL].
Optional files (cook, fundraising) report [SKIP] rather than [FAIL] when absent.

Run:
    python -m src.data.validate_data
or:
    make validate
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.constants import Col
from src.data.loaders import load_approval, load_clean_house_data, load_pres_lean
from src.data.validators import validate_cook_csv
from src.data.validators import validate_fundraising as _validate_fundraising_schema
from src.logging_config import log_fail, log_ok, log_skip, setup_cli_logging
from src.paths import DataFile

logger = logging.getLogger(__name__)

__all__ = ["validate_all", "validate_required", "validate_cook", "validate_fundraising"]


def _validate_file(
    label: str,
    paths: list[Path],
    skip_reason: str,
    load: Callable[[], pd.DataFrame],
    ok_detail: Callable[[pd.DataFrame], str],
    min_rows: int = 0,
) -> bool:
    """Check existence, run load(), log [OK]/[SKIP]/[FAIL]. Returns True if passed or absent."""
    if not all(p.exists() for p in paths):
        log_skip(label, skip_reason)
        return True
    try:
        df = load()
        if min_rows and len(df) < min_rows:
            raise ValueError(f"only {len(df):,} rows, expected {min_rows:,}+")
        log_ok(label, ok_detail(df))
        return True
    except Exception as exc:
        log_fail(label, exc)
        return False


def _load_cook() -> pd.DataFrame:
    df = pd.read_csv(DataFile.COOK, dtype={Col.DISTRICT: int})
    validate_cook_csv(df)
    return df


def _load_fundraising() -> pd.DataFrame:
    df = pd.read_csv(DataFile.FUNDRAISING)
    _validate_fundraising_schema(df)
    return df


def validate_cook() -> bool:
    return _validate_file(
        label="cook ratings",
        paths=[DataFile.COOK],
        skip_reason="optional, not present",
        load=_load_cook,
        ok_detail=lambda df: f"{len(df)} districts",
    )


def validate_fundraising() -> bool:
    return _validate_file(
        label="fundraising",
        paths=[DataFile.FUNDRAISING],
        skip_reason="optional, not present",
        load=_load_fundraising,
        ok_detail=lambda df: (
            f"{len(df):,} district-year rows, {int(df['year'].min())}-{int(df['year'].max())}"
        ),
    )


def validate_required() -> bool:
    """Validate the required files only (house, approval, pres lean)."""
    return all(
        [
            _validate_file(
                label="house results",
                paths=[DataFile.HOUSE],
                skip_reason="not downloaded",
                load=lambda: load_clean_house_data(DataFile.HOUSE),
                ok_detail=lambda df: (
                    f"{len(df):,} rows, years {df['year'].min()}-{df['year'].max()}"
                ),
                min_rows=20_000,
            ),
            _validate_file(
                label="approval ratings",
                paths=[DataFile.APPROVAL],
                skip_reason="not downloaded",
                load=lambda: load_approval(DataFile.APPROVAL),
                ok_detail=lambda df: f"{len(df)} election years",
                min_rows=20,
            ),
            _validate_file(
                label="presidential lean",
                paths=[DataFile.PRES_08_20, DataFile.PRES_20_24],
                skip_reason="not downloaded",
                load=lambda: load_pres_lean(DataFile.PRES_08_20, DataFile.PRES_20_24),
                ok_detail=lambda df: f"{len(df):,} district-year rows",
            ),
        ]
    )


def validate_all() -> bool:
    """Validate all downloaded files. Returns True if all present files pass."""
    results = [validate_required(), validate_cook(), validate_fundraising()]
    ok = all(results)
    logger.info("")
    if ok:
        logger.info("All files valid.")
    else:
        logger.error("Validation failed - see above.")
    return ok


def main() -> None:
    setup_cli_logging()
    sys.exit(not validate_all())


if __name__ == "__main__":
    main()
