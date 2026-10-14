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

import pandas as pd

from src.paths import DataFile

logger = logging.getLogger(__name__)

__all__ = ["validate_all", "validate_required", "validate_cook", "validate_fundraising"]


def _ok(label: str, detail: str = "") -> None:
    logger.info("[OK]   %s%s", label, f" — {detail}" if detail else "")


def _skip(label: str, reason: str) -> None:
    logger.info("[SKIP] %s — %s", label, reason)


def _fail(label: str, exc: Exception) -> None:
    logger.error("[FAIL] %s — %s", label, exc)


def _validate_house() -> bool:
    if not DataFile.HOUSE.exists():
        _skip("house results", "not downloaded")
        return True
    try:
        from src.data.loaders import load_clean_house_data
        df = load_clean_house_data(DataFile.HOUSE)
        if len(df) < 20_000:
            raise ValueError(f"only {len(df):,} rows, expected 20 000+")
        _ok("house results", f"{len(df):,} rows, years {df['year'].min()}-{df['year'].max()}")
        return True
    except Exception as exc:
        _fail("house results", exc)
        return False


def _validate_approval() -> bool:
    if not DataFile.APPROVAL.exists():
        _skip("approval ratings", "not downloaded")
        return True
    try:
        from src.data.loaders import load_approval
        df = load_approval(DataFile.APPROVAL)
        if len(df) < 20:
            raise ValueError(f"only {len(df)} rows, expected 20+")
        _ok("approval ratings", f"{len(df)} election years")
        return True
    except Exception as exc:
        _fail("approval ratings", exc)
        return False


def _validate_pres_lean() -> bool:
    if not DataFile.PRES_08_20.exists() or not DataFile.PRES_20_24.exists():
        _skip("presidential lean", "not downloaded")
        return True
    try:
        from src.data.loaders import load_pres_lean
        from src.data.validators import validate_pres_lean
        df = load_pres_lean(DataFile.PRES_08_20, DataFile.PRES_20_24)
        validate_pres_lean(df)
        _ok("presidential lean", f"{len(df):,} district-year rows")
        return True
    except Exception as exc:
        _fail("presidential lean", exc)
        return False


def validate_cook() -> bool:
    if not DataFile.COOK.exists():
        _skip("cook ratings", "optional, not present")
        return True
    try:
        from src.data.validators import validate_cook_csv
        df = pd.read_csv(DataFile.COOK, dtype={"district": int})
        validate_cook_csv(df)
        _ok("cook ratings", f"{len(df)} districts")
        return True
    except Exception as exc:
        _fail("cook ratings", exc)
        return False


def validate_fundraising() -> bool:
    if not DataFile.FUNDRAISING.exists():
        _skip("fundraising", "optional, not present")
        return True
    try:
        from src.data.validators import validate_fundraising as _schema_check
        df = pd.read_csv(DataFile.FUNDRAISING)
        _schema_check(df)
        _ok("fundraising", f"{len(df):,} district-year rows, {int(df['year'].min())}-{int(df['year'].max())}")
        return True
    except Exception as exc:
        _fail("fundraising", exc)
        return False


def validate_required() -> bool:
    """Validate the required files only (house, approval, pres lean)."""
    results = [_validate_house(), _validate_approval(), _validate_pres_lean()]
    return all(results)


def validate_all() -> bool:
    """Validate all downloaded files. Returns True if all present files pass."""
    results = [
        _validate_house(),
        _validate_approval(),
        _validate_pres_lean(),
        validate_cook(),
        validate_fundraising(),
    ]
    ok = all(results)
    logger.info("")
    if ok:
        logger.info("All files valid.")
    else:
        logger.error("%d file(s) failed validation.", sum(1 for r in results if not r))
    return ok


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    sys.exit(0 if validate_all() else 1)


if __name__ == "__main__":
    main()
