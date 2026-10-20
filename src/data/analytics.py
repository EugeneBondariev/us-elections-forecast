"""District-level inertia features derived from historical House vote share.

Uses SQLite window functions (LAG, SQRT) on district-year dem_vote_share data
to compute two lagged features that capture political momentum and competitive
volatility — properties the existing rolling-average features miss.

Data source: data/raw/house_elections.csv (MIT Election Lab, via DataFile.HOUSE).
Run with: make district-analytics

Output
------
data/interim/district_analytics.csv — columns: year, state_po, district,
                                       win_streak, swing_volatility
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.constants import Col
from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "build_analytics_db",
    "compute_win_streak",
    "compute_swing_volatility",
    "build_district_analytics",
    "load_district_analytics",
]

_WIN_STREAK = "win_streak"
_SWING_VOL = "swing_volatility"

_SQL_WIN_STREAK = """
WITH lagged AS (
    SELECT year, state_po, district,
           LAG(CASE WHEN dem_vote_share >= 0.5 THEN 'D' ELSE 'R' END, 1) OVER w AS p1,
           LAG(CASE WHEN dem_vote_share >= 0.5 THEN 'D' ELSE 'R' END, 2) OVER w AS p2,
           LAG(CASE WHEN dem_vote_share >= 0.5 THEN 'D' ELSE 'R' END, 3) OVER w AS p3,
           LAG(CASE WHEN dem_vote_share >= 0.5 THEN 'D' ELSE 'R' END, 4) OVER w AS p4
    FROM results
    WINDOW w AS (PARTITION BY state_po, district ORDER BY year)
)
SELECT year, state_po, district,
    CASE
        WHEN p1 IS NULL THEN 0
        WHEN p4 IS NOT NULL AND p1 = p2 AND p2 = p3 AND p3 = p4 THEN 4
        WHEN p3 IS NOT NULL AND p1 = p2 AND p2 = p3 THEN 3
        WHEN p2 IS NOT NULL AND p1 = p2 THEN 2
        ELSE 1
    END AS win_streak
FROM lagged
"""

_SQL_SWING_VOL = """
WITH lagged AS (
    SELECT year, state_po, district,
           LAG(dem_vote_share, 1) OVER w AS d1,
           LAG(dem_vote_share, 2) OVER w AS d2,
           LAG(dem_vote_share, 3) OVER w AS d3,
           LAG(dem_vote_share, 4) OVER w AS d4
    FROM results
    WINDOW w AS (PARTITION BY state_po, district ORDER BY year)
)
SELECT year, state_po, district,
    CASE
        WHEN d1 IS NOT NULL AND d2 IS NOT NULL AND d3 IS NOT NULL AND d4 IS NOT NULL
        THEN SQRT(
            ((d1 - (d1+d2+d3+d4)/4.0) * (d1 - (d1+d2+d3+d4)/4.0)
           + (d2 - (d1+d2+d3+d4)/4.0) * (d2 - (d1+d2+d3+d4)/4.0)
           + (d3 - (d1+d2+d3+d4)/4.0) * (d3 - (d1+d2+d3+d4)/4.0)
           + (d4 - (d1+d2+d3+d4)/4.0) * (d4 - (d1+d2+d3+d4)/4.0)) / 4.0
        )
        ELSE NULL
    END AS swing_volatility
FROM lagged
"""


def build_analytics_db(dems_reps: pd.DataFrame) -> sqlite3.Connection:
    """Load district-year vote share data into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cols = [Col.YEAR, Col.STATE, Col.DISTRICT, Col.DEM_VOTE_SHARE]
    dems_reps[cols].to_sql(
        name="results",
        con=conn,
        index=False,
        if_exists="replace",
    )
    return conn


def compute_win_streak(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return per-district-year consecutive-wins count for the most recent winner's party.

    Strictly lagged: uses only cycles before the current year. First cycle is 0.
    """
    return pd.read_sql(sql=_SQL_WIN_STREAK, con=conn)


def compute_swing_volatility(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return population std-dev of dem_vote_share over the 4 prior cycles.

    NULL when fewer than 4 prior cycles are available for a district.
    """
    return pd.read_sql(sql=_SQL_SWING_VOL, con=conn)


def build_district_analytics(dems_reps: pd.DataFrame) -> pd.DataFrame:
    """Compute win_streak and swing_volatility for every district-year row."""
    conn = build_analytics_db(dems_reps)
    streak_df = compute_win_streak(conn)
    vol_df = compute_swing_volatility(conn)

    result = streak_df.merge(
        vol_df,
        on=["year", "state_po", "district"],
        how="left",
    )

    # Col.YEAR/STATE/DISTRICT already match — rename is a no-op kept for
    # explicitness and future-proofing if Col values ever diverge.
    result = result.rename(columns={
        "year": "year",          # Col.YEAR
        "state_po": "state_po",  # Col.STATE
        "district": "district",  # Col.DISTRICT
    })

    result["year"] = result["year"].astype(int)
    result["district"] = result["district"].astype(int)

    logger.info(
        "District analytics: %d rows, years %s–%s, districts %s–%s",
        len(result),
        result["year"].min(),
        result["year"].max(),
        result["district"].min(),
        result["district"].max(),
    )
    return result


def load_district_analytics(cache_path: Path | None = None) -> pd.DataFrame | None:
    """Load district analytics from disk, validating with pandera schema."""
    path = cache_path if cache_path is not None else DataFile.DISTRICT_ANALYTICS
    if not path.exists():
        logger.info(
            "District analytics cache not found: %s — skipping analytics features",
            path,
        )
        return None
    df = pd.read_csv(path, dtype={"district": int, "year": int})
    from src.data.validators import validate_district_analytics
    validate_district_analytics(df)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.data.loaders import load_clean_house_data
    from src.features.features import build_dems_reps, get_top_candidate_per_party

    df = load_clean_house_data(DataFile.HOUSE)
    top = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top)
    analytics = build_district_analytics(dems_reps)
    out = ensure(DataFile.DISTRICT_ANALYTICS)
    analytics.to_csv(out, index=False)
    logger.info("District analytics saved → %s (%d rows)", out, len(analytics))


if __name__ == "__main__":
    main()
