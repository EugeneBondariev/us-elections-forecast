"""Polsby-Popper compactness scores from Census TIGER/Line shapefiles.

Gerrymandered districts have fragmented boundaries that inflate perimeter without
adding area, lowering Polsby-Popper (PP) scores. PP = 4π × area / perimeter²,
ranging 0–1 where 1 is a perfect circle (most compact). Districts with very low
PP scores (< 0.1) often have strong partisan lean baked in through line-drawing.

Data source: Census TIGER/Line 118th Congress shapefiles (free, no API key).
URL: https://www2.census.gov/geo/tiger/TIGER2022/CD/tl_2022_us_cd118.zip

Run with: make gerrymandering

Output: data/interim/gerrymandering.csv — state_po, district, polsby_popper,
        district_area_km2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_tiger_shapefile",
    "parse_tiger_shapefile",
    "build_compactness_features",
    "load_gerrymandering",
]

_TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/CD/tl_2022_us_cd118.zip"
_CRS_EQUAL_AREA = "EPSG:5070"  # CONUS Albers Equal Area — accurate area/perimeter on land
_MISSING_DISTRICT = "ZZ"  # sentinel for territories / non-voting delegates to drop


def fetch_tiger_shapefile(cache_path: Path | None = None) -> Path:
    """Download the TIGER CD shapefile zip, skipping if already cached."""
    out = cache_path or DataFile.TIGER_SHAPEFILE
    if Path(out).exists():
        logger.info("TIGER shapefile cache hit: %s", out)
        return Path(out)
    logger.info("Downloading Census TIGER CD shapefile → %s", out)
    resp = requests.get(_TIGER_URL, stream=True, timeout=120)
    resp.raise_for_status()
    ensure(out).write_bytes(resp.content)
    logger.info("Downloaded %d bytes", Path(out).stat().st_size)
    return Path(out)


def parse_tiger_shapefile(zip_path: Path) -> Any:
    """Read the TIGER zip directly into a GeoDataFrame via geopandas zip:// prefix."""
    import geopandas as gpd

    gdf = gpd.read_file(f"zip://{zip_path}")
    return gdf


def build_compactness_features(gdf: Any) -> pd.DataFrame:
    """Compute Polsby-Popper compactness and district area from TIGER geometry."""
    import numpy as np

    from src.constants import STATE_FIPS_TO_PO

    # Project to equal-area CRS for accurate geometry measurements
    projected = gdf.to_crs(_CRS_EQUAL_AREA)

    area_m2 = projected.geometry.area
    perimeter_m = projected.geometry.length

    polsby_popper = (4 * np.pi * area_m2) / (perimeter_m**2)
    area_km2 = area_m2 / 1_000_000

    # TIGER attributes: STATEFP (2-digit FIPS), CD118FP (2-digit district number)
    df = pd.DataFrame(
        {
            "statefp": gdf["STATEFP"],
            "cd_fp": gdf["CD118FP"],
            "polsby_popper": polsby_popper.values,
            "district_area_km2": area_km2.values,
        }
    )

    # Map FIPS → postal abbreviation; drop territories (DC, PR, VI, etc.)
    df["state_po"] = df["statefp"].map(STATE_FIPS_TO_PO)
    df = df[df["state_po"].notna()].copy()

    # Parse district: "00" → 0 (at-large), "01" → 1, etc.
    df["district"] = df["cd_fp"].apply(lambda x: int(x.strip()) if str(x).strip().isdigit() else 0)

    result = df[["state_po", "district", "polsby_popper", "district_area_km2"]].copy()
    result["polsby_popper"] = result["polsby_popper"].clip(lower=0.0, upper=1.0)
    result = result.sort_values(["state_po", "district"]).reset_index(drop=True)

    logger.info(
        "Compactness features: %d districts, PP range [%.3f, %.3f]",
        len(result),
        result["polsby_popper"].min(),
        result["polsby_popper"].max(),
    )
    return result


def load_gerrymandering(cache_path: Path | None = None) -> pd.DataFrame | None:
    """Load gerrymandering features from disk, validating with pandera schema."""
    path = cache_path or DataFile.GERRYMANDERING
    if not Path(path).exists():
        logger.info("gerrymandering.csv absent — compactness features will be skipped")
        return None
    df = pd.read_csv(path, dtype={"district": int})
    from src.data.validators import validate_gerrymandering

    validate_gerrymandering(df)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    zip_path = fetch_tiger_shapefile(cache_path=DataFile.TIGER_SHAPEFILE)
    gdf = parse_tiger_shapefile(zip_path=zip_path)
    features = build_compactness_features(gdf=gdf)
    features.to_csv(ensure(DataFile.GERRYMANDERING), index=False)
    logger.info(
        "Gerrymandering features saved → %s (%d districts)",
        DataFile.GERRYMANDERING,
        len(features),
    )


if __name__ == "__main__":
    main()
