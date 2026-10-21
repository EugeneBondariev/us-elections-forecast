"""Polsby-Popper compactness scores from Census TIGERweb REST API.

Gerrymandered districts have fragmented boundaries that inflate perimeter without
adding area, lowering Polsby-Popper (PP) scores. PP = 4π × area / perimeter²,
ranging 0–1 where 1 is a perfect circle (most compact). Districts with very low
PP scores (< 0.1) often have strong partisan lean baked in through line-drawing.

Data source: Census TIGERweb REST API — 119th Congress congressional districts
(layer 4). The 119th Congress uses the same post-2020-census boundaries as the
118th; these are the districts contested in the 2026 elections.
URL: https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Legislative/MapServer/4/query

Run with: make gerrymandering

Output: data/interim/gerrymandering.csv — state_po, district, polsby_popper,
        district_area_km2
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_tiger_geodataframe",
    "build_compactness_features",
    "load_gerrymandering",
]

_TIGERWEB_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Legislative/MapServer/4/query"
)
_CRS_EQUAL_AREA = "EPSG:5070"  # CONUS Albers Equal Area — accurate area/perimeter on land
# WAF blocks responses larger than ~50 districts' worth of detailed geometry; paginate below this.
_BATCH_SIZE = 50


def fetch_tiger_geodataframe(cache_path: Path | None = None) -> Any:
    """Fetch 119th Congress district boundaries from TIGERweb REST API.

    Paginates in batches of 50 (WAF blocks larger geometry responses).
    Results are cached as GeoJSON at cache_path to avoid redundant API calls.
    """
    import geopandas as gpd

    cache = cache_path or DataFile.TIGER_GEOJSON
    if Path(cache).exists():
        logger.info("TIGERweb cache hit: %s", cache)
        return gpd.read_file(str(cache))

    logger.info("Fetching 119th Congress district boundaries from TIGERweb API")
    all_features: list[dict] = []
    offset = 0
    while True:
        params: dict[str, str | int] = {
            "where": "AREALAND IS NOT NULL",
            "outFields": "STATE,CD119",
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": _BATCH_SIZE,
            "resultOffset": offset,
        }
        resp = requests.get(_TIGERWEB_URL, params=params, timeout=120)
        resp.raise_for_status()
        if "application/geo+json" not in resp.headers.get("content-type", ""):
            raise RuntimeError(
                f"TIGERweb WAF rejection at offset={offset}. "
                f"Content-type: {resp.headers.get('content-type')}. "
                f"Response: {resp.text[:200]}"
            )
        page = resp.json()
        batch = page.get("features", [])
        all_features.extend(batch)
        logger.info(
            "Fetched %d districts (offset=%d, total=%d)", len(batch), offset, len(all_features)
        )
        if len(batch) < _BATCH_SIZE:
            break
        offset += _BATCH_SIZE

    geojson_text = json.dumps({"type": "FeatureCollection", "features": all_features})
    gdf = gpd.read_file(io.StringIO(geojson_text))

    ensure(Path(cache)).write_text(geojson_text, encoding="utf-8")
    logger.info(
        "TIGERweb GeoJSON cached → %s (%d districts, %d bytes)",
        cache,
        len(all_features),
        len(geojson_text),
    )
    return gdf


def build_compactness_features(gdf: Any) -> pd.DataFrame:
    """Compute Polsby-Popper compactness and district area from TIGERweb geometry."""
    import numpy as np

    from src.constants import STATE_FIPS_TO_PO

    # Project to equal-area CRS for accurate geometry measurements
    projected = gdf.to_crs(_CRS_EQUAL_AREA)

    area_m2 = projected.geometry.area
    perimeter_m = projected.geometry.length

    polsby_popper = (4 * np.pi * area_m2) / (perimeter_m**2)
    area_km2 = area_m2 / 1_000_000

    # TIGERweb API fields: STATE (2-digit FIPS), CD119 (2-digit district number)
    df = pd.DataFrame(
        {
            "statefp": gdf["STATE"],
            "cd_fp": gdf["CD119"],
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
    gdf = fetch_tiger_geodataframe(cache_path=DataFile.TIGER_GEOJSON)
    features = build_compactness_features(gdf=gdf)
    features.to_csv(ensure(DataFile.GERRYMANDERING), index=False)
    logger.info(
        "Gerrymandering features saved → %s (%d districts)",
        DataFile.GERRYMANDERING,
        len(features),
    )


if __name__ == "__main__":
    main()
