"""Unit tests for src/data/gerrymandering.py. No network access, no disk I/O."""

from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

from src.data.gerrymandering import build_compactness_features

_GEOPANDAS_AVAILABLE = importlib.util.find_spec("geopandas") is not None
_skip_no_geopandas = pytest.mark.skipif(
    not _GEOPANDAS_AVAILABLE,
    reason="geopandas not installed",
)


# ---------------------------------------------------------------------------
# Synthetic GeoDataFrame helper
# ---------------------------------------------------------------------------


def _make_gdf(n: int = 4) -> "GeoDataFrame":
    """Build a minimal GeoDataFrame with square districts for testing.

    n square districts in a row, 1°×1° each in geographic coords (EPSG:4326).
    All assigned to California (FIPS "06") with sequential district codes.
    """
    import geopandas as gpd
    from shapely.geometry import box

    geometries = [box(i, 0, i + 1, 1) for i in range(n)]
    state_fps = ["06"] * n
    cd_fps = [f"{i + 1:02d}" for i in range(n)]
    return gpd.GeoDataFrame(
        {"STATE": state_fps, "CD119": cd_fps},
        geometry=geometries,
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# TestBuildCompactnessFeatures
# ---------------------------------------------------------------------------


@_skip_no_geopandas
class TestBuildCompactnessFeatures:
    def test_output_columns(self) -> None:
        gdf = _make_gdf()
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        assert list(result.columns) == [
            "state_po",
            "district",
            "polsby_popper",
            "district_area_km2",
        ]

    def test_polsby_popper_range(self) -> None:
        gdf = _make_gdf()
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        assert (result["polsby_popper"] >= 0.0).all()
        assert (result["polsby_popper"] <= 1.0).all()

    def test_square_has_known_pp(self) -> None:
        """A perfect square has PP = π/4 ≈ 0.785."""
        gdf = _make_gdf(n=1)
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        pp = result["polsby_popper"].iloc[0]
        assert pp == pytest.approx(math.pi / 4, abs=0.05)

    def test_district_dtype(self) -> None:
        gdf = _make_gdf()
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        assert pd.api.types.is_integer_dtype(result["district"])

    def test_drops_territories(self) -> None:
        """Puerto Rico (FIPS "72") is not in STATE_FIPS_TO_PO and must be dropped."""
        import geopandas as gpd
        from shapely.geometry import box

        # Mix: one CA district + one PR district
        geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
        gdf = gpd.GeoDataFrame(
            {"STATE": ["06", "72"], "CD119": ["01", "01"]},
            geometry=geometries,
            crs="EPSG:4326",
        )
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        assert "PR" not in result["state_po"].values
        assert len(result) == 1

    def test_row_count_matches_valid_districts(self) -> None:
        gdf = _make_gdf(n=4)
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        assert len(result) == 4

    def test_sorted_by_state_district(self) -> None:
        gdf = _make_gdf(n=3)
        result = build_compactness_features(gdf=gdf, fips_map={"06": "CA"})
        pairs = list(zip(result["state_po"], result["district"]))
        assert pairs == sorted(pairs)
