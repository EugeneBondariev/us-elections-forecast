"""US Census ACS 5-year demographic features per congressional district.

Teaches: Census Bureau REST API, FIPS code mapping, derived feature computation
from survey microdata, log-transformation for skewed income distributions, and
cross-source geographic joins.

Demographics are the primary driver of long-run partisan lean shifts after
redistricting; adding them gives the model district-level context it cannot
infer from election history alone (especially new or heavily-redrawn districts).

Data source
-----------
American Community Survey (ACS) 5-year estimates, most recent completed cycle.
API key required. Register free at https://api.census.gov/data/key_signup.html
then set CENSUS_API_KEY=<your_key> in the environment before running.

Fetch with: make census

Output
------
data/interim/census_acs.csv  —  state_po, district, pct_bachelor_plus, pct_non_white,
                                 median_income, median_age
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.constants import Col
from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_census_acs",
    "parse_census_response",
    "build_census_features",
    "load_census",
]

_BASE_URL = "https://api.census.gov/data"
_DATASET = "acs/acs5"
_DEFAULT_YEAR = 2022  # most recent completed ACS cycle

# ACS variable codes and short names
_VARS: dict[str, str] = {
    "B15003_022E": "bachelor",  # bachelor's degree (age 25+)
    "B15003_023E": "master",  # master's degree
    "B15003_024E": "professional",  # professional school degree
    "B15003_025E": "doctorate",  # doctorate degree
    "B15003_001E": "educ_total",  # total pop 25+ (education denominator)
    "B19013_001E": "median_income",  # median household income ($)
    "B02001_001E": "pop_total",  # total population
    "B02001_002E": "pop_white",  # white alone
    "B01002_001E": "median_age",  # median age
}
_MISSING_SENTINEL = -666666666  # Census API uses this for suppressed/missing


def _parse_district_code(code: str) -> int:
    try:
        val = int(code.strip())
    except (ValueError, AttributeError):
        return 0
    return val if val != 0 else 0


def fetch_census_acs(
    year: int = _DEFAULT_YEAR,
    api_key: str | None = None,
    cache_path: Path | None = None,
) -> list[list[str]]:
    """Fetch raw ACS data from the Census Bureau API, with optional file caching."""
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    params: dict[str, str] = {
        "get": ",".join(_VARS.keys()),
        "for": "congressional district:*",
        "in": "state:*",
    }
    if api_key is not None:
        params["key"] = api_key

    resp = requests.get(
        f"{_BASE_URL}/{year}/{_DATASET}",
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "html" in content_type or resp.text.lstrip().startswith("<"):
        if "Invalid Key" in resp.text:
            raise ValueError(
                "Census API rejected the key — it may not be activated yet. "
                "Check your email for a confirmation link from api.census.gov "
                "and click it before retrying."
            )
        raise ValueError(
            "Census API returned HTML instead of JSON — an API key is required. "
            "Register free at https://api.census.gov/data/key_signup.html "
            "then set CENSUS_API_KEY=<your_key> and re-run."
        )
    raw: list[list[str]] = resp.json()

    if cache_path is not None:
        out = ensure(Path(cache_path))
        with open(out, "w") as f:
            json.dump(raw, f)

    logger.info("Fetched %d districts from Census ACS %d", len(raw) - 1, year)
    return raw


def parse_census_response(raw: list[list[str]], fips_map: dict[str, str]) -> pd.DataFrame:
    """Convert the raw Census API list-of-lists into a clean DataFrame."""
    header, *rows = raw
    df = pd.DataFrame(data=rows, columns=header)

    # Rename ACS variable codes → short names; leave "state" and "congressional district"
    df = df.rename(columns=_VARS)

    # Map FIPS code → postal abbreviation; drop territories and DC
    df[Col.STATE] = df["state"].map(fips_map)
    df = df[df[Col.STATE].notna()].copy()

    # Parse district code: "00" → 0 (at-large), "12" → 12
    df[Col.DISTRICT] = df["congressional district"].apply(_parse_district_code)

    # Convert ACS value columns to numeric and replace missing sentinel with NaN
    for col in _VARS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(_MISSING_SENTINEL, float("nan"))

    df = df.drop(columns=["congressional district", "state"])
    return df


def build_census_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived demographic features from the parsed ACS DataFrame."""
    df = raw_df.copy()

    degree_sum = df["bachelor"] + df["master"] + df["professional"] + df["doctorate"]
    df[Col.PCT_BACHELOR_PLUS] = degree_sum.where(df["educ_total"] > 0) / df["educ_total"]

    non_white = (df["pop_total"] - df["pop_white"]) / df["pop_total"]
    df[Col.PCT_NON_WHITE] = non_white.where(df["pop_total"] > 0).clip(lower=0.0, upper=1.0)

    df[Col.MEDIAN_INCOME] = df["median_income"].where(df["median_income"].notna())
    df[Col.MEDIAN_AGE] = df["median_age"].where(df["median_age"].notna())

    result = df[
        [
            Col.STATE,
            Col.DISTRICT,
            Col.PCT_BACHELOR_PLUS,
            Col.PCT_NON_WHITE,
            Col.MEDIAN_INCOME,
            Col.MEDIAN_AGE,
        ]
    ].copy()
    result = result[result[Col.STATE].notna()].reset_index(drop=True)
    return result


def load_census(cache_path: Path) -> pd.DataFrame | None:
    """Load census features from disk, validating with pandera schema."""
    if not cache_path.exists():
        logger.info("Census cache not found: %s — skipping census features", cache_path)
        return None
    df = pd.read_csv(cache_path)
    from src.data.validators import validate_census

    validate_census(df)
    return df


def main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    api_key = os.getenv("CENSUS_API_KEY")
    from src.data.loaders import load_fips_states

    raw = fetch_census_acs(api_key=api_key, cache_path=DataFile.CENSUS_RAW)
    df = parse_census_response(raw=raw, fips_map=load_fips_states())
    features = build_census_features(df)
    features.to_csv(ensure(DataFile.CENSUS), index=False)
    logger.info("Census features saved → %s (%d districts)", DataFile.CENSUS, len(features))


if __name__ == "__main__":
    main()
