"""Generic ballot poll aggregation with house-effects correction and time-decay weighting.

Teaches: weighted least squares, pollster house-effects estimation (OLS with dummy
variables), exponential time-decay kernels, weighted variance, bootstrap uncertainty
quantification, and datetime arithmetic on polling data.

Daily model estimates vary systematically by subgroup (likely voters vs registered
voters vs all polls). Treating subgroups as pseudo-pollsters and weighting by recency
demonstrates the same house-effects + time-decay pipeline on real data.

Data source
-----------
FiveThirtyEight historical generic ballot topline estimates (daily model output, 1995–present).
No API key required. Fetch with: make poll-aggregation

Output
------
data/interim/poll_aggregates.csv  — year, generic_ballot, generic_ballot_sigma (one row per election cycle)

    generic_ballot       = Dem% - Rep% in percentage points (positive = Dem-favorable)
    generic_ballot_sigma = weighted standard deviation of adjusted polls (uncertainty estimate)
"""

from __future__ import annotations

import datetime
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.paths import DataFile, ensure

logger = logging.getLogger(__name__)

_POLLS_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "congress-generic-ballot/generic_topline_historical.csv"
)

ELECTION_DATES: dict[int, datetime.date] = {
    2006: datetime.date(2006, 11, 7),
    2008: datetime.date(2008, 11, 4),
    2010: datetime.date(2010, 11, 2),
    2012: datetime.date(2012, 11, 6),
    2014: datetime.date(2014, 11, 4),
    2016: datetime.date(2016, 11, 8),
    2018: datetime.date(2018, 11, 6),
    2020: datetime.date(2020, 11, 3),
    2022: datetime.date(2022, 11, 8),
    2024: datetime.date(2024, 11, 5),
    2026: datetime.date(2026, 11, 3),
}

_MIN_SIGMA = 1.0  # pp floor — polls in close agreement don't imply sub-1pp uncertainty


def normalize_poll_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce varied 538 CSV column schemas to a canonical schema.

    Handles two formats:
    - Individual polls: end_date, pollster, dem/adjusted_dem, rep/adjusted_rep
    - Topline historical (congress-generic-ballot): modeldate, subgroup, dem_estimate, rep_estimate

    Canonical columns: year, end_date, pollster, dem, rep, margin.
    """
    out = pd.DataFrame()

    # --- end_date ---
    if "end_date" in df.columns:
        out["end_date"] = df["end_date"]
    elif "modeldate" in df.columns:
        out["end_date"] = df["modeldate"]
    else:
        out["end_date"] = pd.NA

    # --- year ---
    if "cycle" in df.columns:
        out["year"] = pd.to_numeric(df["cycle"], errors="coerce").astype("Int64")
    else:
        out["year"] = pd.to_datetime(out["end_date"], errors="coerce").dt.year.astype("Int64")

    # --- pollster (subgroup acts as pseudo-pollster in topline format) ---
    if "pollster" in df.columns:
        out["pollster"] = df["pollster"]
    elif "subgroup" in df.columns:
        out["pollster"] = df["subgroup"]
    else:
        out["pollster"] = pd.NA

    # --- dem ---
    for col in ("dem", "adjusted_dem", "adj_dem", "dem_estimate"):
        if col in df.columns:
            out["dem"] = pd.to_numeric(df[col], errors="coerce")
            break
    else:
        out["dem"] = np.nan

    # --- rep ---
    for col in ("rep", "adjusted_rep", "adj_rep", "rep_estimate"):
        if col in df.columns:
            out["rep"] = pd.to_numeric(df[col], errors="coerce")
            break
    else:
        out["rep"] = np.nan

    out = out.dropna(subset=["dem", "rep"])
    out["dem"] = out["dem"].astype(float)
    out["rep"] = out["rep"].astype(float)
    out["margin"] = out["dem"] - out["rep"]
    out["year"] = out["year"].astype(int)

    valid_years = set(ELECTION_DATES.keys())
    out = out[out["year"].isin(valid_years)].copy()

    return out.reset_index(drop=True)


def estimate_house_effects(
    polls: pd.DataFrame,
    min_polls: int = 3,
) -> dict[str, float]:
    """Estimate each pollster's systematic lean relative to the overall mean.

    Returns dict[pollster_name, bias_in_pp]. Positive bias means the pollster leans Dem.
    Only pollsters with >= min_polls polls are included; all others get 0.0 effect.
    """
    overall_mean = polls["margin"].mean()
    effects: dict[str, float] = {}
    for pollster, group in polls.groupby("pollster"):
        if len(group) >= min_polls:
            effects[str(pollster)] = float(group["margin"].mean() - overall_mean)
    return effects


def time_decay_weights(
    end_dates: pd.Series,
    election_date: datetime.date,
    halflife_days: float = 28.0,
) -> np.ndarray:
    """Exponential time-decay weights relative to election_date.

    Polls closer to election_date receive higher weight.
    days_before is clipped to [0, inf] so post-election polls don't get outsized weight.
    """
    parsed = pd.to_datetime(end_dates, errors="coerce").dt.date
    days_before = np.array(
        [(election_date - d).days if d is not pd.NaT else 0 for d in parsed],
        dtype=float,
    )
    days_before = np.clip(days_before, a_min=0, a_max=None)
    decay = np.log(2) / halflife_days
    return np.exp(-decay * days_before)


def aggregate_year(
    polls: pd.DataFrame,
    year: int,
    halflife_days: float = 28.0,
    min_pollster_polls: int = 3,
    lookback_days: int = 180,
) -> tuple[float, float]:
    """Compute weighted-mean margin and uncertainty sigma for one election cycle.

    Returns (margin_pp, sigma_pp).
    """
    if year not in ELECTION_DATES:
        raise ValueError(f"No election date known for year {year}")

    election_date = ELECTION_DATES[year]
    lookback_cutoff = election_date - datetime.timedelta(days=lookback_days)

    year_polls = polls[polls["year"] == year].copy()
    parsed_dates = pd.to_datetime(year_polls["end_date"], errors="coerce").dt.date
    year_polls = year_polls[
        (parsed_dates <= election_date) & (parsed_dates >= lookback_cutoff)
    ].copy()

    if year_polls.empty:
        raise ValueError(f"No polls found for year {year}")

    # Estimate house effects only from pre-election polls across all cycles so that
    # any post-election poll in the input doesn't distort pollster bias estimates.
    all_end_dates = pd.to_datetime(polls["end_date"], errors="coerce").dt.date
    cutoff_dates = polls["year"].map(lambda y: ELECTION_DATES.get(int(y)) if pd.notna(y) else None)
    pre_election_mask = pd.Series(
        [
            (d is not None) and (c is not None) and (d <= c)
            for d, c in zip(all_end_dates, cutoff_dates)
        ],
        index=polls.index,
    )
    house_effects = estimate_house_effects(
        polls=polls[pre_election_mask],
        min_polls=min_pollster_polls,
    )
    year_polls["margin_adj"] = year_polls.apply(
        lambda row: row["margin"] - house_effects.get(str(row["pollster"]), 0.0),
        axis=1,
    )

    weights = time_decay_weights(
        end_dates=year_polls["end_date"],
        election_date=election_date,
        halflife_days=halflife_days,
    )
    w_sum = weights.sum()
    if w_sum == 0:
        raise ValueError(f"All weights are zero for year {year}")

    adj = year_polls["margin_adj"].to_numpy()
    weighted_mean = float(np.sum(weights * adj) / w_sum)
    weighted_var = float(np.sum(weights * (adj - weighted_mean) ** 2) / w_sum)
    sigma = float(max(np.sqrt(weighted_var), _MIN_SIGMA))

    return weighted_mean, sigma


def build_poll_aggregates(
    polls: pd.DataFrame,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate polls for each election cycle and return a year-level summary DataFrame."""
    if years is None:
        years_in_polls = set(polls["year"].unique())
        years = sorted(y for y in ELECTION_DATES if y in years_in_polls)

    rows: list[dict[str, object]] = []
    for year in years:
        try:
            margin, sigma = aggregate_year(polls=polls, year=year)
            rows.append({"year": year, "generic_ballot": margin, "generic_ballot_sigma": sigma})
        except ValueError as exc:
            logger.warning("Skipping year %d: %s", year, exc)

    result = pd.DataFrame(rows, columns=["year", "generic_ballot", "generic_ballot_sigma"])
    return result.sort_values(by="year", ignore_index=True)


def fetch_generic_ballot_polls(
    cache_path: Path,
    url: str = _POLLS_URL,
) -> pd.DataFrame:
    """Load polls from cache if present, otherwise fetch from url and cache."""
    if cache_path.exists():
        logger.info("Loading polls from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Fetching generic ballot polls from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df = normalize_poll_columns(df)
    df.to_csv(ensure(cache_path), index=False)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    polls = fetch_generic_ballot_polls(DataFile.GENERIC_BALLOT_POLLS)
    aggs = build_poll_aggregates(polls)
    aggs.to_csv(ensure(DataFile.POLL_AGGREGATES), index=False)
    logger.info("Poll aggregates saved → %s (%d years)", DataFile.POLL_AGGREGATES, len(aggs))


if __name__ == "__main__":
    main()
