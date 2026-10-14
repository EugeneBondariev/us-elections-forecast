from __future__ import annotations

import logging
import os
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.constants import (
    ECON_GROWTH,
    GENERIC_BALLOT,
    OPEN_SEATS_2026,
    PRES_PARTY,
    PRES_YEAR_FOR_HOUSE,
    REDISTRICTING_YEARS,
    UNEMPLOYMENT,
    Col,
    Val,
)
from src.data.fundraising import load_fundraising
from src.data.loaders import load_approval, load_clean_house_data, load_pres_lean
from src.data.validators import validate_approval_coverage, validate_feature_matrix
from src.paths import DataFile

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetResult",
    "get_top_candidate_per_party",
    "build_dems_reps",
    "build_winners",
    "add_incumbency_features",
    "build_feature_matrix",
    "build_pred_2026",
    "build_dataset",
]


class DatasetResult(NamedTuple):
    X: pd.DataFrame
    y: pd.Series
    winners: pd.DataFrame
    years: pd.Series
    dems_reps: pd.DataFrame
    pres_lean: pd.DataFrame | None
    fundraising: pd.DataFrame | None

_DROP_COLS = [
    Col.YEAR,
    Col.STATE,
    Col.DISTRICT,
    Col.PARTY,
    Col.CANDIDATE_VOTES,
    Col.TOTAL_VOTES,
    Col.VOTE_SHARE,
    Col.DEM_WON,
]


def get_top_candidate_per_party(df: pd.DataFrame) -> pd.DataFrame:
    """Return the highest-vote candidate per party-district-year."""
    return (
        df.loc[
            df.groupby([Col.YEAR, Col.STATE, Col.DISTRICT, Col.PARTY])[Col.CANDIDATE_VOTES].idxmax()
        ]
        .drop(columns=[Col.CANDIDATE])
        .reset_index(drop=True)
    )


def build_dems_reps(top_candidates: pd.DataFrame, dem_lean_window: int = 4) -> pd.DataFrame:
    """Merge Dem/Rep top candidates and compute rolling lean and trend features."""
    logger.info("Computing dem lean features (window=%d)", dem_lean_window)
    reps = top_candidates[top_candidates[Col.PARTY] == Val.REPUBLICAN][
        [Col.YEAR, Col.STATE, Col.DISTRICT, Col.CANDIDATE_VOTES]
    ].rename(columns={Col.CANDIDATE_VOTES: Col.REP_VOTES})

    dems = top_candidates[top_candidates[Col.PARTY] == Val.DEMOCRAT][
        [Col.YEAR, Col.STATE, Col.DISTRICT, Col.CANDIDATE_VOTES]
    ].rename(columns={Col.CANDIDATE_VOTES: Col.DEM_VOTES})

    dems_reps = dems.merge(reps, on=[Col.YEAR, Col.STATE, Col.DISTRICT], how="left")
    dems_reps[Col.DEM_VOTE_SHARE] = dems_reps[Col.DEM_VOTES] / (
        dems_reps[Col.DEM_VOTES] + dems_reps[Col.REP_VOTES]
    )
    dems_reps = dems_reps.sort_values([Col.STATE, Col.DISTRICT, Col.YEAR])

    dems_reps[Col.DEM_LEAN] = dems_reps.groupby([Col.STATE, Col.DISTRICT])[
        Col.DEM_VOTE_SHARE
    ].transform(lambda x: x.shift(1).rolling(dem_lean_window).mean())

    dems_reps[Col.DEM_LEAN_TREND] = dems_reps.groupby([Col.STATE, Col.DISTRICT])[
        Col.DEM_VOTE_SHARE
    ].transform(
        lambda x: x.shift(1)
        .rolling(dem_lean_window)
        .apply(
            lambda v: (
                np.polyfit(range(dem_lean_window), v, 1)[0] if not np.any(np.isnan(v)) else np.nan
            ),
            raw=True,
        )
    )
    dems_reps[Col.PREV_DEM_SHARE] = dems_reps.groupby([Col.STATE, Col.DISTRICT])[
        Col.DEM_VOTE_SHARE
    ].transform(lambda x: x.shift(1))
    return dems_reps.reset_index(drop=True)


def build_winners(
    top_candidates: pd.DataFrame,
    dems_reps: pd.DataFrame,
    approval: pd.DataFrame,
    pres_lean: pd.DataFrame | None = None,
    fundraising: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build winners table by merging top candidates with structural and economic features."""
    logger.info("Building winners table")
    winners = top_candidates.loc[
        top_candidates.groupby([Col.YEAR, Col.STATE, Col.DISTRICT])[Col.CANDIDATE_VOTES].idxmax()
    ].reset_index(drop=True)
    winners[Col.VOTE_SHARE] = winners[Col.CANDIDATE_VOTES] / winners[Col.TOTAL_VOTES]
    winners = winners.merge(
        dems_reps[
            [
                Col.YEAR,
                Col.STATE,
                Col.DISTRICT,
                Col.DEM_LEAN,
                Col.DEM_LEAN_TREND,
                Col.PREV_DEM_SHARE,
            ]
        ],
        on=[Col.YEAR, Col.STATE, Col.DISTRICT],
        how="left",
    )
    winners = winners.merge(approval, on=Col.YEAR, how="left")
    winners[Col.GENERIC_BALLOT] = winners[Col.YEAR].map(GENERIC_BALLOT)
    # Sign convention matches pres_approval: positive = Dem-favorable
    pres_is_rep = winners[Col.YEAR].map(PRES_PARTY) == "REPUBLICAN"
    sign = pres_is_rep.map({True: -1, False: 1})
    winners[Col.ECON_GROWTH] = winners[Col.YEAR].map(ECON_GROWTH) * sign
    # High unemployment under Rep president = good for Dems → flip sign the other way
    winners[Col.UNEMPLOYMENT] = winners[Col.YEAR].map(UNEMPLOYMENT) * (-sign)
    if pres_lean is not None:
        winners["_pres_year"] = winners[Col.YEAR].map(PRES_YEAR_FOR_HOUSE)
        lean_lookup = pres_lean.rename(columns={"pres_year": "_pres_year"})
        winners = winners.merge(
            lean_lookup[[Col.STATE, Col.DISTRICT, "_pres_year", Col.PRES_LEAN]],
            on=[Col.STATE, Col.DISTRICT, "_pres_year"],
            how="left",
        ).drop(columns=["_pres_year"])
        winners[Col.PRES_LEAN] = winners[Col.PRES_LEAN].fillna(0.5)
    if fundraising is not None:
        _fund_cols = [
            Col.FUNDRAISING_DEM_SHARE, Col.LARGE_DONOR_DEM_SHARE,
            Col.PAC_DEM_SHARE, Col.SPENDING_DEM_SHARE, Col.SURPLUS_DEM_SHARE,
        ]
        available = [c for c in _fund_cols if c in fundraising.columns]
        winners = winners.merge(
            fundraising[[Col.YEAR, Col.STATE, Col.DISTRICT] + available],
            on=[Col.YEAR, Col.STATE, Col.DISTRICT],
            how="left",
        )
        for c in available:
            winners[c] = winners[c].fillna(0.5)
    return winners


def add_incumbency_features(winners: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add is_incumbent, prev_winner, and redistricting flag to the winners table."""
    logger.info("Adding incumbency features")
    winner_names = (
        df.loc[df.groupby([Col.YEAR, Col.STATE, Col.DISTRICT])[Col.CANDIDATE_VOTES].idxmax()][
            [Col.YEAR, Col.STATE, Col.DISTRICT, Col.CANDIDATE]
        ]
        .sort_values([Col.STATE, Col.DISTRICT, Col.YEAR])
        .reset_index(drop=True)
    )
    winner_names[Col.PREV_WINNER] = winner_names.groupby([Col.STATE, Col.DISTRICT])[
        Col.CANDIDATE
    ].shift(1)
    all_cands = (
        df.groupby([Col.YEAR, Col.STATE, Col.DISTRICT])[Col.CANDIDATE]
        .apply(set)
        .reset_index()
        .rename(columns={Col.CANDIDATE: Col.CANDIDATES})
    )
    winner_names = winner_names.merge(all_cands, on=[Col.YEAR, Col.STATE, Col.DISTRICT], how="left")
    winner_names[Col.IS_INCUMBENT] = winner_names.apply(
        lambda r: (
            float(r[Col.PREV_WINNER] in r[Col.CANDIDATES])
            if pd.notna(r[Col.PREV_WINNER])
            else np.nan
        ),
        axis=1,
    )
    winners = winners.merge(
        winner_names[[Col.YEAR, Col.STATE, Col.DISTRICT, Col.IS_INCUMBENT]],
        on=[Col.YEAR, Col.STATE, Col.DISTRICT],
        how="left",
    )
    winners[Col.REDISTRICTING] = winners[Col.YEAR].isin(REDISTRICTING_YEARS).astype(int)
    return winners


def build_feature_matrix(winners: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (X, y, years) — years is a Series aligned with X for walk-forward splitting."""
    logger.info("Building feature matrix")
    seats_per_year = (
        winners.groupby([Col.YEAR, Col.PARTY])[Col.DISTRICT].count().unstack(fill_value=0)
    )
    seats_per_year = seats_per_year[[Val.DEMOCRAT, Val.REPUBLICAN]]
    seats_per_year["Majority"] = np.where(
        seats_per_year[Val.DEMOCRAT] > seats_per_year[Val.REPUBLICAN],
        Val.DEMOCRAT,
        Val.REPUBLICAN,
    )
    majority_map = seats_per_year["Majority"].shift(1).to_dict()

    w = winners.copy()
    w[Col.NATION_INCUMBENT] = w[Col.YEAR].map(majority_map)
    w = w.sort_values([Col.STATE, Col.DISTRICT, Col.YEAR])
    w[Col.DISTRICT_INCUMBENT] = w.groupby([Col.STATE, Col.DISTRICT])[Col.PARTY].shift(1)
    w[Col.DEM_WON] = (w[Col.PARTY] == Val.DEMOCRAT).astype(int)
    w = w[w[Col.DISTRICT_INCUMBENT].isin([Val.DEMOCRAT, Val.REPUBLICAN])]
    w[Col.INC_X_BALLOT] = w[Col.IS_INCUMBENT] * w[Col.GENERIC_BALLOT]
    w = pd.get_dummies(
        w,
        columns=[Col.NATION_INCUMBENT, Col.DISTRICT_INCUMBENT, Col.STATE],
        drop_first=True,
    )

    X = w.drop(columns=_DROP_COLS, errors="ignore").dropna()
    X.columns = pd.Index(
        [str(c) for c in X.columns]
    )  # StrEnum → plain str for sklearn compatibility
    y = w[Col.DEM_WON][X.index]
    years = w[Col.YEAR][X.index]
    logger.info("Feature matrix: %d rows, %d features", len(X), len(X.columns))
    validate_feature_matrix(X)
    return X, y, years


def build_dataset() -> DatasetResult:
    """Load all pipeline inputs and return a DatasetResult with every component.

    All callers can destructure only what they need:
        ds = build_dataset()
        X, y = ds.X, ds.y                          # simple analysis scripts
        X, y, years = ds.X, ds.y, ds.years         # walk-forward / CV
        # ds.dems_reps / ds.pres_lean / ds.fundraising  for build_pred_2026
    """
    df = load_clean_house_data(DataFile.HOUSE)
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(DataFile.APPROVAL)
    # PRES_PARTY has a 2026 entry too, but build_pred_2026() hardcodes that year's
    # approval directly (pres_approval=-32 etc.) rather than sourcing it from this
    # file, and no house-election-year row ever equals 2026 (the data only goes
    # through 2024) - requiring it here would demand a row that's never actually used.
    validate_approval_coverage(approval, required_years=set(top_candidates[Col.YEAR]))
    pres_lean = load_pres_lean(DataFile.PRES_08_20, DataFile.PRES_20_24)
    fundraising = load_fundraising(
        cache_path=DataFile.FUNDRAISING,
        api_key=os.getenv("FEC_API_KEY"),
    )
    winners = build_winners(top_candidates, dems_reps, approval, pres_lean, fundraising)
    winners = add_incumbency_features(winners, df)
    X, y, years = build_feature_matrix(winners)
    return DatasetResult(X=X, y=y, winners=winners, years=years,
                         dems_reps=dems_reps, pres_lean=pres_lean, fundraising=fundraising)


def build_pred_2026(
    winners: pd.DataFrame,
    dems_reps: pd.DataFrame,
    feature_columns: pd.Index,
    pres_approval: int = -32,
    pres_disapproval: int = 65,
    open_seats: set = OPEN_SEATS_2026,
    pres_lean: pd.DataFrame | None = None,
    fundraising: pd.DataFrame | None = None,
    econ_growth_2026: float = -ECON_GROWTH[2026],  # negative: Rep president, weak economy
    unemployment_2026: float = UNEMPLOYMENT[
        2026
    ],  # positive: Rep president, high unemp = Dem-favorable
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pred_features, district_ids) where district_ids labels each row."""
    logger.info("Building 2026 prediction features")
    incumbents = (
        winners[winners[Col.YEAR] == 2024][[Col.STATE, Col.DISTRICT, Col.PARTY]]
        .rename(columns={Col.PARTY: Col.DISTRICT_INCUMBENT})
        .reset_index(drop=True)
    )

    district_ids = incumbents[[Col.STATE, Col.DISTRICT]].copy()

    dem_lean_2026 = (
        dems_reps[dems_reps[Col.YEAR].isin([2018, 2020, 2022, 2024])]
        .groupby([Col.STATE, Col.DISTRICT])[Col.DEM_VOTE_SHARE]
        .mean()
        .reset_index()
        .rename(columns={Col.DEM_VOTE_SHARE: Col.DEM_LEAN})
    )
    dem_trend_2026 = (
        dems_reps[dems_reps[Col.YEAR].isin([2018, 2020, 2022, 2024])]
        .sort_values([Col.STATE, Col.DISTRICT, Col.YEAR])
        .groupby([Col.STATE, Col.DISTRICT])[Col.DEM_VOTE_SHARE]
        .apply(
            lambda v: (
                np.polyfit(range(len(v)), v.values, 1)[0]
                if len(v) == 4 and not np.any(np.isnan(v.values))
                else np.nan
            )
        )
        .reset_index()
        .rename(columns={Col.DEM_VOTE_SHARE: Col.DEM_LEAN_TREND})
    )

    pred = incumbents.merge(dem_lean_2026, on=[Col.STATE, Col.DISTRICT], how="left")
    pred = pred.merge(dem_trend_2026, on=[Col.STATE, Col.DISTRICT], how="left")
    pred[Col.NATION_INCUMBENT] = Val.REPUBLICAN
    pred[Col.PRES_APPROVAL] = pres_approval
    pred[Col.PRES_DISAPPROVAL] = pres_disapproval
    pred[Col.GENERIC_BALLOT] = GENERIC_BALLOT[2026]
    pred[Col.ECON_GROWTH] = econ_growth_2026
    pred[Col.UNEMPLOYMENT] = unemployment_2026
    pred[Col.IS_INCUMBENT] = pred.apply(
        lambda r: 0 if (r[Col.STATE], r[Col.DISTRICT]) in open_seats else 1, axis=1
    )
    pred[Col.REDISTRICTING] = 0
    prev_share_2026 = dems_reps[dems_reps[Col.YEAR] == 2024][
        [Col.STATE, Col.DISTRICT, Col.DEM_VOTE_SHARE]
    ].rename(columns={Col.DEM_VOTE_SHARE: Col.PREV_DEM_SHARE})
    pred = pred.merge(prev_share_2026, on=[Col.STATE, Col.DISTRICT], how="left")
    pred[Col.PREV_DEM_SHARE] = pred[Col.PREV_DEM_SHARE].fillna(pred[Col.PREV_DEM_SHARE].median())
    pred[Col.INC_X_BALLOT] = pred[Col.IS_INCUMBENT] * pred[Col.GENERIC_BALLOT]
    if pres_lean is not None and Col.PRES_LEAN in list(feature_columns):
        lean_2026 = (
            pres_lean[pres_lean["pres_year"].isin([2020, 2024])]
            .groupby([Col.STATE, Col.DISTRICT])[Col.PRES_LEAN]
            .mean()
            .reset_index()
        )
        pred = pred.merge(lean_2026, on=[Col.STATE, Col.DISTRICT], how="left")
        pred[Col.PRES_LEAN] = pred[Col.PRES_LEAN].fillna(0.5)
    if fundraising is not None:
        _fund_cols = [
            Col.FUNDRAISING_DEM_SHARE, Col.LARGE_DONOR_DEM_SHARE,
            Col.PAC_DEM_SHARE, Col.SPENDING_DEM_SHARE, Col.SURPLUS_DEM_SHARE,
        ]
        available = [c for c in _fund_cols if c in fundraising.columns and c in list(feature_columns)]
        if available:
            fund_2026 = fundraising[fundraising[Col.YEAR] == 2026][
                [Col.STATE, Col.DISTRICT] + available
            ]
            pred = pred.merge(fund_2026, on=[Col.STATE, Col.DISTRICT], how="left")
            for c in available:
                pred[c] = pred[c].fillna(0.5)

    pred = pd.get_dummies(
        pred,
        columns=[Col.NATION_INCUMBENT, Col.DISTRICT_INCUMBENT, Col.STATE],
        drop_first=True,
    )
    pred = pred.reindex(columns=feature_columns, fill_value=0)
    pred[Col.DEM_LEAN] = pred[Col.DEM_LEAN].fillna(pred[Col.DEM_LEAN].median())
    pred[Col.DEM_LEAN_TREND] = pred[Col.DEM_LEAN_TREND].fillna(0)
    logger.info("  → %d districts", len(pred))
    return pred, district_ids
