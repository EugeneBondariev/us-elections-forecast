from __future__ import annotations

import logging
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from src.constants import (
    PRES_PARTY,
    PRES_YEAR_FOR_HOUSE,
    REDISTRICTING_YEARS,
    Col,
    Val,
)
from src.data.fundraising import load_fundraising
from src.data.loaders import (
    load_approval,
    load_candidate_quality,
    load_census,
    load_clean_house_data,
    load_district_analytics,
    load_generic_ballot,
    load_gerrymandering,
    load_indicators,
    load_open_seats,
    load_poll_sigma,
    load_pres_lean,
    load_speech_sentiment,
    load_speech_sentiment_hist,
)
from src.data.validators import (
    ApprovalDataSchema,
    DemsRepsSchema,
    HouseDataSchema,
    TopCandidatesSchema,
    WinnersSchema,
    validate_approval_coverage,
    validate_feature_matrix,
)
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
    y_vote_share: pd.Series
    winners: DataFrame[WinnersSchema]
    years: pd.Series
    dems_reps: DataFrame[DemsRepsSchema]
    pres_lean: pd.DataFrame | None
    fundraising: pd.DataFrame | None
    census: pd.DataFrame | None
    candidate_quality: pd.DataFrame | None
    poll_sigma: dict[int, float]
    district_analytics: pd.DataFrame | None
    gerrymandering: pd.DataFrame | None
    speech_sentiment: pd.DataFrame | None
    speech_sentiment_hist: pd.DataFrame | None


_DROP_COLS = [
    Col.YEAR,
    Col.STATE,
    Col.DISTRICT,
    Col.PARTY,
    Col.CANDIDATE_VOTES,
    Col.TOTAL_VOTES,
    Col.VOTE_SHARE,
    Col.DEM_WON,
    # Current-year vote share is the regression target, not a feature
    Col.DEM_VOTE_SHARE,
    # Replaced by net_pres_approval (approval + disapproval, both already sign-flipped)
    Col.PRES_APPROVAL,
    Col.PRES_DISAPPROVAL,
    # VIF ~182: spending is a lagging echo of fundraising; keep fundraising_dem_share.
    # Note: high VIF alone is not sufficient reason to drop a feature from a tree model —
    # VIF measures linear coefficient instability, not tree split quality. prev_dem_vote_share
    # (VIF ~26) was restored after removal caused sigma_national to jump from ~0.35 to 0.67.
    Col.SPENDING_DEM_SHARE,
]


def get_top_candidate_per_party(df: DataFrame[HouseDataSchema]) -> DataFrame[TopCandidatesSchema]:
    """Return the highest-vote candidate per party-district-year."""
    return (
        df.loc[  # type: ignore[return-value]
            df.groupby([Col.YEAR, Col.STATE, Col.DISTRICT, Col.PARTY])[Col.CANDIDATE_VOTES].idxmax()
        ]
        .drop(columns=[Col.CANDIDATE])
        .reset_index(drop=True)
    )


def build_dems_reps(
    top_candidates: DataFrame[TopCandidatesSchema], dem_lean_window: int = 4
) -> DataFrame[DemsRepsSchema]:
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
        lambda x: (
            x.shift(1)
            .rolling(dem_lean_window)
            .apply(
                lambda v: (
                    np.polyfit(range(dem_lean_window), v, 1)[0]
                    if not np.any(np.isnan(v))
                    else np.nan
                ),
                raw=True,
            )
        )
    )
    dems_reps[Col.PREV_DEM_SHARE] = dems_reps.groupby([Col.STATE, Col.DISTRICT])[
        Col.DEM_VOTE_SHARE
    ].transform(lambda x: x.shift(1))
    return dems_reps.reset_index(drop=True)  # type: ignore[return-value]


def build_winners(
    top_candidates: DataFrame[TopCandidatesSchema],
    dems_reps: DataFrame[DemsRepsSchema],
    approval: DataFrame[ApprovalDataSchema],
    pres_lean: pd.DataFrame | None = None,
    fundraising: pd.DataFrame | None = None,
    census: pd.DataFrame | None = None,
    candidate_quality: pd.DataFrame | None = None,
    district_analytics: pd.DataFrame | None = None,
    gerrymandering: pd.DataFrame | None = None,
) -> DataFrame[WinnersSchema]:
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
                Col.DEM_VOTE_SHARE,
                Col.DEM_LEAN,
                Col.DEM_LEAN_TREND,
                Col.PREV_DEM_SHARE,
            ]
        ],
        on=[Col.YEAR, Col.STATE, Col.DISTRICT],
        how="left",
    )
    winners = winners.merge(approval, on=Col.YEAR, how="left")
    _generic_ballot = load_generic_ballot()
    winners[Col.GENERIC_BALLOT] = winners[Col.YEAR].map(_generic_ballot)
    # Sign convention matches pres_approval: positive = Dem-favorable
    pres_is_rep = winners[Col.YEAR].map(PRES_PARTY) == "REPUBLICAN"
    sign = pres_is_rep.map({True: -1, False: 1})
    _econ_growth, _unemployment = load_indicators()
    winners[Col.ECON_GROWTH] = winners[Col.YEAR].map(_econ_growth) * sign
    # High unemployment under Rep president = good for Dems → flip sign the other way
    winners[Col.UNEMPLOYMENT] = winners[Col.YEAR].map(_unemployment) * (-sign)
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
            Col.FUNDRAISING_DEM_SHARE,
            Col.LARGE_DONOR_DEM_SHARE,
            Col.PAC_DEM_SHARE,
            Col.SPENDING_DEM_SHARE,
            Col.SURPLUS_DEM_SHARE,
        ]
        available = [c for c in _fund_cols if c in fundraising.columns]
        winners = winners.merge(
            fundraising[[Col.YEAR, Col.STATE, Col.DISTRICT] + available],
            on=[Col.YEAR, Col.STATE, Col.DISTRICT],
            how="left",
        )
        for c in available:
            winners[c] = winners[c].fillna(0.5)
    if census is not None:
        _census_cols = [Col.PCT_BACHELOR_PLUS, Col.PCT_NON_WHITE, Col.MEDIAN_INCOME, Col.MEDIAN_AGE]
        winners = winners.merge(
            census[[Col.STATE, Col.DISTRICT] + _census_cols],
            on=[Col.STATE, Col.DISTRICT],
            how="left",
        )
        for c in _census_cols:
            winners[c] = winners[c].fillna(winners[c].median())
    if candidate_quality is not None:
        _cq_cols = [Col.CANDIDATE_QUALITY, Col.EXPERIENCE_CYCLES_DEM, Col.EXPERIENCE_CYCLES_REP]
        available_cq = [c for c in _cq_cols if c in candidate_quality.columns]
        winners = winners.merge(
            candidate_quality[[Col.YEAR, Col.STATE, Col.DISTRICT] + available_cq],
            on=[Col.YEAR, Col.STATE, Col.DISTRICT],
            how="left",
        )
        winners[Col.CANDIDATE_QUALITY] = winners[Col.CANDIDATE_QUALITY].fillna(0.5)
        for c in [Col.EXPERIENCE_CYCLES_DEM, Col.EXPERIENCE_CYCLES_REP]:
            if c in winners.columns:
                winners[c] = winners[c].fillna(0)
    if district_analytics is not None:
        _an_cols = [Col.WIN_STREAK, Col.SWING_VOLATILITY]
        available_an = [c for c in _an_cols if c in district_analytics.columns]
        winners = winners.merge(
            district_analytics[[Col.YEAR, Col.STATE, Col.DISTRICT] + available_an],
            on=[Col.YEAR, Col.STATE, Col.DISTRICT],
            how="left",
        )
        if Col.WIN_STREAK in winners.columns:
            winners[Col.WIN_STREAK] = winners[Col.WIN_STREAK].fillna(0).astype(int)
        if Col.SWING_VOLATILITY in winners.columns:
            winners[Col.SWING_VOLATILITY] = winners[Col.SWING_VOLATILITY].fillna(
                winners[Col.SWING_VOLATILITY].median()
            )
    if gerrymandering is not None:
        _geo_cols = [Col.POLSBY_POPPER, Col.DISTRICT_AREA_KM2]
        available_geo = [c for c in _geo_cols if c in gerrymandering.columns]
        winners = winners.merge(
            gerrymandering[[Col.STATE, Col.DISTRICT] + available_geo],
            on=[Col.STATE, Col.DISTRICT],
            how="left",
        )
        for c in available_geo:
            winners[c] = winners[c].fillna(winners[c].median())
    return winners  # type: ignore[return-value]


def add_incumbency_features(
    winners: DataFrame[WinnersSchema], df: DataFrame[HouseDataSchema]
) -> DataFrame[WinnersSchema]:
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


def build_feature_matrix(
    winners: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Returns (X, y, y_vote_share, years) — all Series are aligned with X."""
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
    w[Col.NET_PRES_APPROVAL] = w[Col.PRES_APPROVAL] + w[Col.PRES_DISAPPROVAL]
    # State-level mean of the already-lagged dem_lean — captures state partisan context
    # without dummy explosion (50 sparse dummies) or information loss (5 coarse regions).
    w[Col.STATE_PARTISAN_LEAN] = w.groupby([Col.STATE, Col.YEAR])[Col.DEM_LEAN].transform("mean")
    w = pd.get_dummies(
        w,
        columns=[Col.NATION_INCUMBENT, Col.DISTRICT_INCUMBENT],
        drop_first=True,
    )

    X = w.drop(columns=_DROP_COLS, errors="ignore").dropna()
    X.columns = pd.Index(
        [str(c) for c in X.columns]
    )  # StrEnum → plain str for sklearn compatibility
    y = w[Col.DEM_WON][X.index]
    y_vote_share = w[Col.DEM_VOTE_SHARE][X.index].copy()
    # Uncontested races have no two-party vote share. Impute a plausible value so the
    # regressor sees the full range of district competitiveness. 0.85/0.15 reflects the
    # typical two-party share when a major-party candidate runs unopposed.
    uncontested_dem = y_vote_share.isna() & (y == 1)
    uncontested_rep = y_vote_share.isna() & (y == 0)
    y_vote_share.loc[uncontested_dem] = 0.85
    y_vote_share.loc[uncontested_rep] = 0.15
    years = w[Col.YEAR][X.index]
    logger.info("Feature matrix: %d rows, %d features", len(X), len(X.columns))
    validate_feature_matrix(X)
    return X, y, y_vote_share, years


def build_dataset(min_train_year: int | None = None) -> DatasetResult:
    """Load all pipeline inputs and return a DatasetResult with every component.

    All callers can destructure only what they need:
        ds = build_dataset()
        X, y = ds.X, ds.y                          # simple analysis scripts
        X, y, years = ds.X, ds.y, ds.years         # walk-forward / CV
        ds.dems_reps / ds.pres_lean / ds.fundraising  for build_pred_2026

    min_train_year: if set, drop rows from X/y/years where year < min_train_year.
        Feature engineering (rolling lean, trend) still uses the full history so
        early-cycle features remain accurate. Only the rows exposed to the model
        are trimmed. Typical value: 1994 (modern partisan alignment era).
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
    census = load_census()
    candidate_quality = load_candidate_quality()
    poll_sigma = load_poll_sigma()
    district_analytics = load_district_analytics()
    gerrymandering = load_gerrymandering()
    speech_sentiment = load_speech_sentiment()
    speech_sentiment_hist = load_speech_sentiment_hist()
    winners = build_winners(
        top_candidates=top_candidates,
        dems_reps=dems_reps,
        approval=approval,
        pres_lean=pres_lean,
        fundraising=fundraising,
        census=census,
        candidate_quality=candidate_quality,
        district_analytics=district_analytics,
        gerrymandering=gerrymandering,
    )
    winners = add_incumbency_features(winners, df)
    if speech_sentiment_hist is not None:
        winners = winners.merge(
            speech_sentiment_hist[
                [Col.YEAR, Col.STATE, Col.DISTRICT, Col.SPEECH_SENTIMENT, Col.SPEECH_COUNT]
            ],
            on=[Col.YEAR, Col.STATE, Col.DISTRICT],
            how="left",
        )
        winners[Col.SPEECH_SENTIMENT] = winners[Col.SPEECH_SENTIMENT].fillna(0.0)
        winners[Col.SPEECH_COUNT] = winners[Col.SPEECH_COUNT].fillna(0).astype(int)
    X, y, y_vote_share, years = build_feature_matrix(winners)
    if min_train_year is not None:
        mask = years >= min_train_year
        X, y, y_vote_share, years = (
            X.loc[mask],
            y.loc[mask],
            y_vote_share.loc[mask],
            years.loc[mask],
        )
        logger.info("Filtered to min_train_year=%d: %d rows remaining", min_train_year, len(X))
    return DatasetResult(
        X=X,
        y=y,
        y_vote_share=y_vote_share,
        winners=winners,
        years=years,
        dems_reps=dems_reps,
        pres_lean=pres_lean,
        fundraising=fundraising,
        census=census,
        candidate_quality=candidate_quality,
        poll_sigma=poll_sigma,
        district_analytics=district_analytics,
        gerrymandering=gerrymandering,
        speech_sentiment=speech_sentiment,
        speech_sentiment_hist=speech_sentiment_hist,
    )


def build_pred_2026(
    winners: pd.DataFrame,
    dems_reps: pd.DataFrame,
    feature_columns: pd.Index,
    pres_approval: int = -32,
    pres_disapproval: int = 65,
    open_seats: set[tuple[str, int]] | None = None,
    pres_lean: pd.DataFrame | None = None,
    fundraising: pd.DataFrame | None = None,
    econ_growth_2026: float | None = None,
    unemployment_2026: float | None = None,
    census: pd.DataFrame | None = None,
    candidate_quality: pd.DataFrame | None = None,
    district_analytics: pd.DataFrame | None = None,
    gerrymandering: pd.DataFrame | None = None,
    speech_sentiment: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pred_features, district_ids) where district_ids labels each row."""
    logger.info("Building 2026 prediction features")
    if open_seats is None:
        open_seats = load_open_seats(2026)
    if econ_growth_2026 is None or unemployment_2026 is None:
        _econ_growth, _unemployment = load_indicators()
        if econ_growth_2026 is None:
            econ_growth_2026 = -_econ_growth[2026]  # negative: Rep president
        if unemployment_2026 is None:
            unemployment_2026 = _unemployment[
                2026
            ]  # positive: Rep pres, high unemp = Dem-favorable
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
                np.polyfit(range(len(v)), v.to_numpy(), 1)[0]
                if len(v) == 4 and not np.any(np.isnan(v.to_numpy()))
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
    pred[Col.GENERIC_BALLOT] = load_generic_ballot()[2026]
    pred[Col.ECON_GROWTH] = econ_growth_2026
    pred[Col.UNEMPLOYMENT] = unemployment_2026
    _open_seats: set[tuple[str, int]] = open_seats
    pred[Col.IS_INCUMBENT] = pred.apply(
        lambda r: 0 if (r[Col.STATE], r[Col.DISTRICT]) in _open_seats else 1, axis=1
    )
    pred[Col.REDISTRICTING] = 0
    prev_share_2026 = dems_reps[dems_reps[Col.YEAR] == 2024][
        [Col.STATE, Col.DISTRICT, Col.DEM_VOTE_SHARE]
    ].rename(columns={Col.DEM_VOTE_SHARE: Col.PREV_DEM_SHARE})
    pred = pred.merge(prev_share_2026, on=[Col.STATE, Col.DISTRICT], how="left")
    pred[Col.PREV_DEM_SHARE] = pred[Col.PREV_DEM_SHARE].fillna(pred[Col.PREV_DEM_SHARE].median())
    pred[Col.INC_X_BALLOT] = pred[Col.IS_INCUMBENT] * pred[Col.GENERIC_BALLOT]
    pred[Col.NET_PRES_APPROVAL] = pred[Col.PRES_APPROVAL] + pred[Col.PRES_DISAPPROVAL]
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
            Col.FUNDRAISING_DEM_SHARE,
            Col.LARGE_DONOR_DEM_SHARE,
            Col.PAC_DEM_SHARE,
            Col.SPENDING_DEM_SHARE,
            Col.SURPLUS_DEM_SHARE,
        ]
        available = [
            c for c in _fund_cols if c in fundraising.columns and c in list(feature_columns)
        ]
        if available:
            fund_2026 = fundraising[fundraising[Col.YEAR] == 2026][
                [Col.STATE, Col.DISTRICT] + available
            ]
            pred = pred.merge(fund_2026, on=[Col.STATE, Col.DISTRICT], how="left")
            for c in available:
                pred[c] = pred[c].fillna(0.5)

    if census is not None:
        _census_cols = [Col.PCT_BACHELOR_PLUS, Col.PCT_NON_WHITE, Col.MEDIAN_INCOME, Col.MEDIAN_AGE]
        available_c = [c for c in _census_cols if c in list(feature_columns)]
        if available_c:
            pred = pred.merge(
                census[[Col.STATE, Col.DISTRICT] + available_c],
                on=[Col.STATE, Col.DISTRICT],
                how="left",
            )
            for c in available_c:
                pred[c] = pred[c].fillna(pred[c].median())
    if candidate_quality is not None:
        _cq_cols = [Col.CANDIDATE_QUALITY, Col.EXPERIENCE_CYCLES_DEM, Col.EXPERIENCE_CYCLES_REP]
        available_cq = [
            c for c in _cq_cols if c in list(feature_columns) and c in candidate_quality.columns
        ]
        if available_cq:
            cq_2026 = candidate_quality[candidate_quality[Col.YEAR] == 2026][
                [Col.STATE, Col.DISTRICT] + available_cq
            ]
            pred = pred.merge(cq_2026, on=[Col.STATE, Col.DISTRICT], how="left")
            if Col.CANDIDATE_QUALITY in pred.columns:
                pred[Col.CANDIDATE_QUALITY] = pred[Col.CANDIDATE_QUALITY].fillna(0.5)
            for c in [Col.EXPERIENCE_CYCLES_DEM, Col.EXPERIENCE_CYCLES_REP]:
                if c in pred.columns:
                    pred[c] = pred[c].fillna(0)
    # district_analytics: use 2024 values as the inertia signal going into 2026
    if district_analytics is not None:
        _an_cols = [c for c in [Col.WIN_STREAK, Col.SWING_VOLATILITY] if c in list(feature_columns)]
        if _an_cols:
            an_2026 = district_analytics[district_analytics[Col.YEAR] == 2024][
                [Col.STATE, Col.DISTRICT] + _an_cols
            ]
            pred = pred.merge(an_2026, on=[Col.STATE, Col.DISTRICT], how="left")
            if Col.WIN_STREAK in pred.columns:
                pred[Col.WIN_STREAK] = pred[Col.WIN_STREAK].fillna(0).astype(int)
            if Col.SWING_VOLATILITY in pred.columns:
                pred[Col.SWING_VOLATILITY] = pred[Col.SWING_VOLATILITY].fillna(
                    pred[Col.SWING_VOLATILITY].median()
                )
    # gerrymandering: static geometry — same values for 2026 as for training
    if gerrymandering is not None:
        _geo_cols = [
            c for c in [Col.POLSBY_POPPER, Col.DISTRICT_AREA_KM2] if c in list(feature_columns)
        ]
        if _geo_cols:
            pred = pred.merge(
                gerrymandering[[Col.STATE, Col.DISTRICT] + _geo_cols],
                on=[Col.STATE, Col.DISTRICT],
                how="left",
            )
            for c in _geo_cols:
                pred[c] = pred[c].fillna(pred[c].median())
    # speech_sentiment: 2026-cycle Congress.gov data; historical equivalent merged into winners during training
    if speech_sentiment is not None:
        _sp_cols = [
            c for c in [Col.SPEECH_SENTIMENT, Col.SPEECH_COUNT] if c in list(feature_columns)
        ]
        if _sp_cols:
            pred = pred.merge(
                speech_sentiment[[Col.STATE, Col.DISTRICT] + _sp_cols],
                on=[Col.STATE, Col.DISTRICT],
                how="left",
            )
            if Col.SPEECH_SENTIMENT in pred.columns:
                pred[Col.SPEECH_SENTIMENT] = pred[Col.SPEECH_SENTIMENT].fillna(0.0)
            if Col.SPEECH_COUNT in pred.columns:
                pred[Col.SPEECH_COUNT] = pred[Col.SPEECH_COUNT].fillna(0).astype(int)
    pred = pd.get_dummies(
        pred,
        columns=[Col.NATION_INCUMBENT, Col.DISTRICT_INCUMBENT],
        drop_first=True,
    )
    pred = pred.reindex(columns=feature_columns, fill_value=0)
    pred[Col.DEM_LEAN] = pred[Col.DEM_LEAN].fillna(pred[Col.DEM_LEAN].median())
    pred[Col.DEM_LEAN_TREND] = pred[Col.DEM_LEAN_TREND].fillna(0)
    # state_partisan_lean is in feature_columns; reindex filled it with 0 — recompute from
    # the dem_lean values we just restored (STATE column is gone after reindex, so we need
    # to rebuild it from district_ids which still has STATE).
    state_lean = (
        district_ids[[Col.STATE]]
        .assign(**{str(Col.DEM_LEAN): pred[Col.DEM_LEAN].values})
        .groupby(Col.STATE)[str(Col.DEM_LEAN)]
        .transform("mean")
    )
    pred[str(Col.STATE_PARTISAN_LEAN)] = state_lean.values
    logger.info("  → %d districts", len(pred))
    return pred, district_ids
