import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Col:
    # raw source columns
    YEAR = "year"
    STATE = "state_po"
    DISTRICT = "district"
    PARTY = "party"
    CANDIDATE = "candidate"
    CANDIDATE_VOTES = "candidatevotes"
    TOTAL_VOTES = "totalvotes"
    STAGE = "stage"
    SPECIAL = "special"
    RUNOFF = "runoff"
    WRITE_IN = "writein"
    UNOFFICIAL = "unofficial"
    FUSION_TICKET = "fusion_ticket"
    STATE_FIPS = "state_fips"
    STATE_CEN = "state_cen"
    STATE_IC = "state_ic"
    OFFICE = "office"
    MODE = "mode"
    VERSION = "version"
    # approval source columns
    START = "start"
    # engineered columns
    DEM_VOTES = "dem_votes"
    REP_VOTES = "rep_votes"
    DEM_VOTE_SHARE = "dem_vote_share"
    DEM_LEAN = "dem_lean"
    DEM_LEAN_TREND = "dem_lean_trend"
    VOTE_SHARE = "vote_share"
    PRES_APPROVAL = "pres_approval"
    PRES_DISAPPROVAL = "pres_disapproval"
    PRES_PARTY = "pres_party"
    GENERIC_BALLOT = "generic_ballot"
    PREV_WINNER = "prev_winner"
    CANDIDATES = "candidates"
    IS_INCUMBENT = "is_incumbent_on_ballot"
    REDISTRICTING = "redistricting"
    NATION_INCUMBENT = "nation_incumbent_party"
    DISTRICT_INCUMBENT = "district_incumbent_party"
    DEM_WON = "dem_won"


class Val:
    GEN = "GEN"
    DEMOCRAT = "DEMOCRAT"
    REPUBLICAN = "REPUBLICAN"
    INDEPENDENT = "INDEPENDENT"
    DEM_FARMER_LABOR = "DEMOCRATIC-FARMER-LABOR"
    FOGLIETTA_DEM = "FOGLIETTA (DEMOCRAT)"
    INDEPENDENT_REP = "INDEPENDENT-REPUBLICAN"


PRES_PARTY = {
    1976: "REPUBLICAN",  # Ford
    1978: "DEMOCRAT",  # Carter
    1980: "DEMOCRAT",
    1982: "REPUBLICAN",  # Reagan
    1984: "REPUBLICAN",
    1986: "REPUBLICAN",
    1988: "REPUBLICAN",
    1990: "REPUBLICAN",  # Bush Sr
    1992: "REPUBLICAN",
    1994: "DEMOCRAT",  # Clinton
    1996: "DEMOCRAT",
    1998: "DEMOCRAT",
    2000: "DEMOCRAT",
    2002: "REPUBLICAN",  # Bush Jr
    2004: "REPUBLICAN",
    2006: "REPUBLICAN",
    2008: "REPUBLICAN",
    2010: "DEMOCRAT",  # Obama
    2012: "DEMOCRAT",
    2014: "DEMOCRAT",
    2016: "DEMOCRAT",
    2018: "REPUBLICAN",  # Trump
    2020: "REPUBLICAN",
    2022: "DEMOCRAT",  # Biden
    2024: "DEMOCRAT",
    2026: "REPUBLICAN",  # Trump
}

# Net Dem advantage in generic congressional ballot (Dem% - Rep%), final pre-election polling.
# Positive = Dem-favorable environment, negative = Rep-favorable.
# Sources: Gallup final likely-voter poll 1976-2000 (news.gallup.com/poll/21094),
#          RealClearPolitics final avg 2002-2016, FiveThirtyEight final avg 2018-2024.
# 1988: no Gallup final ballot on record; value estimated from adjacent context.
# 1980: Gallup RV showed D+15 (structurally biased); LV environment was Rep-favorable.
GENERIC_BALLOT = {
    1976: 14,   # Gallup LV
    1978: 9,    # Gallup
    1980: -5,   # Gallup LV (RV D+15 misleadingly large; actual House D+2.6)
    1982: 12,   # Gallup RV
    1984: 9,    # Gallup RV
    1986: 10,   # Gallup
    1988: 4,    # estimated — no Gallup final ballot conducted
    1990: 8,    # Gallup
    1992: 12,   # Gallup
    1994: -7,   # Gallup LV (RV showed D+4 — misleading; actual R+6.5)
    1996: 0,    # Gallup RV: tied
    1998: 1,    # Gallup LV ~0; RV showed D+7 (biased); actual D+1.8
    2000: 1,    # Gallup LV (RV showed D+8)
    2002: -2,   # RCP final avg R+1.7
    2004: 1,    # Gallup LV
    2006: 7,    # Gallup LV (RCP showed D+11; LV more accurate; actual D+7.9)
    2008: 12,   # Gallup LV
    2010: -9,   # RCP final avg R+9
    2012: 1,    # RCP avg
    2014: -2,   # RCP R+2.4
    2016: 5,    # RCP D+5
    2018: 7,    # RCP D+7.3
    2020: 7,    # RCP D+6.8
    2022: -1,   # FiveThirtyEight final avg R+0.9
    2024: 0,    # RCP R+0.3
    2026: 7,    # RCP avg Oct 2026
}

REDISTRICTING_YEARS = {1982, 1992, 2002, 2012, 2022}

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


def load_clean_house_data(path: str) -> pd.DataFrame:
    logger.info("Loading house data from %s", path)
    df = pd.read_csv(path, low_memory=False)

    bool_cols = [
        Col.RUNOFF,
        Col.SPECIAL,
        Col.WRITE_IN,
        Col.UNOFFICIAL,
        Col.FUSION_TICKET,
    ]
    exclude_if_true = [Col.SPECIAL, Col.RUNOFF, Col.UNOFFICIAL, Col.WRITE_IN]

    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().str.upper() == "TRUE"

    df = df[df[Col.STAGE] == Val.GEN]
    for col in exclude_if_true:
        df = df[df[col] == False]

    drop = [
        "state",
        Col.STATE_FIPS,
        Col.STATE_CEN,
        Col.STATE_IC,
        Col.FUSION_TICKET,
        Col.OFFICE,
        Col.MODE,
        Col.VERSION,
        Col.STAGE,
        Col.SPECIAL,
        Col.RUNOFF,
        Col.UNOFFICIAL,
        Col.WRITE_IN,
    ]
    df = df.drop(columns=drop)
    df = df[df[Col.PARTY].notna()]
    df = df[(df[Col.CANDIDATE_VOTES] >= 0) & (df[Col.TOTAL_VOTES] >= 0)]

    party_map = {
        Val.DEM_FARMER_LABOR: Val.DEMOCRAT,
        Val.FOGLIETTA_DEM: Val.DEMOCRAT,
        Val.INDEPENDENT_REP: Val.REPUBLICAN,
    }
    df[Col.PARTY] = df[Col.PARTY].replace(party_map)
    df = df.reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))
    return df


def get_top_candidate_per_party(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[
            df.groupby([Col.YEAR, Col.STATE, Col.DISTRICT, Col.PARTY])[
                Col.CANDIDATE_VOTES
            ].idxmax()
        ]
        .drop(columns=[Col.CANDIDATE])
        .reset_index(drop=True)
    )


def build_dems_reps(
    top_candidates: pd.DataFrame, dem_lean_window: int = 4
) -> pd.DataFrame:
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
                np.polyfit(range(dem_lean_window), v, 1)[0]
                if not np.any(np.isnan(v))
                else np.nan
            ),
            raw=True,
        )
    )
    return dems_reps.reset_index(drop=True)


def load_approval(path: str) -> pd.DataFrame:
    logger.info("Loading approval ratings from %s", path)
    approval = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[Col.START, "end", Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL, "unsure"],
    )
    approval[Col.YEAR] = pd.to_datetime(approval[Col.START]).dt.year
    approval = approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]
    approval[Col.PRES_PARTY] = approval[Col.YEAR].map(PRES_PARTY)

    # flip sign so higher always means "better for Democrats":
    # dem pres: high approval = good for Dems → keep positive
    # rep pres: high approval = bad for Dems → flip negative
    sign = np.where(approval[Col.PRES_PARTY] == Val.DEMOCRAT, 1, -1)
    approval[Col.PRES_APPROVAL] = approval[Col.PRES_APPROVAL] * sign
    approval[Col.PRES_DISAPPROVAL] = approval[Col.PRES_DISAPPROVAL] * (-sign)
    return approval[[Col.YEAR, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL]]


def build_winners(
    top_candidates: pd.DataFrame,
    dems_reps: pd.DataFrame,
    approval: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Building winners table")
    winners = top_candidates.loc[
        top_candidates.groupby([Col.YEAR, Col.STATE, Col.DISTRICT])[
            Col.CANDIDATE_VOTES
        ].idxmax()
    ].reset_index(drop=True)
    winners[Col.VOTE_SHARE] = winners[Col.CANDIDATE_VOTES] / winners[Col.TOTAL_VOTES]
    winners = winners.merge(
        dems_reps[
            [Col.YEAR, Col.STATE, Col.DISTRICT, Col.DEM_LEAN, Col.DEM_LEAN_TREND]
        ],
        on=[Col.YEAR, Col.STATE, Col.DISTRICT],
        how="left",
    )
    winners = winners.merge(approval, on=Col.YEAR, how="left")
    winners[Col.GENERIC_BALLOT] = winners[Col.YEAR].map(GENERIC_BALLOT)
    return winners


def add_incumbency_features(winners: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding incumbency features")
    winner_names = (
        df.loc[
            df.groupby([Col.YEAR, Col.STATE, Col.DISTRICT])[
                Col.CANDIDATE_VOTES
            ].idxmax()
        ][[Col.YEAR, Col.STATE, Col.DISTRICT, Col.CANDIDATE]]
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
    winner_names = winner_names.merge(
        all_cands, on=[Col.YEAR, Col.STATE, Col.DISTRICT], how="left"
    )
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


def build_feature_matrix(winners: pd.DataFrame) -> tuple:
    """Returns (X, y, years) — years is a Series aligned with X for walk-forward splitting."""
    logger.info("Building feature matrix")
    seats_per_year = (
        winners.groupby([Col.YEAR, Col.PARTY])[Col.DISTRICT]
        .count()
        .unstack(fill_value=0)
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
    w = pd.get_dummies(
        w,
        columns=[Col.NATION_INCUMBENT, Col.DISTRICT_INCUMBENT],
        drop_first=True,
    )

    X = w.drop(columns=_DROP_COLS).dropna()
    y = w[Col.DEM_WON][X.index]
    years = w[Col.YEAR][X.index]
    logger.info("Feature matrix: %d rows, %d features", len(X), len(X.columns))
    return X, y, years


def build_pred_2026(
    winners: pd.DataFrame,
    dems_reps: pd.DataFrame,
    X_columns,
    pres_approval: int = -32,
    pres_disapproval: int = 65,
) -> tuple:
    """Returns (pred_features, district_ids) where district_ids labels each row.

    pres_approval / pres_disapproval follow the sign convention from load_approval:
    raw rating multiplied by -1 for a Republican president, so a higher value means
    more Dem-favorable (e.g. raw 42% approval → pres_approval = -42).
    Defaults reflect Oct 2026 Trump polling (538 avg).
    generic_ballot is taken directly from GENERIC_BALLOT[2026].
    """
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
    pred[Col.IS_INCUMBENT] = 1
    pred[Col.REDISTRICTING] = 0

    pred = pd.get_dummies(
        pred,
        columns=[Col.NATION_INCUMBENT, Col.DISTRICT_INCUMBENT],
        drop_first=True,
    )
    pred = pred.reindex(columns=X_columns, fill_value=0)
    pred[Col.DEM_LEAN] = pred[Col.DEM_LEAN].fillna(pred[Col.DEM_LEAN].median())
    pred[Col.DEM_LEAN_TREND] = pred[Col.DEM_LEAN_TREND].fillna(0)
    logger.info("  → %d districts", len(pred))
    return pred, district_ids
