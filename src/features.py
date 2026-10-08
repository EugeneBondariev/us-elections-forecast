import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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

# Net Dem advantage in generic congressional ballot (Dem% - Rep%), pre-election polling.
# Positive = Dem-favorable environment, negative = Rep-favorable.
# TODO: replace with the real values, tackle silent quitters
GENERIC_BALLOT = {
    1976: 12,
    1978: 8,
    1980: -5,
    1982: 9,
    1984: 2,
    1986: 7,
    1988: 4,
    1990: 7,
    1992: 10,
    1994: -6,
    1996: 4,
    1998: 3,
    2000: 1,
    2002: -5,
    2004: -1,
    2006: 11,
    2008: 12,
    2010: -9,
    2012: 1,
    2014: -6,
    2016: 1,
    2018: 8,
    2020: 6,
    2022: -3,
    2024: -2,
    2026: 7,
}

REDISTRICTING_YEARS = {1982, 1992, 2002, 2012, 2022}

_DROP_COLS = [
    "year",
    "state_po",
    "district",
    "party",
    "candidatevotes",
    "totalvotes",
    "vote_share",
    "dem_won",
]


def load_house_data(path: str) -> pd.DataFrame:
    logger.info("Loading house data from %s", path)
    df = pd.read_csv(path, low_memory=False)

    bool_cols = ["runoff", "special", "writein", "unofficial", "fusion_ticket"]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.strip().str.upper() == "TRUE"

    df = df[df["stage"] == "GEN"]
    for col in ["special", "runoff", "unofficial", "writein"]:
        df = df[df[col] == False]

    drop = [
        "state_fips",
        "state_cen",
        "state_ic",
        "fusion_ticket",
        "office",
        "mode",
        "version",
        "stage",
        "special",
        "runoff",
        "unofficial",
        "writein",
    ]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df[df["party"].notna()]
    df = df[(df["candidatevotes"] >= 0) & (df["totalvotes"] >= 0)]

    party_map = {
        "DEMOCRATIC-FARMER-LABOR": "DEMOCRAT",
        "FOGLIETTA (DEMOCRAT)": "DEMOCRAT",
        "INDEPENDENT-REPUBLICAN": "REPUBLICAN",
    }
    df["party"] = df["party"].replace(party_map)
    df = df.reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))
    return df


def build_party_votes(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        df.groupby(["year", "state_po", "district", "party"])["candidatevotes"].idxmax()
    ][
        ["year", "state_po", "district", "party", "candidatevotes", "totalvotes"]
    ].reset_index(
        drop=True
    )


def build_dems_reps(
    party_votes: pd.DataFrame, dem_lean_window: int = 4
) -> pd.DataFrame:
    logger.info("Computing dem lean features (window=%d)", dem_lean_window)
    reps = party_votes[party_votes["party"] == "REPUBLICAN"][
        ["year", "state_po", "district", "candidatevotes"]
    ].rename(columns={"candidatevotes": "rep_votes"})

    dems = (
        party_votes[party_votes["party"] == "DEMOCRAT"]
        .rename(columns={"candidatevotes": "dem_votes"})
        .copy()
    )

    dems_reps = dems.merge(reps, on=["year", "state_po", "district"], how="left")
    dems_reps["dem_vote_share"] = dems_reps["dem_votes"] / (
        dems_reps["dem_votes"] + dems_reps["rep_votes"]
    )
    dems_reps = dems_reps.sort_values(["state_po", "district", "year"])

    dems_reps["dem_lean"] = dems_reps.groupby(["state_po", "district"])[
        "dem_vote_share"
    ].transform(lambda x: x.shift(1).rolling(dem_lean_window).mean())

    dems_reps["dem_lean_trend"] = dems_reps.groupby(["state_po", "district"])[
        "dem_vote_share"
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
    return dems_reps


def load_approval(path: str) -> pd.DataFrame:
    logger.info("Loading approval ratings from %s", path)
    approval = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["start", "end", "approve", "disapprove", "unsure"],
    )
    approval["year"] = pd.to_datetime(approval["start"]).dt.year
    approval = approval[["year", "approve", "disapprove"]].rename(
        columns={"approve": "pres_approval", "disapprove": "pres_disapproval"}
    )
    approval["pres_party"] = approval["year"].map(PRES_PARTY)

    # flip sign so higher always means "better for Democrats":
    # dem pres: high approval = good for Dems → keep positive
    # rep pres: high approval = bad for Dems → flip negative
    sign = np.where(approval["pres_party"] == "DEMOCRAT", 1, -1)
    approval["pres_approval"] = approval["pres_approval"] * sign
    approval["pres_disapproval"] = approval["pres_disapproval"] * (-sign)
    return approval[["year", "pres_approval", "pres_disapproval"]]


def build_winners(
    party_votes: pd.DataFrame,
    dems_reps: pd.DataFrame,
    approval: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Building winners table")
    winners = party_votes.loc[
        party_votes.groupby(["year", "state_po", "district"])["candidatevotes"].idxmax()
    ].reset_index(drop=True)
    winners["vote_share"] = winners["candidatevotes"] / winners["totalvotes"]
    winners = winners.merge(
        dems_reps[["year", "state_po", "district", "dem_lean", "dem_lean_trend"]],
        on=["year", "state_po", "district"],
        how="left",
    )
    winners = winners.merge(approval, on="year", how="left")
    winners["generic_ballot"] = winners["year"].map(GENERIC_BALLOT)
    return winners


def add_incumbency_features(winners: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding incumbency features")
    winner_names = (
        df.loc[df.groupby(["year", "state_po", "district"])["candidatevotes"].idxmax()][
            ["year", "state_po", "district", "candidate"]
        ]
        .sort_values(["state_po", "district", "year"])
        .reset_index(drop=True)
    )
    winner_names["prev_winner"] = winner_names.groupby(["state_po", "district"])[
        "candidate"
    ].shift(1)
    all_cands = (
        df.groupby(["year", "state_po", "district"])["candidate"]
        .apply(set)
        .reset_index()
        .rename(columns={"candidate": "candidates"})
    )
    winner_names = winner_names.merge(
        all_cands, on=["year", "state_po", "district"], how="left"
    )
    winner_names["is_incumbent_on_ballot"] = winner_names.apply(
        lambda r: (
            float(r["prev_winner"] in r["candidates"])
            if pd.notna(r["prev_winner"])
            else np.nan
        ),
        axis=1,
    )
    winners = winners.merge(
        winner_names[["year", "state_po", "district", "is_incumbent_on_ballot"]],
        on=["year", "state_po", "district"],
        how="left",
    )
    winners["redistricting"] = winners["year"].isin(REDISTRICTING_YEARS).astype(int)
    return winners


def build_feature_matrix(winners: pd.DataFrame) -> tuple:
    """Returns (X, y, years) — years is a Series aligned with X for walk-forward splitting."""
    logger.info("Building feature matrix")
    seats_per_year = (
        winners.groupby(["year", "party"])["district"].count().unstack(fill_value=0)
    )
    seats_per_year = seats_per_year[["DEMOCRAT", "REPUBLICAN"]]
    seats_per_year["Majority"] = np.where(
        seats_per_year["DEMOCRAT"] > seats_per_year["REPUBLICAN"],
        "DEMOCRAT",
        "REPUBLICAN",
    )
    majority_map = seats_per_year["Majority"].shift(1).to_dict()

    w = winners.copy()
    w["nation_incumbent_party"] = w["year"].map(majority_map)
    w = w.sort_values(["state_po", "district", "year"])
    w["district_incumbent_party"] = w.groupby(["state_po", "district"])["party"].shift(
        1
    )
    w["dem_won"] = (w["party"] == "DEMOCRAT").astype(int)
    w = w[w["district_incumbent_party"] != "INDEPENDENT"]
    w = pd.get_dummies(
        w,
        columns=["nation_incumbent_party", "district_incumbent_party"],
        drop_first=True,
    )

    X = w.drop(columns=_DROP_COLS).dropna()
    y = w["dem_won"][X.index]
    years = w["year"][X.index]
    logger.info("Feature matrix: %d rows, %d features", len(X), len(X.columns))
    return X, y, years


def build_pred_2026(
    winners: pd.DataFrame,
    dems_reps: pd.DataFrame,
    X_columns,
) -> tuple:
    """Returns (pred_features, district_ids) where district_ids labels each row."""
    logger.info("Building 2026 prediction features")
    incumbents = (
        winners[winners["year"] == 2024][["state_po", "district", "party"]]
        .rename(columns={"party": "district_incumbent_party"})
        .reset_index(drop=True)
    )

    district_ids = incumbents[["state_po", "district"]].copy()

    dem_lean_2026 = (
        dems_reps[dems_reps["year"].isin([2018, 2020, 2022, 2024])]
        .groupby(["state_po", "district"])["dem_vote_share"]
        .mean()
        .reset_index()
        .rename(columns={"dem_vote_share": "dem_lean"})
    )
    dem_trend_2026 = (
        dems_reps[dems_reps["year"].isin([2018, 2020, 2022, 2024])]
        .sort_values(["state_po", "district", "year"])
        .groupby(["state_po", "district"])["dem_vote_share"]
        .apply(
            lambda v: (
                np.polyfit(range(len(v)), v.values, 1)[0]
                if len(v) == 4 and not np.any(np.isnan(v.values))
                else np.nan
            )
        )
        .reset_index()
        .rename(columns={"dem_vote_share": "dem_lean_trend"})
    )

    pred = incumbents.merge(dem_lean_2026, on=["state_po", "district"], how="left")
    pred = pred.merge(dem_trend_2026, on=["state_po", "district"], how="left")
    pred["nation_incumbent_party"] = "REPUBLICAN"
    pred["pres_approval"] = -32
    pred["pres_disapproval"] = 65
    pred["generic_ballot"] = 7
    pred["is_incumbent_on_ballot"] = 1
    pred["redistricting"] = 0

    pred = pd.get_dummies(
        pred,
        columns=["nation_incumbent_party", "district_incumbent_party"],
        drop_first=True,
    )
    pred = pred.reindex(columns=X_columns, fill_value=0)
    pred["dem_lean"] = pred["dem_lean"].fillna(pred["dem_lean"].median())
    pred["dem_lean_trend"] = pred["dem_lean_trend"].fillna(0)
    logger.info("  → %d districts", len(pred))
    return pred, district_ids
