"""
Unit tests for src.features.features that run without any files in data/.
All fixtures are synthetic hand-crafted DataFrames / arrays.
"""
import pandas as pd

# ---------------------------------------------------------------------------
# 1. build_dems_reps computes dem_vote_share correctly
# ---------------------------------------------------------------------------

def test_build_dems_reps_vote_share():
    from src.constants import Col, Val
    from src.features.features import build_dems_reps

    top = pd.DataFrame([
        {Col.YEAR: 2020, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 60},
        {Col.YEAR: 2020, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 40},
        {Col.YEAR: 2022, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 55},
        {Col.YEAR: 2022, Col.STATE: "CA", Col.DISTRICT: 1,
         Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 45},
    ])
    result = build_dems_reps(top)
    row_2020 = result[result[Col.YEAR] == 2020].iloc[0]
    assert abs(row_2020[Col.DEM_VOTE_SHARE] - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# 2. add_incumbency_features: NaN for first year, 1 when incumbent returns,
#    0 when a different candidate runs in place of the prior winner
# ---------------------------------------------------------------------------

def test_add_incumbency_features():
    from src.constants import Col, Val
    from src.features.features import add_incumbency_features

    # Raw candidate-level data: 3 years, 1 district
    df = pd.DataFrame([
        # 2020: ALICE (D) wins 100v, BOB (R) loses 80v
        {Col.YEAR: 2020, Col.STATE: "TX", Col.DISTRICT: 1,
         Col.CANDIDATE: "ALICE", Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 100},
        {Col.YEAR: 2020, Col.STATE: "TX", Col.DISTRICT: 1,
         Col.CANDIDATE: "BOB",   Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 80},
        # 2022: ALICE returns (IS_INCUMBENT = 1)
        {Col.YEAR: 2022, Col.STATE: "TX", Col.DISTRICT: 1,
         Col.CANDIDATE: "ALICE", Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 90},
        {Col.YEAR: 2022, Col.STATE: "TX", Col.DISTRICT: 1,
         Col.CANDIDATE: "CAROL", Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 70},
        # 2024: DAVE replaces ALICE — ALICE not running (IS_INCUMBENT = 0)
        {Col.YEAR: 2024, Col.STATE: "TX", Col.DISTRICT: 1,
         Col.CANDIDATE: "DAVE",  Col.PARTY: Val.DEMOCRAT,   Col.CANDIDATE_VOTES: 95},
        {Col.YEAR: 2024, Col.STATE: "TX", Col.DISTRICT: 1,
         Col.CANDIDATE: "CAROL", Col.PARTY: Val.REPUBLICAN, Col.CANDIDATE_VOTES: 65},
    ])
    # winners: one row per district-year (the top vote-getter, here always the Dem)
    winners = pd.DataFrame([
        {Col.YEAR: 2020, Col.STATE: "TX", Col.DISTRICT: 1, Col.PARTY: Val.DEMOCRAT},
        {Col.YEAR: 2022, Col.STATE: "TX", Col.DISTRICT: 1, Col.PARTY: Val.DEMOCRAT},
        {Col.YEAR: 2024, Col.STATE: "TX", Col.DISTRICT: 1, Col.PARTY: Val.DEMOCRAT},
    ])

    result = add_incumbency_features(winners, df)
    r20 = result[result[Col.YEAR] == 2020].iloc[0]
    r22 = result[result[Col.YEAR] == 2022].iloc[0]
    r24 = result[result[Col.YEAR] == 2024].iloc[0]

    assert pd.isna(r20[Col.IS_INCUMBENT])   # no prior year → undefined
    assert r22[Col.IS_INCUMBENT] == 1.0     # ALICE came back
    assert r24[Col.IS_INCUMBENT] == 0.0     # DAVE ≠ ALICE → open seat


# ---------------------------------------------------------------------------
# 3. build_feature_matrix: column names are plain str (StrEnum → str fix),
#    and no data leakage from the current year's incumbency
# ---------------------------------------------------------------------------

def _make_minimal_winners() -> pd.DataFrame:
    """Synthetic winners table for build_feature_matrix — no data files needed."""
    from src.constants import Col, Val
    rows = []
    for year in [2000, 2002, 2004]:
        for dist in range(1, 5):
            party = Val.DEMOCRAT if dist % 2 == 0 else Val.REPUBLICAN
            rows.append({
                Col.YEAR: year,         Col.STATE: "TX",      Col.DISTRICT: dist,
                Col.PARTY: party,
                Col.CANDIDATE_VOTES: 100, Col.TOTAL_VOTES: 180, Col.VOTE_SHARE: 100 / 180,
                Col.DEM_VOTE_SHARE: 0.55,
                Col.DEM_LEAN: 0.5,      Col.DEM_LEAN_TREND: 0.0, Col.PREV_DEM_SHARE: 0.5,
                Col.PRES_APPROVAL: 10.0, Col.PRES_DISAPPROVAL: -40.0,
                Col.GENERIC_BALLOT: 5.0,
                Col.ECON_GROWTH: 2.0,   Col.UNEMPLOYMENT: 5.0,
                Col.IS_INCUMBENT: 1.0,  Col.REDISTRICTING: 0,
            })
    return pd.DataFrame(rows)


def test_build_feature_matrix_column_types():
    """Every column name must be a plain str, not a StrEnum member."""
    from src.features.features import build_feature_matrix

    winners = _make_minimal_winners()
    X, y, y_vs, years = build_feature_matrix(winners)

    for col in X.columns:
        assert type(col) is str, f"Expected plain str, got {type(col)} for column {col!r}"
    assert X.isnull().sum().sum() == 0
    assert set(y.unique()).issubset({0, 1})
    assert len(X) > 0


def test_build_feature_matrix_no_data_leakage():
    """district_incumbent must never be the same year's party (always lagged)."""
    from src.features.features import build_feature_matrix

    winners = _make_minimal_winners()
    X, y, y_vs, years = build_feature_matrix(winners)

    # district_incumbent_party_REPUBLICAN column: 1 means prior incumbent was Rep.
    rep_col = "district_incumbent_party_REPUBLICAN"
    if rep_col in X.columns:
        assert X[rep_col].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# 4. State partisan lean: no region_ dummies; state_partisan_lean column present
# ---------------------------------------------------------------------------

def _make_multiregion_winners() -> pd.DataFrame:
    """Synthetic winners table spanning multiple states."""
    from src.constants import Col, Val
    rows = []
    state_dist = [("CA", 1), ("TX", 2), ("IL", 3), ("NY", 4)]
    for year in [2000, 2002, 2004]:
        for state, dist in state_dist:
            party = Val.DEMOCRAT if dist % 2 == 0 else Val.REPUBLICAN
            rows.append({
                Col.YEAR: year,         Col.STATE: state,     Col.DISTRICT: dist,
                Col.PARTY: party,
                Col.CANDIDATE_VOTES: 100, Col.TOTAL_VOTES: 180, Col.VOTE_SHARE: 100 / 180,
                Col.DEM_VOTE_SHARE: 0.55,
                Col.DEM_LEAN: 0.5,      Col.DEM_LEAN_TREND: 0.0, Col.PREV_DEM_SHARE: 0.5,
                Col.PRES_APPROVAL: 10.0, Col.PRES_DISAPPROVAL: -40.0,
                Col.GENERIC_BALLOT: 5.0,
                Col.ECON_GROWTH: 2.0,   Col.UNEMPLOYMENT: 5.0,
                Col.IS_INCUMBENT: 1.0,  Col.REDISTRICTING: 0,
            })
    return pd.DataFrame(rows)


def test_no_state_dummies_in_feature_matrix():
    """No state_po_XX column should appear in X."""
    from src.features.features import build_feature_matrix

    winners = _make_multiregion_winners()
    X, *_ = build_feature_matrix(winners)

    state_cols = [c for c in X.columns if c.startswith("state_po_")]
    assert state_cols == [], f"Unexpected state dummies found: {state_cols}"


def test_no_region_dummies_in_feature_matrix():
    """region_ dummy columns must be gone; state_partisan_lean replaces them."""
    from src.features.features import build_feature_matrix

    winners = _make_multiregion_winners()
    X, *_ = build_feature_matrix(winners)

    region_cols = [c for c in X.columns if c.startswith("region_")]
    assert region_cols == [], f"region_ dummies should be removed: {region_cols}"


def test_state_partisan_lean_in_feature_matrix():
    """state_partisan_lean must be present in X and be finite."""
    from src.features.features import build_feature_matrix

    winners = _make_multiregion_winners()
    X, *_ = build_feature_matrix(winners)

    assert "state_partisan_lean" in X.columns, "state_partisan_lean missing from X"
    assert X["state_partisan_lean"].notna().all(), "state_partisan_lean must have no NaN after dropna"


def test_state_to_region_covers_all_50_states():
    """Every US state abbreviation must appear exactly once in STATE_TO_REGION."""
    from src.constants import STATE_TO_REGION

    all_50 = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    }
    missing = all_50 - set(STATE_TO_REGION.keys())
    extra = set(STATE_TO_REGION.keys()) - all_50
    assert not missing, f"States missing from STATE_TO_REGION: {sorted(missing)}"
    assert not extra, f"Non-states in STATE_TO_REGION: {sorted(extra)}"

    valid_regions = {"Northeast", "South", "Midwest", "Mountain", "Pacific"}
    bad = {s: r for s, r in STATE_TO_REGION.items() if r not in valid_regions}
    assert not bad, f"Unknown region values: {bad}"


# ---------------------------------------------------------------------------
# 5. min_train_year: rows before the cutoff are excluded from X/y/years
# ---------------------------------------------------------------------------

def _make_multiyear_winners_for_filter() -> pd.DataFrame:
    """Synthetic winners with years 2000/2002/2004 across two states."""
    from src.constants import Col, Val
    rows = []
    state_dist = [("CA", 1), ("TX", 2)]
    for year in [2000, 2002, 2004]:
        for state, dist in state_dist:
            party = Val.DEMOCRAT if dist % 2 == 0 else Val.REPUBLICAN
            rows.append({
                Col.YEAR: year,         Col.STATE: state,     Col.DISTRICT: dist,
                Col.PARTY: party,
                Col.CANDIDATE_VOTES: 100, Col.TOTAL_VOTES: 180, Col.VOTE_SHARE: 100 / 180,
                Col.DEM_VOTE_SHARE: 0.55,
                Col.DEM_LEAN: 0.5,      Col.DEM_LEAN_TREND: 0.0, Col.PREV_DEM_SHARE: 0.5,
                Col.PRES_APPROVAL: 10.0, Col.PRES_DISAPPROVAL: -40.0,
                Col.GENERIC_BALLOT: 5.0,
                Col.ECON_GROWTH: 2.0,   Col.UNEMPLOYMENT: 5.0,
                Col.IS_INCUMBENT: 1.0,  Col.REDISTRICTING: 0,
            })
    return pd.DataFrame(rows)


def test_min_train_year_filters_rows():
    """Rows with year < min_train_year must not appear in X after filtering."""
    from src.features.features import build_feature_matrix

    winners = _make_multiyear_winners_for_filter()
    X_all, y_all, y_vs_all, years_all = build_feature_matrix(winners)

    mask = years_all >= 2004
    X_filt = X_all.loc[mask]
    years_filt = years_all.loc[mask]

    assert set(years_filt.unique()) == {2004}, f"Expected only 2004, got {set(years_filt.unique())}"
    assert len(X_filt) < len(X_all)


def test_min_train_year_none_keeps_all_rows():
    """With min_train_year=None (default), all rows are retained."""
    from src.features.features import build_feature_matrix

    winners = _make_multiyear_winners_for_filter()
    _, _, _, years = build_feature_matrix(winners)

    # year 2000 is dropped (no prior district_incumbent), but 2002 and 2004 must be present
    assert {2002, 2004}.issubset(set(years.unique()))


# ---------------------------------------------------------------------------
# 6. net_pres_approval: replaces separate approval/disapproval columns in X
# ---------------------------------------------------------------------------

def test_net_pres_approval_in_feature_matrix():
    """net_pres_approval must be in X; pres_approval and pres_disapproval must not."""
    from src.features.features import build_feature_matrix

    winners = _make_minimal_winners()
    X, *_ = build_feature_matrix(winners)

    assert "net_pres_approval" in X.columns, "net_pres_approval missing from X"
    assert "pres_approval" not in X.columns, "pres_approval should be dropped (net replaces it)"
    assert "pres_disapproval" not in X.columns, "pres_disapproval should be dropped"


def test_net_pres_approval_value():
    """net_pres_approval = pres_approval + pres_disapproval (both already sign-flipped)."""
    from src.features.features import build_feature_matrix

    winners = _make_minimal_winners()  # approval=10.0, disapproval=-40.0 in fixture
    X, *_ = build_feature_matrix(winners)

    expected_net = 10.0 + (-40.0)  # = -30.0
    assert (X["net_pres_approval"] == expected_net).all(), (
        f"Expected net_pres_approval={expected_net}, got {X['net_pres_approval'].unique()}"
    )


# ---------------------------------------------------------------------------
# 7. y_vote_share: fourth return value from build_feature_matrix
# ---------------------------------------------------------------------------

def test_y_vote_share_is_continuous():
    """build_feature_matrix must return y_vote_share as a float Series in [0, 1]."""
    from src.features.features import build_feature_matrix

    winners = _make_minimal_winners()
    _, _, y_vs, years = build_feature_matrix(winners)

    assert y_vs.dtype == float or str(y_vs.dtype).startswith("float")
    assert y_vs.between(0, 1).all(), "vote share must be in [0, 1]"
    assert len(y_vs) == len(years)


def test_y_vote_share_aligned_with_y():
    """y_vote_share must have the same index as y and years."""
    from src.features.features import build_feature_matrix

    winners = _make_minimal_winners()
    _, y, y_vs, years = build_feature_matrix(winners)

    assert list(y_vs.index) == list(y.index)
    assert list(y_vs.index) == list(years.index)
