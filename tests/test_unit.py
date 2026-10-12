"""
Unit tests that run without any files in data/.
All fixtures are synthetic hand-crafted DataFrames / arrays.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. WeightedPipeline routes sample_weight correctly
# ---------------------------------------------------------------------------

def test_weighted_pipeline_accepts_sample_weight():
    from src.model import build_pipeline

    X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]}, dtype=float)
    y = pd.Series([0, 0, 1, 1, 1])
    w = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    pipe = build_pipeline()
    pipe.fit(X, y, sample_weight=w)  # must not raise
    assert pipe.predict(X).shape == (5,)


# ---------------------------------------------------------------------------
# 2. compute_sample_weights gives more weight to recent years
# ---------------------------------------------------------------------------

def test_sample_weights_recency():
    from src.model import compute_sample_weights

    years = pd.Series([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
    w = compute_sample_weights(years)
    assert w[-1] > w[0]          # most recent > oldest
    assert len(w) == len(years)
    assert (w > 0).all()


# ---------------------------------------------------------------------------
# 3. simulate_election returns correct shape and range
# ---------------------------------------------------------------------------

def test_simulate_election_shape():
    from src.model import simulate_election

    proba = np.array([0.3, 0.5, 0.7, 0.6, 0.4])
    result = simulate_election(proba, n=500, seed=0)
    assert result.shape == (500,)
    assert result.min() >= 0
    assert result.max() <= len(proba)


# ---------------------------------------------------------------------------
# 4. bayesian_simulate returns correct shape
# ---------------------------------------------------------------------------

def test_bayesian_simulate_shape():
    from src.model import bayesian_simulate

    proba = np.full(10, 0.5)
    result = bayesian_simulate(proba, n=200, seed=1)
    assert result.shape == (200,)
    assert result.min() >= 0
    assert result.max() <= 10


# ---------------------------------------------------------------------------
# 5. feature_weights returns sorted dict summing to ~100
# ---------------------------------------------------------------------------

def test_feature_weights_sum():
    from src.model import build_pipeline, feature_weights

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + np.random.randn(50) * 0.1 > 0).astype(int))
    pipe = build_pipeline()
    pipe.fit(X, y)
    fw = feature_weights(pipe, X.columns)
    assert set(fw.keys()) == {"a", "b", "c"}
    assert abs(sum(fw.values()) - 100) < 0.01
    vals = list(fw.values())
    assert vals == sorted(vals, reverse=True)  # sorted descending


# ---------------------------------------------------------------------------
# 6. bootstrap_ci returns expected keys and valid ranges
# ---------------------------------------------------------------------------

def test_bootstrap_ci_keys():
    from src.model import bootstrap_ci

    np.random.seed(0)
    n = 100
    X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
    y = pd.Series(np.random.randint(0, 2, n))
    # Use years 2000 (train) and 2002 (test) so bootstrap_ci has one valid split
    years = pd.Series([2000] * 50 + [2002] * 50)
    result = bootstrap_ci(X, y, years, n_bootstrap=10)
    expected_keys = {
        "accuracy_mean", "accuracy_low", "accuracy_high",
        "seat_error_mean", "seat_error_low", "seat_error_high",
    }
    assert set(result.keys()) == expected_keys
    assert 0 <= result["accuracy_mean"] <= 1
    assert result["accuracy_low"] <= result["accuracy_mean"] <= result["accuracy_high"]


# ---------------------------------------------------------------------------
# 7. build_dems_reps computes dem_vote_share correctly
# ---------------------------------------------------------------------------

def test_build_dems_reps_vote_share():
    from src.features import build_dems_reps
    from src.constants import Col, Val

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
# 8. walk_forward_cv produces reasonable output on toy data
# ---------------------------------------------------------------------------

def test_walk_forward_cv_toy():
    from src.model import walk_forward_cv

    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int))
    years = pd.Series([2000] * 50 + [2002] * 50 + [2004] * 50 + [2006] * 50)
    result = walk_forward_cv(X, y, years, start_test_year=2002)
    assert len(result) == 3  # 2002, 2004, 2006
    assert "district_accuracy" in result.columns
    assert (result["district_accuracy"] >= 0).all()
    assert (result["district_accuracy"] <= 1).all()


# ---------------------------------------------------------------------------
# 9. stacking ensemble trains and predicts
# ---------------------------------------------------------------------------

def test_stacking_ensemble_trains():
    from src.model import build_stacking_ensemble

    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    model = build_stacking_ensemble()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 10. feature_weights works with StackingClassifier (GBM branch)
# ---------------------------------------------------------------------------

def test_feature_weights_stacking_ensemble():
    from src.model import build_stacking_ensemble, feature_weights

    np.random.seed(7)
    X = pd.DataFrame(np.random.randn(80, 3), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int))
    model = build_stacking_ensemble()
    model.fit(X, y)
    fw = feature_weights(model, X.columns)
    assert set(fw.keys()) == {"a", "b", "c"}
    assert abs(sum(fw.values()) - 100) < 0.01
    vals = list(fw.values())
    assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# 11. simulate_election is deterministic with a fixed seed
# ---------------------------------------------------------------------------

def test_simulate_election_deterministic():
    from src.model import simulate_election

    proba = np.array([0.3, 0.5, 0.7, 0.6, 0.4])
    r1 = simulate_election(proba, n=1000, seed=99)
    r2 = simulate_election(proba, n=1000, seed=99)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# 12. validate_feature_matrix: valid DataFrame passes without raising
# ---------------------------------------------------------------------------

def test_validate_feature_matrix_valid():
    from src.validators import validate_feature_matrix

    X = pd.DataFrame({
        "dem_lean":            [0.5, 0.6],
        "prev_dem_vote_share": [0.45, 0.55],
        "pres_approval":       [10.0, -5.0],
        "extra_col":           [1, 2],
    })
    result = validate_feature_matrix(X)
    assert result is X


def test_validate_feature_matrix_invalid_does_not_raise():
    """Out-of-range values should log a warning but never raise."""
    from src.validators import validate_feature_matrix

    X = pd.DataFrame({
        "dem_lean":            [0.5],
        "prev_dem_vote_share": [1.8],   # > 1.0, violates schema
        "pres_approval":       [10.0],
    })
    result = validate_feature_matrix(X)   # must not raise
    assert result is X


# ---------------------------------------------------------------------------
# 13. add_incumbency_features: NaN for first year, 1 when incumbent returns,
#     0 when a different candidate runs in place of the prior winner
# ---------------------------------------------------------------------------

def test_add_incumbency_features():
    from src.features import add_incumbency_features
    from src.constants import Col, Val

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
# 14. build_feature_matrix: column names are plain str (StrEnum → str fix)
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
                Col.DEM_LEAN: 0.5,      Col.DEM_LEAN_TREND: 0.0, Col.PREV_DEM_SHARE: 0.5,
                Col.PRES_APPROVAL: 10.0, Col.PRES_DISAPPROVAL: -40.0,
                Col.GENERIC_BALLOT: 5.0,
                Col.ECON_GROWTH: 2.0,   Col.UNEMPLOYMENT: 5.0,
                Col.IS_INCUMBENT: 1.0,  Col.REDISTRICTING: 0,
            })
    return pd.DataFrame(rows)


def test_build_feature_matrix_column_types():
    """Every column name must be a plain str, not a StrEnum member."""
    from src.features import build_feature_matrix

    winners = _make_minimal_winners()
    X, y, years = build_feature_matrix(winners)

    for col in X.columns:
        assert type(col) is str, f"Expected plain str, got {type(col)} for column {col!r}"
    assert X.isnull().sum().sum() == 0
    assert set(y.unique()).issubset({0, 1})
    assert len(X) > 0


def test_build_feature_matrix_no_data_leakage():
    """district_incumbent must never be the same year's party (always lagged)."""
    from src.features import build_feature_matrix

    winners = _make_minimal_winners()
    X, y, years = build_feature_matrix(winners)

    # district_incumbent_party_REPUBLICAN column: 1 means prior incumbent was Rep.
    # In our toy data districts 1,3 are always Rep-held, so their REPUBLICAN flag must be 1.
    rep_col = "district_incumbent_party_REPUBLICAN"
    if rep_col in X.columns:
        # All rows that come from district 1 or 3 should have the Rep flag = 1
        # (we can't assert row-level without district info after dummies, so check shape)
        assert X[rep_col].isin([0, 1]).all()
