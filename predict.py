from pathlib import Path

from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    build_party_votes,
    build_pred_2026,
    build_winners,
    load_approval,
    load_house_data,
)
from src.model import cross_validate, feature_weights, train

DATA_DIR = Path("data")


def main() -> None:
    df = load_house_data(DATA_DIR / "1976-2024-house.tab")
    party_votes = build_party_votes(df)
    dems_reps = build_dems_reps(party_votes)
    approval = load_approval(DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls")

    winners = build_winners(party_votes, dems_reps, approval)
    winners = add_incumbency_features(winners, df)
    X, y = build_feature_matrix(winners)

    print(f"CV accuracy: {cross_validate(X, y):.4f}")

    pipeline = train(X, y)
    pred_2026 = build_pred_2026(winners, dems_reps, X.columns)
    predictions = pipeline.predict(pred_2026)

    print(f"Democrat seats:   {predictions.sum()}")
    print(f"Republican seats: {len(predictions) - predictions.sum()}")
    print("\nFeature weights:")
    for feat, w in feature_weights(pipeline, X.columns).items():
        print(f"  {feat}: {w:.1f}%")


if __name__ == "__main__":
    main()
