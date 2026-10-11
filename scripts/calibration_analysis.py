"""Reliability diagram: compare raw LR vs isotonic-calibrated LR probabilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    build_winners,
    get_top_candidate_per_party,
)
from src.loaders import load_approval, load_clean_house_data
from src.model import build_calibrated_pipeline, build_pipeline

DATA_DIR = Path("data")


def main():
    df = load_clean_house_data(DATA_DIR / "1976-2024-house.tab")
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls")
    winners = build_winners(top_candidates, dems_reps, approval)
    winners = add_incumbency_features(winners, df)
    X, y, _ = build_feature_matrix(winners)

    print("Computing cross-validated probabilities (cv=5)...")
    proba_raw = cross_val_predict(build_pipeline(), X, y, cv=5, method="predict_proba")[:, 1]
    proba_cal = cross_val_predict(build_calibrated_pipeline(), X, y, cv=5, method="predict_proba")[:, 1]

    brier_raw = brier_score_loss(y, proba_raw)
    brier_cal = brier_score_loss(y, proba_cal)
    print(f"Brier score — raw LR:        {brier_raw:.4f}")
    print(f"Brier score — calibrated LR: {brier_cal:.4f}")

    frac_raw, mean_raw = calibration_curve(y, proba_raw, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y, proba_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    ax.plot(mean_raw, frac_raw, "o-", color="#e67e22", label=f"Raw LR (Brier={brier_raw:.4f})")
    ax.plot(mean_cal, frac_cal, "s-", color="#2980b9", label=f"Calibrated LR (Brier={brier_cal:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual Dem wins)")
    ax.set_title("Reliability diagram — LR calibration")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = DATA_DIR / "calibration_curve.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
