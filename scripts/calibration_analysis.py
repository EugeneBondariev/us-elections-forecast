"""Reliability diagram: compare raw LR vs isotonic-calibrated LR probabilities."""

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

from src.features import build_dataset
from src.model import build_calibrated_pipeline, build_pipeline
from src.paths import OUTPUTS_DIR


def main() -> None:
    X, y, _ = build_dataset()

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
    fig.tight_layout()
    out = OUTPUTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
