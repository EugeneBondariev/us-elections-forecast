"""SHAP analysis: load per-district feature explanations saved by predict.py.

Reads reports/shap_values.csv (written by `make predict`) instead of retraining
the model. Run via: python -m src.visualization.shap_analysis
"""
from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import FIGURES_DIR, DataFile, ensure

logger = logging.getLogger(__name__)


def main() -> None:
    if not DataFile.SHAP_CSV.exists():
        logger.error(
            "SHAP CSV not found at %s — run `make predict` first.", DataFile.SHAP_CSV
        )
        sys.exit(1)

    combined = pd.read_csv(DataFile.SHAP_CSV)
    if "proba_dem" not in combined.columns:
        logger.error(
            "SHAP CSV is in the old format (no proba_dem column) — re-run `make predict` to regenerate it."
        )
        sys.exit(1)
    meta_cols = ["label", "proba_dem"]
    shap_df = combined.drop(columns=meta_cols)
    districts = combined[meta_cols].copy()

    logger.info("SHAP values loaded: %d districts, %d features", len(shap_df), shap_df.shape[1])

    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    logger.info("Top 10 features by mean |SHAP|:")
    for feat, val in mean_abs.head(10).items():
        logger.info("  %-45s  %.4f", feat, val)

    # Global feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = mean_abs.head(10)
    ax.barh(top10.index[::-1], top10.to_numpy()[::-1], color=["#3366cc"] * len(top10))
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (LightGBM, 2026 prediction set)")
    fig.tight_layout()
    importance_path = ensure(FIGURES_DIR / "shap_importance.png")
    fig.savefig(importance_path, dpi=150)
    logger.info("Saved: %s", importance_path)
    plt.close(fig)

    # Most competitive district waterfall
    competitive = (
        districts
        .assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
        .sort_values("closeness")
    )
    label = competitive.iloc[0]["label"]
    row_idx = competitive.index[0]  # original integer position, preserved from CSV read
    sv = shap_df.iloc[row_idx].sort_values(key=abs, ascending=False).head(8)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_colors = ["#3366cc" if v > 0 else "#cc3333" for v in sv.values]
    ax2.barh(sv.index[::-1], sv.values[::-1], color=bar_colors[::-1])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value (positive = Dem-favoring)")
    ax2.set_title(f"SHAP waterfall — most competitive district: {label}")
    fig2.tight_layout()
    waterfall_path = ensure(FIGURES_DIR / "shap_waterfall.png")
    fig2.savefig(waterfall_path, dpi=150)
    logger.info("Saved: %s", waterfall_path)
    plt.close(fig2)


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    main()
