"""SHAP analysis: compute and save per-district feature explanations."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.forecast import run_forecast

DATA_DIR = Path("data")


def main():
    print("Running forecast pipeline...")
    result = run_forecast(DATA_DIR)
    shap_df: pd.DataFrame = result["shap_values"]
    districts: pd.DataFrame = result["districts"]

    print(f"SHAP values computed for {len(shap_df)} districts, {shap_df.shape[1]} features")

    # Mean absolute SHAP per feature
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP|:")
    for feat, val in mean_abs.head(10).items():
        print(f"  {feat:<45} {val:.4f}")

    # Bar plot: global feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = mean_abs.head(10)
    colors = ["#3366cc"] * len(top10)
    ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (LightGBM, 2026 prediction set)")
    plt.tight_layout()
    out_path = DATA_DIR / "shap_importance.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close()

    # Most competitive district waterfall
    if "label" in districts.columns:
        competitive = (
            districts
            .assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
            .sort_values("closeness")
            .reset_index(drop=True)
        )
        label = competitive.iloc[0]["label"]
        row_idx = competitive.index[0]
        if row_idx < len(shap_df):
            sv = shap_df.iloc[row_idx].sort_values(key=abs, ascending=False).head(8)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            bar_colors = ["#3366cc" if v > 0 else "#cc3333" for v in sv.values]
            ax2.barh(sv.index[::-1], sv.values[::-1], color=bar_colors[::-1])
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("SHAP value (positive = Dem-favoring)")
            ax2.set_title(f"SHAP waterfall — most competitive district: {label}")
            plt.tight_layout()
            waterfall_path = DATA_DIR / "shap_waterfall.png"
            plt.savefig(waterfall_path, dpi=150)
            print(f"Saved: {waterfall_path}")
            plt.close()

    # Save full SHAP table
    if "label" in districts.columns:
        shap_out = shap_df.copy()
        shap_out.insert(0, "label", districts["label"].values)
        shap_out.to_csv(DATA_DIR / "shap_values.csv", index=False)
        print(f"Saved: {DATA_DIR / 'shap_values.csv'}")


if __name__ == "__main__":
    main()
