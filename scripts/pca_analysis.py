"""PCA analysis of the House election feature matrix.

Usage:
    python scripts/pca_analysis.py
"""
import os
import sys
from pathlib import Path

# Allow imports from the project root when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features import (
    add_incumbency_features,
    build_dems_reps,
    build_feature_matrix,
    build_winners,
    get_top_candidate_per_party,
)
from src.fundraising import load_fundraising
from src.loaders import load_approval, load_clean_house_data, load_pres_lean

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "pca_plot.png"
N_COMPONENTS = 10
TOP_N_FEATURES = 5


def main():
    # ── Load data via the project pipeline ──────────────────────────────────
    df = load_clean_house_data(DATA_DIR / "1976-2024-house.tab")
    top_candidates = get_top_candidate_per_party(df)
    dems_reps = build_dems_reps(top_candidates)
    approval = load_approval(DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls")
    pres_lean = load_pres_lean(
        DATA_DIR / "president-2008-2020-by-district.csv",
        DATA_DIR / "president-2020-2024-by-district.csv",
    )
    fundraising = load_fundraising(
        cache_path=DATA_DIR / "fec_fundraising.csv",
        api_key=os.getenv("FEC_API_KEY"),
    )
    winners = build_winners(top_candidates, dems_reps, approval, pres_lean, fundraising)
    winners = add_incumbency_features(winners, df)
    X, y, _ = build_feature_matrix(winners)

    feature_names = list(X.columns)
    print(f"\nFeature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    # ── StandardScaler → PCA ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(N_COMPONENTS, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # ── Explained variance ───────────────────────────────────────────────────
    print(f"\nExplained variance ratio ({n_components} components):")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        cumulative = pca.explained_variance_ratio_[:i].sum()
        print(f"  PC{i:>2}: {var:.4f}  (cumulative: {cumulative:.4f})")

    # ── Top features for PC1 and PC2 ─────────────────────────────────────────
    for pc_idx, pc_label in [(0, "PC1"), (1, "PC2")]:
        loadings = pca.components_[pc_idx]
        top_indices = np.argsort(np.abs(loadings))[::-1][:TOP_N_FEATURES]
        print(f"\nTop {TOP_N_FEATURES} features by absolute loading for {pc_label}:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. {feature_names[idx]:<40s}  loading={loadings[idx]:+.4f}")

    # ── 2D scatter plot (PC1 vs PC2, coloured by dem_won) ────────────────────
    colours = ["#cc3333" if label == 0 else "#3366cc" for label in y.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colours, alpha=0.3, s=8, linewidths=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of House election features (blue=Dem win, red=Rep win)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#3366cc", label="Dem win"),
            Patch(color="#cc3333", label="Rep win"),
        ],
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nScatter plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
