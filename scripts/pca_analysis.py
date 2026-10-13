"""PCA analysis of the House election feature matrix.

Usage:
    python scripts/pca_analysis.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features import build_dataset
from src.paths import OUTPUTS_DIR
OUTPUT_PATH = OUTPUTS_DIR / "pca_plot.png"
N_COMPONENTS = 10
TOP_N_FEATURES = 5


def main() -> None:
    ds = build_dataset()
    X, y = ds.X, ds.y

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
