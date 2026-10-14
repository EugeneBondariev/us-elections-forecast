"""Statistical hypothesis testing for House election features.

Tests whether each feature in the model is a statistically significant
predictor of dem_won via:
  1. Point-biserial correlation + two-sample t-test (continuous features)
  2. Permutation test on top-5 features by absolute correlation
  3. Chi-square test (binary / one-hot features)

Saves a bar chart of -log10(p-value) for the top 20 features to
data/feature_significance.png.

Usage:
    python -m src.features.feature_significance
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.features.features import build_dataset
from src.paths import FIGURES_DIR

OUTPUT_PATH = FIGURES_DIR / "feature_significance.png"
N_PERMUTATIONS = 1000
TOP_N_PERMUTATION = 5
TOP_N_CHART = 20
ALPHA = 0.05


# ── helpers ──────────────────────────────────────────────────────────────────


def _sig(p: float) -> str:
    return "YES" if p < ALPHA else "no"


# ── 1. Continuous features: correlation + t-test ─────────────────────────────


def test_continuous(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        group1 = X.loc[y == 1, col].values
        group0 = X.loc[y == 0, col].values
        corr, _ = stats.pearsonr(X[col].values, y.values)
        t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)
        rows.append(
            {
                "feature": col,
                "correlation": round(corr, 4),
                "t_statistic": round(t_stat, 4),
                "p_value": p_val,
                "significant": _sig(p_val),
            }
        )
    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return df


# ── 2. Permutation test on top-N continuous features ─────────────────────────


def permutation_test(
    X: pd.DataFrame, y: pd.Series, features: list[str], n_perm: int = N_PERMUTATIONS
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    y_arr = y.to_numpy()
    rows = []
    for col in features:
        x_arr = X[col].values
        obs_corr, _ = stats.pearsonr(x_arr, y_arr)
        obs_abs = abs(obs_corr)

        perm_corrs = np.empty(n_perm)
        for i in range(n_perm):
            y_perm = rng.permutation(y_arr)
            perm_corrs[i], _ = stats.pearsonr(x_arr, y_perm)

        p_val = float(np.mean(np.abs(perm_corrs) >= obs_abs))
        rows.append(
            {
                "feature": col,
                "observed_correlation": round(obs_corr, 4),
                "permutation_p_value": p_val,
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_p_value").reset_index(drop=True)


# ── 3. Binary features: chi-square test ──────────────────────────────────────


def test_binary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        contingency = pd.crosstab(X[col], y)
        # Ensure both rows (0/1) and both columns (0/1) are present
        contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": col,
                "chi2": round(chi2, 4),
                "p_value": p_val,
                "significant": _sig(p_val),
            }
        )
    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return df


# ── plotting ─────────────────────────────────────────────────────────────────


def save_significance_chart(
    continuous_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_N_CHART,
) -> None:
    """Bar chart of -log10(p) for the top_n features across both test types."""
    # Build a combined view: one p-value per feature (use t-test for continuous,
    # chi2 for binary).
    combined = pd.concat(
        [
            continuous_df[["feature", "p_value"]],
            binary_df[["feature", "p_value"]],
        ],
        ignore_index=True,
    )
    # Clamp zero p-values so -log10 does not blow up.
    combined["p_value"] = combined["p_value"].clip(lower=1e-300)
    combined["neg_log10_p"] = -np.log10(combined["p_value"])
    combined = combined.sort_values("neg_log10_p", ascending=False).head(top_n)
    combined = combined.sort_values("neg_log10_p", ascending=True)  # ascending for hbar

    significance_threshold = -np.log10(ALPHA)  # = 1.301 for p=0.05

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = [
        "#3366cc" if v >= significance_threshold else "#aaaaaa" for v in combined["neg_log10_p"]
    ]
    ax.barh(combined["feature"], combined["neg_log10_p"], color=colors, height=0.7)
    ax.axvline(
        significance_threshold,
        color="#cc3333",
        linestyle="--",
        linewidth=1.2,
        label=f"p = {ALPHA} threshold",
    )
    ax.set_xlabel("-log10(p-value)")
    ax.set_title(
        f"Feature significance: top {top_n} features\n"
        "(blue = significant at p<0.05, grey = not significant)"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    ds = build_dataset()
    X, y = ds.X, ds.y

    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"dem_won base rate: {y.mean():.3f}\n")

    # ── Identify binary vs continuous features ────────────────────────────────
    binary_features = [
        c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]
    continuous_features = [c for c in X.columns if c not in binary_features]

    print(f"Continuous features : {len(continuous_features)}")
    print(f"Binary features     : {len(binary_features)}\n")

    # ── 1. Continuous: correlation + t-test ───────────────────────────────────
    print("=" * 70)
    print("1. POINT-BISERIAL CORRELATION + T-TEST (continuous features)")
    print("=" * 70)
    cont_results = test_continuous(X[continuous_features], y)
    print(cont_results.to_string(index=False))

    # ── 2. Permutation test on top-5 continuous features ─────────────────────
    print(f"\n{'=' * 70}")
    print(
        f"2. PERMUTATION TEST (top {TOP_N_PERMUTATION} features by |correlation|, "
        f"n_permutations={N_PERMUTATIONS})"
    )
    print("=" * 70)
    # Re-sort by absolute correlation to pick the genuinely strongest features.
    top5_abs = (
        cont_results.assign(abs_corr=cont_results["correlation"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(TOP_N_PERMUTATION)["feature"]
        .tolist()
    )
    perm_results = permutation_test(X[continuous_features], y, top5_abs)
    print(perm_results.to_string(index=False))

    # ── 3. Binary: chi-square test ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("3. CHI-SQUARE TEST (binary / one-hot features)")
    print("=" * 70)
    bin_results = test_binary(X[binary_features], y)
    print(bin_results.to_string(index=False))

    # ── Summary counts ────────────────────────────────────────────────────────
    n_sig_cont = (cont_results["p_value"] < ALPHA).sum()
    n_sig_bin = (bin_results["p_value"] < ALPHA).sum()
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(
        f"  Significant continuous features (p<{ALPHA}): "
        f"{n_sig_cont}/{len(continuous_features)}"
    )
    print(f"  Significant binary features     (p<{ALPHA}): " f"{n_sig_bin}/{len(binary_features)}")

    # ── Save chart ────────────────────────────────────────────────────────────
    save_significance_chart(cont_results, bin_results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
