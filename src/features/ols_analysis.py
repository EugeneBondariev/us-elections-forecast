"""
OLS and Logit regression analysis of House election outcomes (statsmodels).

Demonstrates econometric interpretation of the forecast's structural features:
 - Linear Probability Model (OLS) with HC3 heteroskedasticity-robust SEs
 - Logit with average marginal effects (AME)
 - Coefficient comparison plot

Saves: data/ols_summary.txt, data/ols_coef_comparison.png
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features.features import build_dataset
from src.paths import FIGURES_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Interpretable structural features — excludes state/party one-hot dummies
STRUCTURAL_FEATURES = [
    "dem_lean",
    "prev_dem_vote_share",
    "pres_approval",
    "pres_disapproval",
    "generic_ballot",
    "econ_growth",
    "unemployment",
    "is_incumbent_on_ballot",
    "inc_x_ballot",
    "redistricting",
    "dem_lean_trend",
]

FEATURE_LABELS = {
    "dem_lean":                  "Hist. Dem vote share (4yr avg)",
    "prev_dem_vote_share":       "Previous Dem vote share",
    "pres_approval":             "Presidential approval",
    "pres_disapproval":          "Presidential disapproval",
    "generic_ballot":            "Generic ballot margin (D–R)",
    "econ_growth":               "GDP growth (sign-adjusted)",
    "unemployment":              "Unemployment (sign-adjusted)",
    "is_incumbent_on_ballot":    "Incumbent running (0/1)",
    "inc_x_ballot":              "Incumbent × Generic ballot",
    "redistricting":             "Redistricting year (0/1)",
    "dem_lean_trend":            "Dem lean trend (4yr slope)",
}


def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ds = build_dataset()
    return ds.X, ds.y, ds.years


def main() -> None:
    figures_dir = FIGURES_DIR
    reports_dir = REPORTS_DIR
    log.info("Loading data...")
    X_full, y, _ = _load_data()

    # Restrict to structural features only for interpretable econometric analysis
    available = [f for f in STRUCTURAL_FEATURES if f in X_full.columns]
    X = X_full[available].copy()
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    log.info("Sample: %d district-elections, %d structural features", len(X), len(available))
    log.info("Features: %s", available)
    log.info("Share Dem wins: %.1f%%", y.mean() * 100)

    X_sm = sm.add_constant(X)

    # -----------------------------------------------------------------------
    # 1. Linear Probability Model (OLS with HC3 robust SEs)
    # -----------------------------------------------------------------------
    # OLS on a binary outcome = Linear Probability Model.
    # Advantage: coefficients are directly interpretable as ΔP(Dem win).
    # Limitation: predicted values can fall outside [0,1]; heteroskedastic by
    # construction → use HC3 (heteroskedasticity-robust) standard errors.
    log.info("\n--- Linear Probability Model ---")
    ols = sm.OLS(y, X_sm).fit(cov_type="HC3")
    print(ols.summary(
        title="Linear Probability Model (HC3 robust SEs)",
        xname=["const"] + [FEATURE_LABELS.get(f, f) for f in available],
    ))

    # -----------------------------------------------------------------------
    # 2. Logit Model
    # -----------------------------------------------------------------------
    # Proper binary model; coefficients are log-odds changes.
    # Use average marginal effects (AME) for comparable magnitudes to OLS.
    log.info("\n--- Logit Model ---")
    logit = sm.Logit(y, X_sm).fit(disp=False, maxiter=200)
    print(logit.summary(
        title="Logit Model",
        xname=["const"] + [FEATURE_LABELS.get(f, f) for f in available],
    ))

    log.info("\n--- Average Marginal Effects (Logit) ---")
    ame = logit.get_margeff(at="overall")
    print(ame.summary())

    # -----------------------------------------------------------------------
    # 3. Compare: OLS coefficients vs Logit AME
    # -----------------------------------------------------------------------
    ols_coef = ols.params.drop("const")
    ols_ci = ols.conf_int().drop("const")
    ols_ci.columns = ["lo", "hi"]

    ame_df = ame.summary_frame()
    # Map pretty labels back to short names for join
    label_to_feat = {v: k for k, v in FEATURE_LABELS.items()}
    ame_df.index = [label_to_feat.get(i, i) for i in ame_df.index]
    common = [f for f in available if f in ame_df.index]

    ame_vals = ame_df.loc[common, "dy/dx"]
    ame_lo   = ame_df.loc[common, "Conf. Int. Low"]
    ame_hi   = ame_df.loc[common, "Cont. Int. Hi."]

    labels = [FEATURE_LABELS.get(f, f) for f in common]
    n = len(common)
    y_pos = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.55 + 1)))

    def _barh(ax, vals, ci_lo, ci_hi, title, xlabel):
        colors = ["steelblue" if v > 0 else "tomato" for v in vals]
        err_lo = np.maximum(vals - ci_lo, 0)
        err_hi = np.maximum(ci_hi - vals, 0)
        ax.barh(y_pos, vals, xerr=[err_lo, err_hi], align="center",
                color=colors, alpha=0.75, height=0.6, ecolor="black", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        ax.tick_params(axis="x", labelsize=8)

    _barh(
        axes[0],
        ols_coef.loc[common].values,
        ols_ci.loc[common, "lo"].values,
        ols_ci.loc[common, "hi"].values,
        "OLS (Linear Probability Model)\nHC3 robust 95% CI",
        "ΔP(Dem win) per unit  [OLS coef]",
    )
    _barh(
        axes[1],
        ame_vals.values,
        ame_lo.values,
        ame_hi.values,
        "Logit Average Marginal Effects\n95% CI",
        "∂P(Dem win) / ∂x  [AME]",
    )

    fig.suptitle(
        "Structural drivers of Dem House win probability (1976–2024)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out_png = figures_dir / "ols_coef_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved → %s", out_png)

    # Save text summaries
    out_txt = reports_dir / "ols_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(ols.summary(
            title="Linear Probability Model (HC3 robust SEs)",
            xname=["const"] + [FEATURE_LABELS.get(f, f) for f in available],
        ).as_text())
        fh.write("\n\n")
        fh.write(logit.summary(
            title="Logit Model",
            xname=["const"] + [FEATURE_LABELS.get(f, f) for f in available],
        ).as_text())
        fh.write("\n\n")
        fh.write(ame.summary().as_text())
    log.info("Text summary saved → %s", out_txt)


if __name__ == "__main__":
    main()
