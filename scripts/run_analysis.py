"""Run the full diagnostic analysis pipeline.

Calls every analysis script in sequence and saves all outputs to outputs/.
Safe to re-run — existing outputs are overwritten.

Usage:
    python scripts/run_analysis.py
"""

import time

from scripts.calibration_analysis import main as calibration
from scripts.feature_significance import main as feature_significance
from scripts.learning_curves import main as learning_curves
from scripts.ols_analysis import main as ols
from scripts.pca_analysis import main as pca
from scripts.shap_analysis import main as shap

_STEPS = [
    ("Feature significance",  feature_significance),
    ("OLS / logit analysis",  ols),
    ("PCA analysis",          pca),
    ("Calibration analysis",  calibration),
    ("Learning curves",       learning_curves),
    ("SHAP analysis",         shap),
]


def main() -> None:
    total_start = time.time()
    print(f"Running {len(_STEPS)} analysis steps\n" + "=" * 50)

    for name, fn in _STEPS:
        print(f"\n>>> {name}")
        t = time.time()
        fn()
        print(f"    done in {time.time() - t:.1f}s")

    print(f"\n{'=' * 50}")
    print(f"All done in {time.time() - total_start:.1f}s — outputs/ folder updated.")


if __name__ == "__main__":
    main()
