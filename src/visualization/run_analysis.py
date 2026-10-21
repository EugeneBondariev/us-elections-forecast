"""Run the full diagnostic analysis pipeline.

Calls every analysis script in sequence and saves all outputs to reports/ (reports/figures/ for images).
Console output is also written to logs/analysis/YYYYMMDD_HHMMSS.log.
Safe to re-run — existing outputs are overwritten.

Usage:
    python -m src.visualization.run_analysis
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from src.features.feature_significance import main as feature_significance
from src.features.ols_analysis import main as ols
from src.features.pca_analysis import main as pca
from src.models.learning_curves import main as learning_curves
from src.paths import ensure
from src.visualization.calibration_analysis import main as calibration
from src.visualization.shap_analysis import main as shap

_STEPS = [
    ("Feature significance", feature_significance),
    ("OLS / logit analysis", ols),
    ("PCA analysis", pca),
    ("Calibration analysis", calibration),
    ("Learning curves", learning_curves),
    ("SHAP analysis", shap),
]

_LOG_DIR = Path("logs/analysis")


class _Tee:
    """Write to both the real stdout and a log file simultaneously."""

    def __init__(self, log_path: Path) -> None:
        self._terminal = sys.stdout
        self._file = ensure(log_path).open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        try:
            self._terminal.write(message)
        except UnicodeEncodeError:
            enc = getattr(self._terminal, "encoding", "utf-8") or "utf-8"
            self._terminal.write(message.encode(enc, errors="replace").decode(enc))
        self._file.write(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def main() -> None:
    log_path = _LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        total_start = time.time()
        print(f"Running {len(_STEPS)} analysis steps\n" + "=" * 50)

        for name, fn in _STEPS:
            print(f"\n>>> {name}")
            t = time.time()
            fn()
            print(f"    done in {time.time() - t:.1f}s")

        print(f"\n{'=' * 50}")
        print(f"All done in {time.time() - total_start:.1f}s  —  log: {log_path}")
    finally:
        sys.stdout = tee._terminal
        tee.close()


if __name__ == "__main__":
    main()
