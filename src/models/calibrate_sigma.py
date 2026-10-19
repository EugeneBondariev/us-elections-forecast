"""Calibrate all four Monte Carlo simulation parameters from walk-forward residuals.

Run after any feature/data change that could shift historical walk-forward accuracy:
    python -m src.models.calibrate_sigma

Writes models/calibrated_sigma.json. src.model.load_simulation_params() picks it up
automatically, so run_forecast() (used by app.py, api.py, predict.py) uses the
calibrated values on the next run.
"""
import json
import logging

from src.features.features import build_dataset
from src.logging_config import setup_logging
from src.models.model import calibrate_simulation_params
from src.paths import DataFile

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ds = build_dataset()
    result = calibrate_simulation_params(ds.X, ds.y, ds.years)

    DataFile.CALIBRATED_SIGMA.parent.mkdir(parents=True, exist_ok=True)
    with open(DataFile.CALIBRATED_SIGMA, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved calibrated simulation params -> %s", DataFile.CALIBRATED_SIGMA)
    logger.info("  sigma_national = %.4f", result["sigma_national"])
    logger.info("  sigma_district = %.4f (held fixed — see calibrate_national_sigma docstring)",
                result["sigma_district"])
    logger.info("  concentration  = %.1f", result.get("concentration", float("nan")))
    logger.info("  blend_weight   = %.2f", result.get("blend_weight", float("nan")))
    logger.info("  target seat-count std = %.2f (from %d walk-forward years)",
                result["target_std"], result["n_years"])


if __name__ == "__main__":
    main()
