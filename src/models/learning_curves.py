"""Learning curves: accuracy vs. training set size for LR and LightGBM."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

from src.features.features import build_dataset
from src.models.model import build_gbm_pipeline, build_lr_pipeline
from src.paths import FIGURES_DIR

logger = logging.getLogger(__name__)


def _plot_curve(
    ax: plt.Axes,
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str,
) -> None:
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="#2980b9", label="Train score")
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2980b9"
    )
    ax.plot(train_sizes, test_mean, "s-", color="#e67e22", label="CV score")
    ax.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#e67e22"
    )
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="0.90 reference")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.02)


def main() -> None:
    ds = build_dataset(min_train_year=1994)
    X, y = ds.X, ds.y

    train_sizes_frac = np.linspace(0.05, 1.0, 15)

    logger.info("Computing LR learning curve...")
    train_sizes_lr, train_scores_lr, test_scores_lr = (
        learning_curve(  # pyright: ignore[reportAssignmentType]
            build_lr_pipeline(),
            X,
            y,
            train_sizes=train_sizes_frac,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
    )

    logger.info("Computing LightGBM learning curve...")
    train_sizes_gbm, train_scores_gbm, test_scores_gbm = (
        learning_curve(  # pyright: ignore[reportAssignmentType]
            build_gbm_pipeline(),
            X,
            y,
            train_sizes=train_sizes_frac,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
    )

    logger.info(
        "LR        — final train: %.4f, CV: %.4f",
        train_scores_lr[-1].mean(),
        test_scores_lr[-1].mean(),
    )
    logger.info(
        "LightGBM  — final train: %.4f, CV: %.4f",
        train_scores_gbm[-1].mean(),
        test_scores_gbm[-1].mean(),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _plot_curve(ax1, train_sizes_lr, train_scores_lr, test_scores_lr, "Logistic Regression")
    _plot_curve(ax2, train_sizes_gbm, train_scores_gbm, test_scores_gbm, "LightGBM")
    fig.suptitle("Learning curves — US House district classifier", y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    plt.close(fig)


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    main()
