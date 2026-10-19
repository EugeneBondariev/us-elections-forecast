"""Learning curves: accuracy vs. training set size for LR and LightGBM."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

from src.features.features import build_dataset
from src.models.model import build_gbm_pipeline, build_lr_pipeline
from src.paths import FIGURES_DIR


def _plot_curve(ax: plt.Axes, train_sizes: np.ndarray, train_scores: np.ndarray, cv_scores: np.ndarray, title: str) -> None:
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    cv_mean = cv_scores.mean(axis=1)
    cv_std = cv_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="#2980b9", label="Train score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2980b9")
    ax.plot(train_sizes, cv_mean, "s-", color="#e67e22", label="CV score")
    ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="#e67e22")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="0.90 reference")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.02)


def main() -> None:
    ds = build_dataset()
    X, y = ds.X, ds.y

    train_sizes_frac = np.linspace(0.1, 1.0, 10)

    print("Computing LR learning curve...")
    ts_lr, tr_lr, cv_lr = learning_curve(
        build_lr_pipeline(), X, y,
        train_sizes=train_sizes_frac, cv=5, scoring="accuracy", n_jobs=-1,
    )

    print("Computing LightGBM learning curve...")
    ts_gbm, tr_gbm, cv_gbm = learning_curve(
        build_gbm_pipeline(), X, y,
        train_sizes=train_sizes_frac, cv=5, scoring="accuracy", n_jobs=-1,
    )

    print(f"LR        — final train: {tr_lr[-1].mean():.4f}, CV: {cv_lr[-1].mean():.4f}")
    print(f"LightGBM  — final train: {tr_gbm[-1].mean():.4f}, CV: {cv_gbm[-1].mean():.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _plot_curve(ax1, ts_lr, tr_lr, cv_lr, "Logistic Regression")
    _plot_curve(ax2, ts_gbm, tr_gbm, cv_gbm, "LightGBM")
    fig.suptitle("Learning curves — US House district classifier", y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
