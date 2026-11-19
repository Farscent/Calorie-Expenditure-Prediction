"""Figures regenerated from development data and the frozen holdout evaluation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(X_dev, y_dev, actual, predicted, importance, destination: Path):
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlesize": 12})
    blue, orange = "#2459a6", "#da603c"
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].hist(y_dev, bins=35, color=blue, edgecolor="white")
    axes[0].set(title="Development target distribution", xlabel="Calories (dataset units)", ylabel="Rows")
    axes[1].scatter(X_dev.Duration, y_dev, s=5, alpha=0.2, color=blue, rasterized=True)
    axes[1].set(title="Development observations", xlabel="Duration (as supplied)", ylabel="Calories (dataset units)")
    figure.savefig(destination / "development_eda.png", bbox_inches="tight")
    plt.close(figure)

    correlation = X_dev.select_dtypes(include="number").assign(Calories=y_dev).corr()
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set(xticks=np.arange(len(correlation)), yticks=np.arange(len(correlation)),
             xticklabels=correlation.columns, yticklabels=correlation.columns,
             title="Development Pearson correlations (not causal effects)")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(correlation)):
        for col in range(len(correlation)):
            value = correlation.iloc[row, col]
            axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if abs(value) > 0.6 else "black")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(destination / "development_correlations.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    residual = np.asarray(actual) - np.asarray(predicted)
    axes[0].scatter(actual, predicted, s=8, alpha=0.4, color=blue, rasterized=True)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(limits, limits, color=orange, linewidth=1)
    axes[0].set(title="Holdout: predicted vs. actual", xlabel="Actual Calories", ylabel="Predicted Calories")
    axes[1].scatter(predicted, residual, s=8, alpha=0.4, color=blue, rasterized=True)
    axes[1].axhline(0, color=orange, linewidth=1)
    axes[1].set(title="Holdout residuals", xlabel="Predicted Calories", ylabel="Actual minus predicted")
    bins = [residual.mean() - 0.5, residual.mean() + 0.5] if np.ptp(residual) < 1e-8 else 35
    axes[2].hist(residual, bins=bins, color=blue, edgecolor="white")
    axes[2].set(title="Holdout error distribution", xlabel="Actual minus predicted", ylabel="Rows")
    figure.savefig(destination / "holdout_diagnostics.png", bbox_inches="tight")
    plt.close(figure)

    ranked = importance.sort_values("rmse_increase_mean")
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.barh(ranked.feature, ranked.rmse_increase_mean, xerr=ranked.rmse_increase_std, color=blue, capsize=3)
    axis.set(title="Holdout permutation importance: model reliance, not causality",
             xlabel="Increase in RMSE after shuffling a raw input")
    figure.savefig(destination / "permutation_importance.png", bbox_inches="tight")
    plt.close(figure)
