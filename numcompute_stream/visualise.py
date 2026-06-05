"""visualise.py -- Reusable matplotlib plotting helpers.
"""
from __future__ import annotations

import numpy as np


def _backend(save_path, show):
    import matplotlib

    if save_path is not None and not show:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_metric_over_time(metric_values, title="Metric over time",
                          ylabel="value", xlabel="chunk", save_path=None,
                          show=True):
    """Line plot of one metric across stream chunks.
    """
    plt = _backend(save_path, show)
    vals = np.asarray(metric_values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(vals.size), vals, marker="o", ms=3, lw=1.5, color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def compare_models(metric1, metric2, labels=("model A", "model B"),
                   title="Model comparison", ylabel="accuracy", xlabel="chunk",
                   save_path=None, show=True):
    """Overlay two streaming metric curves for comparison."""
    plt = _backend(save_path, show)
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(m1.size), m1, lw=1.8, label=labels[0], color="#2b6cb0")
    ax.plot(np.arange(m2.size), m2, lw=1.8, label=labels[1], color="#c05621")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs truth",
                                     save_path=None, show=True):
    """Scatter of predictions against ground truth for the latest chunk.
    """
    plt = _backend(save_path, show)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    correct = y_true == y_pred
    rng = np.random.default_rng(0)
    jt = rng.uniform(-0.12, 0.12, size=y_true.size)
    jp = rng.uniform(-0.12, 0.12, size=y_true.size)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_true[correct] + jt[correct], y_pred[correct] + jp[correct],
               s=18, c="#38a169", alpha=0.6, label="correct")
    ax.scatter(y_true[~correct] + jt[~correct], y_pred[~correct] + jp[~correct],
               s=22, c="#e53e3e", alpha=0.7, label="wrong")
    lim = [min(y_true.min(), y_pred.min()) - 0.5,
           max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, "--", color="#718096", lw=1)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("prediction")
    ax.set_title(f"{title} (acc={correct.mean():.3f})")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(matrix, labels=None, title="Confusion matrix",
                          save_path=None, show=True):
    """Heat-map of a confusion matrix (bonus helper)."""
    plt = _backend(save_path, show)
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(m, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = np.arange(m.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig