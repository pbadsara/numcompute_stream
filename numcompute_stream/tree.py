"""tree.py -- Online (Hoeffding) decision-tree classifier.

A classic batch tree cannot grow from chunks, so streaming growth is achieved
with a **Hoeffding tree** :

* Each leaf accumulates, per class, a Gaussian estimator (count / mean / M2 via
  Welford) for every feature plus the observed feature range.
* When a leaf has seen at least ``min_samples_split`` examples it evaluates
  candidate splits. The information gain (Gini or entropy) of the best and the
  runner-up attribute are compared against the **Hoeffding bound**
* ``max_features`` random attributes are considered at each leaf, which also
  enables Random-Forest-style ensembling on top of this tree.

The tree is depth-limited (``max_depth``) and exposes the standard
``partial_fit`` / ``predict`` / ``predict_proba`` / ``score`` API.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- 
# Vectorised numerical helpers
# -----------------------------------------------------------------------------
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised Abramowitz-Stegun (7.1.26) error function, |err| < 1.5e-7."""
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + _P * ax)
    poly = ((((_A[4] * t + _A[3]) * t + _A[2]) * t + _A[1]) * t + _A[0]) * t
    return s * (1.0 - poly * np.exp(-ax * ax))


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF, vectorised."""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    """Impurity of class-count rows.

    ``counts`` may be (K,) or (m, K); returns scalar or (m,).
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    if criterion == "gini":
        return 1.0 - np.sum(p**2, axis=-1)
    # entropy (natural log -> nats; constant factor is irrelevant for comparison)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -np.sum(p * logp, axis=-1)


# ----------------------------------------------------------------------------- 
# Leaf statistics
# -----------------------------------------------------------------------------
class _Leaf:
    """Sufficient statistics for one leaf node."""

    __slots__ = ("depth", "n_class", "stat_n", "stat_mean", "stat_M2",
                 "fmin", "fmax", "seen_since_eval")

    def __init__(self, depth: int, n_classes: int, n_features: int,
                 seed_counts: np.ndarray | None = None):
        self.depth = depth
        self.n_class = (
            seed_counts.astype(float).copy()
            if seed_counts is not None
            else np.zeros(n_classes, dtype=float)
        )
        # per (class, feature) Welford accumulators
        self.stat_n = np.zeros((n_classes, n_features))
        self.stat_mean = np.zeros((n_classes, n_features))
        self.stat_M2 = np.zeros((n_classes, n_features))
        self.fmin = np.full(n_features, np.inf)
        self.fmax = np.full(n_features, -np.inf)
        self.seen_since_eval = 0.0

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update leaf statistics with rows routed here (NaN features skipped)."""
        mask = ~np.isnan(X)
        self.fmin = np.minimum(self.fmin, np.where(mask, X, np.inf).min(0))
        self.fmax = np.maximum(self.fmax, np.where(mask, X, -np.inf).max(0))
        for c in np.unique(y):
            rows = y == c
            Xc = X[rows]
            mc = ~np.isnan(Xc)
            nb = mc.sum(0).astype(float)  # (d,)
            self.n_class[c] += rows.sum()
            Xz = np.where(mc, Xc, 0.0)
            mean_b = np.divide(Xz.sum(0), nb, out=np.zeros(Xc.shape[1]), where=nb > 0)
            diff = np.where(mc, Xc - mean_b, 0.0)
            M2_b = (diff**2).sum(0)
            na = self.stat_n[c]
            n = na + nb
            delta = mean_b - self.stat_mean[c]
            inc = np.divide(nb, n, out=np.zeros_like(n), where=n > 0)
            self.stat_mean[c] = np.where(nb > 0, self.stat_mean[c] + delta * inc,
                                         self.stat_mean[c])
            self.stat_M2[c] = np.where(
                nb > 0,
                self.stat_M2[c] + M2_b
                + delta**2 * np.divide(na * nb, n, out=np.zeros_like(n), where=n > 0),
                self.stat_M2[c],
            )
            self.stat_n[c] = n
        self.seen_since_eval += X.shape[0]

    def total(self) -> float:
        return float(self.n_class.sum())


class _Node:
    """Internal decision node."""

    __slots__ = ("feature", "threshold", "left", "right")

    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

