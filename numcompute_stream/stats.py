"""stats.py -- Streaming statistical estimators.

Every estimator consumes data in chunks and never stores the full stream.

* :class:`StreamingStats` -- per-feature count / mean / variance / min / max via
  the numerically stable Welford (Chan parallel) update; NaN-aware.
* :class:`StreamingHistogram` -- fixed-bin histogram with optional sliding window,
  yielding approximate quantiles from the cumulative distribution.
"""
from __future__ import annotations

import numpy as np


def _safe_div(a, b):
    """Element-wise ``a / b`` returning 0 where ``b == 0`` (no warnings)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


class StreamingStats:
    """Welford running mean/variance per feature, ignoring NaNs.
    """

    def __init__(self) -> None:
        self.n_ = None  # (d,) non-NaN counts per feature
        self.mean_ = None  # (d,)
        self.M2_ = None  # (d,) sum of squared deviations
        self.min_ = None  # (d,)
        self.max_ = None  # (d,)

    def _init(self, d: int) -> None:
        self.n_ = np.zeros(d)
        self.mean_ = np.zeros(d)
        self.M2_ = np.zeros(d)
        self.min_ = np.full(d, np.inf)
        self.max_ = np.full(d, -np.inf)

    def update_stats(self, X: np.ndarray) -> "StreamingStats":
        """Update running statistics with a chunk ``X`` of shape (m, d)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.mean_ is None:
            self._init(X.shape[1])
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"feature mismatch: got {X.shape[1]} columns, "
                f"expected {self.mean_.shape[0]}."
            )

        mask = ~np.isnan(X)
        nb = mask.sum(axis=0).astype(float)  # (d,)
        Xz = np.where(mask, X, 0.0)
        mean_b = _safe_div(Xz.sum(axis=0), nb)  # (d,)
        diff = np.where(mask, X - mean_b, 0.0)
        M2_b = (diff**2).sum(axis=0)  # (d,)

        na = self.n_
        n = na + nb
        delta = mean_b - self.mean_
        new_mean = self.mean_ + delta * _safe_div(nb, n)
        new_M2 = self.M2_ + M2_b + delta**2 * _safe_div(na * nb, n)

        upd = nb > 0
        self.mean_ = np.where(upd, new_mean, self.mean_)
        self.M2_ = np.where(upd, new_M2, self.M2_)
        self.n_ = n

        cmin = np.where(mask, X, np.inf).min(axis=0)
        cmax = np.where(mask, X, -np.inf).max(axis=0)
        self.min_ = np.minimum(self.min_, cmin)
        self.max_ = np.maximum(self.max_, cmax)
        return self

    @property
    def variance(self) -> np.ndarray:
        """Sample variance (ddof=1); 0 where fewer than two observations."""
        return _safe_div(self.M2_, np.maximum(self.n_ - 1.0, 0.0))

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def count(self) -> np.ndarray:
        return self.n_


class StreamingHistogram:
    """Fixed-bin streaming histogram with optional sliding window.
    """

    def __init__(self, bins: int = 32, value_range=None, window: int | None = None):
        self.bins = int(bins)
        self.range = value_range
        self.window = window
        self.edges_ = None
        self.counts_ = np.zeros(self.bins, dtype=float)
        self._chunks: list[np.ndarray] = []

    def update(self, x: np.ndarray) -> "StreamingHistogram":
        """Add 1-D data ``x`` (NaNs ignored) to the histogram."""
        x = np.asarray(x, dtype=float).ravel()
        x = x[~np.isnan(x)]
        if self.edges_ is None:
            if self.range is None:
                lo, hi = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
                if lo == hi:
                    hi = lo + 1.0
            else:
                lo, hi = self.range
            self.edges_ = np.linspace(lo, hi, self.bins + 1)
        c, _ = np.histogram(x, bins=self.edges_)
        if self.window is None:
            self.counts_ += c
        else:
            self._chunks.append(c.astype(float))
            self._chunks = self._chunks[-self.window :]
            self.counts_ = np.sum(self._chunks, axis=0)
        return self

    def quantile(self, q: float) -> float:
        """Approximate the ``q``-quantile (0..1) via cumulative bin counts."""
        if self.edges_ is None or self.counts_.sum() == 0:
            return float("nan")
        total = self.counts_.sum()
        cum = np.cumsum(self.counts_)
        target = q * total
        b = int(np.searchsorted(cum, target))
        b = min(b, self.bins - 1)
        prev = cum[b - 1] if b > 0 else 0.0
        frac = (target - prev) / self.counts_[b] if self.counts_[b] > 0 else 0.0
        lo, hi = self.edges_[b], self.edges_[b + 1]
        return float(lo + frac * (hi - lo))


class P2Quantile:
    """Single-pass P-square quantile estimator.

    Estimates one quantile ``p`` using five markers and O(1) memory.
    """

    def __init__(self, p: float):
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie strictly in (0, 1).")
        self.p = p
        self.q = []  # marker heights
        self.n = [1, 2, 3, 4, 5]  # marker positions
        self.np_ = [1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5]  # desired positions
        self.dn = [0, p / 2, p, (1 + p) / 2, 1]
        self.count = 0

    def update(self, x) -> "P2Quantile":
        for v in np.asarray(x, dtype=float).ravel():
            if np.isnan(v):
                continue
            self._observe(float(v))
        return self

    def _observe(self, x: float) -> None:
        if len(self.q) < 5:
            self.q.append(x)
            self.count += 1
            if len(self.q) == 5:
                self.q.sort()
            return

        # 1. find cell k
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[4]:
            self.q[4] = x
            k = 3
        else:
            k = next(i for i in range(4) if self.q[i] <= x < self.q[i + 1])

        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np_[i] += self.dn[i]

        # 2. adjust interior markers
        for i in range(1, 4):
            d = self.np_[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or (
                d <= -1 and self.n[i - 1] - self.n[i] < -1
            ):
                s = int(np.sign(d))
                qi = self._parabolic(i, s)
                if self.q[i - 1] < qi < self.q[i + 1]:
                    self.q[i] = qi
                else:
                    self.q[i] = self._linear(i, s)
                self.n[i] += s
        self.count += 1

    def _parabolic(self, i: int, s: int) -> float:
        n = self.n
        q = self.q
        return q[i] + s / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + s) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - s) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, s: int) -> float:
        return self.q[i] + s * (self.q[i + s] - self.q[i]) / (
            self.n[i + s] - self.n[i]
        )

    def result(self) -> float:
        """Current quantile estimate (uses the sorted buffer if <5 samples)."""
        if not self.q:
            return float("nan")
        if len(self.q) < 5:
            arr = np.sort(self.q)
            idx = min(int(self.p * (len(arr) - 1) + 0.5), len(arr) - 1)
            return float(arr[idx])
        return float(self.q[2])