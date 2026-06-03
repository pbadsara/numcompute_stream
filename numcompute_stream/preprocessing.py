"""preprocessing.py -- Streaming transformers.

* :class:`StandardScaler` -- running standardisation.
* :class:`Imputer` -- fills NaNs using a running mean / median / most-frequent value.
* :class:`OneHotEncoder` -- expands categorical columns, growing the category set
  incrementally as unseen levels arrive.
"""
from __future__ import annotations

import numpy as np

from .stats import StreamingStats, StreamingHistogram


class StandardScaler:
    """Standardise features to zero mean / unit variance in a streaming fashion.
    """

    def __init__(self, with_mean=True, with_std=True, ema_alpha: float | None = None):
        self.with_mean = with_mean
        self.with_std = with_std
        self.ema_alpha = ema_alpha
        self._stats = StreamingStats()
        self.mean_ = None
        self.var_ = None

    def partial_fit(self, X: np.ndarray, y=None) -> "StandardScaler":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.ema_alpha is None:
            self._stats.update_stats(X)
            self.mean_ = self._stats.mean_.copy()
            self.var_ = self._stats.variance.copy()
        else:
            a = self.ema_alpha
            m = np.nanmean(np.where(np.isnan(X), np.nan, X), axis=0)
            v = np.nanvar(X, axis=0)
            m = np.where(np.isnan(m), 0.0, m)
            v = np.where(np.isnan(v), 0.0, v)
            if self.mean_ is None:
                self.mean_, self.var_ = m, v
            else:
                self.mean_ = (1 - a) * self.mean_ + a * m
                self.var_ = (1 - a) * self.var_ + a * v
        return self

    def fit(self, X, y=None) -> "StandardScaler":
        self._stats = StreamingStats()
        self.mean_ = self.var_ = None
        return self.partial_fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = X.copy()
        if self.with_mean:
            out = out - self.mean_
        if self.with_std:
            # zero-variance features get scale 1 to avoid division by zero.
            scale = np.sqrt(self.var_)
            scale = np.where(scale == 0, 1.0, scale)
            out = out / scale
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Imputer:
    """Replace missing values (NaN) with a running statistic.
    """

    def __init__(self, strategy: str = "mean"):
        if strategy not in {"mean", "median", "most_frequent"}:
            raise ValueError(f"unknown strategy {strategy!r}.")
        self.strategy = strategy
        self._stats = StreamingStats()
        self._hist: list[StreamingHistogram] | None = None
        self._freq: list[dict] | None = None
        self.statistics_ = None

    def partial_fit(self, X: np.ndarray, y=None) -> "Imputer":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = X.shape[1]
        if self.strategy == "mean":
            self._stats.update_stats(X)
            self.statistics_ = self._stats.mean_.copy()
        elif self.strategy == "median":
            if self._hist is None:
                self._hist = [StreamingHistogram(bins=64) for _ in range(d)]
            for j in range(d):
                self._hist[j].update(X[:, j])
            self.statistics_ = np.array(
                [h.quantile(0.5) for h in self._hist], dtype=float
            )
        else:  # most_frequent
            if self._freq is None:
                self._freq = [dict() for _ in range(d)]
            for j in range(d):
                col = X[:, j]
                col = col[~np.isnan(col)]
                vals, cnts = np.unique(col, return_counts=True)
                for v, c in zip(vals, cnts):
                    self._freq[j][v] = self._freq[j].get(v, 0) + int(c)
            self.statistics_ = np.array(
                [max(f, key=f.get) if f else 0.0 for f in self._freq], dtype=float
            )
        return self

    def fit(self, X, y=None) -> "Imputer":
        self.__init__(self.strategy)
        return self.partial_fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("Imputer must be fitted before transform().")
        X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(self.statistics_, idx[1])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OneHotEncoder:
    """Incremental one-hot encoder that grows its category set over the stream.
    """

    def __init__(self, handle_unknown: str = "ignore"):
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        self.handle_unknown = handle_unknown
        self.categories_: list[list] = []

    def partial_fit(self, X: np.ndarray, y=None) -> "OneHotEncoder":
        X = np.atleast_2d(np.asarray(X))
        d = X.shape[1]
        if not self.categories_:
            self.categories_ = [[] for _ in range(d)]
        for j in range(d):
            known = self.categories_[j]
            seen = set(known)
            for v in X[:, j]:
                key = v.item() if hasattr(v, "item") else v
                if key not in seen and not (isinstance(key, float) and np.isnan(key)):
                    known.append(key)
                    seen.add(key)
        return self

    def fit(self, X, y=None) -> "OneHotEncoder":
        self.categories_ = []
        return self.partial_fit(X, y)

    @property
    def n_output_features_(self) -> int:
        return sum(len(c) for c in self.categories_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.categories_:
            raise RuntimeError("OneHotEncoder must be fitted before transform().")
        X = np.atleast_2d(np.asarray(X))
        n = X.shape[0]
        blocks = []
        for j, cats in enumerate(self.categories_):
            block = np.zeros((n, len(cats)), dtype=float)
            index = {c: i for i, c in enumerate(cats)}
            for r, v in enumerate(X[:, j]):
                key = v.item() if hasattr(v, "item") else v
                if key in index:
                    block[r, index[key]] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"unknown category {key!r} in column {j}.")
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.empty((n, 0))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)