"""stream.py -- Streaming trainer / evaluation harness.
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from .metrics import Accuracy, RollingAccuracy


def model_size_bytes(model) -> int:
    """Rough in-memory footprint of a model via its pickled byte length."""
    try:
        return len(pickle.dumps(model))
    except Exception:  # pragma: no cover - defensive
        return -1


class StreamTrainer:
    """Manage model + logging across a stream of chunks.
    """

    def __init__(self, model, prequential: bool = True, rolling_window: int = 200):
        self.model = model
        self.prequential = prequential
        self._cum_acc = Accuracy()
        self._roll = RollingAccuracy(window=rolling_window)
        self.history = {
            "chunk": [], "chunk_accuracy": [], "cumulative_accuracy": [],
            "rolling_accuracy": [], "error": [], "memory_bytes": [],
            "fit_time_s": [], "n_seen": [],
        }
        self._n_seen = 0
        self._chunk_id = 0

    def score_chunk(self, X, y) -> float:
        """Predict on a chunk and update streaming metrics; returns accuracy."""
        y = np.asarray(y).ravel()
        if getattr(self.model, "classes_", None) is None:
            return float("nan")  # model has not seen any data yet
        pred = self.model.predict(X)
        self._cum_acc.update(y, pred)
        self._roll.update(y, pred)
        return float(np.mean(pred == y))

    def fit_chunk(self, X, y, classes=None):
        """Train the model on a chunk (timed)."""
        t0 = time.perf_counter()
        try:
            self.model.partial_fit(X, y, classes=classes)
        except TypeError:
            self.model.partial_fit(X, y)
        return time.perf_counter() - t0

    def run_chunk(self, X, y, classes=None) -> dict:
        """Process one chunk end-to-end and append a log record."""
        y = np.asarray(y).ravel()
        chunk_acc = self.score_chunk(X, y) if self.prequential else float("nan")
        fit_time = self.fit_chunk(X, y, classes=classes)
        if not self.prequential:
            chunk_acc = self.score_chunk(X, y)
        self._n_seen += X.shape[0]
        rec = {
            "chunk": self._chunk_id,
            "chunk_accuracy": chunk_acc,
            "cumulative_accuracy": self._cum_acc.result(),
            "rolling_accuracy": self._roll.result(),
            "error": 1.0 - chunk_acc if chunk_acc == chunk_acc else float("nan"),
            "memory_bytes": model_size_bytes(self.model),
            "fit_time_s": fit_time,
            "n_seen": self._n_seen,
        }
        for k, v in rec.items():
            self.history[k].append(v)
        self._chunk_id += 1
        return rec

    def run(self, chunks, classes=None) -> dict:
        """Consume an iterable of ``(X, y)`` chunks; returns the history dict."""
        for X, y in chunks:
            self.run_chunk(X, y, classes=classes)
        return self.history

    def summary(self) -> dict:
        """Headline numbers after the stream is consumed."""
        h = self.history
        return {
            "n_chunks": len(h["chunk"]),
            "n_samples": self._n_seen,
            "final_cumulative_accuracy": h["cumulative_accuracy"][-1] if h["chunk"] else float("nan"),
            "final_rolling_accuracy": h["rolling_accuracy"][-1] if h["chunk"] else float("nan"),
            "total_fit_time_s": float(np.sum(h["fit_time_s"])),
            "final_memory_bytes": h["memory_bytes"][-1] if h["chunk"] else -1,
        }