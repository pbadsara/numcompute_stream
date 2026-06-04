"""metrics.py -- Streaming classification metrics.
"""
from __future__ import annotations

import numpy as np


def _check(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    return y_true, y_pred


class StreamingMetric:
    """Base class establishing the update / reset / result interface."""

    def update(self, y_true, y_pred):  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self):  # pragma: no cover - interface
        raise NotImplementedError

    def result(self):  # pragma: no cover - interface
        raise NotImplementedError


class Accuracy(StreamingMetric):
    def __init__(self):
        self.reset()

    def reset(self):
        self._correct = 0
        self._total = 0
        return self

    def update(self, y_true, y_pred):
        y_true, y_pred = _check(y_true, y_pred)
        self._correct += int(np.sum(y_true == y_pred))
        self._total += y_true.size
        return self

    def result(self):
        return self._correct / self._total if self._total else float("nan")


class ConfusionMatrix(StreamingMetric):
    """Accumulating confusion matrix with a dynamic label set."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._index: dict = {}
        self.matrix = np.zeros((0, 0), dtype=np.int64)
        return self

    def _ensure(self, label):
        if label not in self._index:
            self._index[label] = len(self._index)
            k = len(self._index)
            m = np.zeros((k, k), dtype=np.int64)
            m[: k - 1, : k - 1] = self.matrix
            self.matrix = m

    def update(self, y_true, y_pred):
        y_true, y_pred = _check(y_true, y_pred)
        for t, p in zip(y_true, y_pred):
            t, p = t.item(), p.item()
            self._ensure(t)
            self._ensure(p)
            self.matrix[self._index[t], self._index[p]] += 1
        return self

    def result(self):
        return self.matrix

    @property
    def labels(self):
        return [k for k, _ in sorted(self._index.items(), key=lambda kv: kv[1])]

    def _per_class(self):
        m = self.matrix.astype(float)
        tp = np.diag(m)
        precision = np.divide(tp, m.sum(0), out=np.zeros_like(tp), where=m.sum(0) > 0)
        recall = np.divide(tp, m.sum(1), out=np.zeros_like(tp), where=m.sum(1) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        return precision, recall, f1


class _ConfusionDerived(StreamingMetric):
    """Shared base for Precision / Recall / F1."""

    def __init__(self, average: str = "macro"):
        if average not in {"macro", "micro", "none"}:
            raise ValueError("average must be 'macro', 'micro' or 'none'.")
        self.average = average
        self.cm = ConfusionMatrix()

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true, y_pred):
        self.cm.update(y_true, y_pred)
        return self

    def _reduce(self, per_class_values, micro_value):
        if self.average == "none":
            return per_class_values
        if self.average == "macro":
            return float(np.mean(per_class_values)) if per_class_values.size else 0.0
        return micro_value


class Precision(_ConfusionDerived):
    def result(self):
        precision, _, _ = self.cm._per_class()
        m = self.cm.matrix.astype(float)
        tp = np.diag(m).sum()
        micro = tp / m.sum() if m.sum() else 0.0  # micro precision == accuracy
        return self._reduce(precision, float(micro))


class Recall(_ConfusionDerived):
    def result(self):
        _, recall, _ = self.cm._per_class()
        m = self.cm.matrix.astype(float)
        tp = np.diag(m).sum()
        micro = tp / m.sum() if m.sum() else 0.0
        return self._reduce(recall, float(micro))


class F1(_ConfusionDerived):
    def result(self):
        _, _, f1 = self.cm._per_class()
        m = self.cm.matrix.astype(float)
        tp = np.diag(m).sum()
        micro = tp / m.sum() if m.sum() else 0.0
        return self._reduce(f1, float(micro))


class BinaryAUC(StreamingMetric):
    """Streaming ROC-AUC for binary problems via score histograms.
    """

    def __init__(self, bins: int = 200, positive_label=1):
        self.bins = bins
        self.positive_label = positive_label
        self.reset()

    def reset(self):
        self.edges = np.linspace(0.0, 1.0, self.bins + 1)
        self.pos = np.zeros(self.bins, dtype=float)
        self.neg = np.zeros(self.bins, dtype=float)
        return self

    def update(self, y_true, y_score):
        y_true, y_score = _check(y_true, y_score)
        y_score = np.clip(y_score.astype(float), 0.0, 1.0)
        is_pos = y_true == self.positive_label
        self.pos += np.histogram(y_score[is_pos], bins=self.edges)[0]
        self.neg += np.histogram(y_score[~is_pos], bins=self.edges)[0]
        return self

    def result(self):
        n_pos, n_neg = self.pos.sum(), self.neg.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # For each score bin, count negatives strictly below + half of ties.
        neg_below = np.concatenate([[0.0], np.cumsum(self.neg)[:-1]])
        wins = (self.pos * neg_below).sum() + 0.5 * (self.pos * self.neg).sum()
        return float(wins / (n_pos * n_neg))


class RollingAccuracy(StreamingMetric):
    """Accuracy over the most recent ``window`` samples (ring buffer)."""

    def __init__(self, window: int = 200):
        self.window = int(window)
        self.reset()

    def reset(self):
        self._buf = np.zeros(self.window, dtype=bool)
        self._i = 0
        self._filled = 0
        return self

    def update(self, y_true, y_pred):
        y_true, y_pred = _check(y_true, y_pred)
        for correct in y_true == y_pred:
            self._buf[self._i] = correct
            self._i = (self._i + 1) % self.window
            self._filled = min(self._filled + 1, self.window)
        return self

    def result(self):
        if self._filled == 0:
            return float("nan")
        return float(self._buf[: self._filled].mean())


def accuracy_score(y_true, y_pred) -> float:
    """Stateless convenience accuracy."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else float("nan")