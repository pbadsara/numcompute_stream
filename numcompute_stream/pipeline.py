"""pipeline.py -- Streaming pipeline.

`Pipeline` chains transformers and a final estimator.
``partial_fit`` updates each transformer on the chunk, transforms forward, then
updates the model -- so the whole chain adapts incrementally on every chunk.
"""
from __future__ import annotations

import numpy as np


class Pipeline:
    """Sequentially chained streaming estimators.
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [n for n, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique.")
        self.steps = steps

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def classes_(self):
        """Delegate to the final estimator (``None`` until first ``partial_fit``)."""
        return getattr(self._final, "classes_", None)

    @property
    def _transformers(self):
        return self.steps[:-1]

    @property
    def _final(self):
        return self.steps[-1][1]

    def _forward(self, X, update: bool, y=None):
        Xt = np.atleast_2d(np.asarray(X, dtype=float))
        for _, tf in self._transformers:
            if update:
                tf.partial_fit(Xt, y)
            Xt = tf.transform(Xt)
        return Xt

    def partial_fit(self, X, y, classes=None) -> "Pipeline":
        Xt = self._forward(X, update=True, y=y)
        final = self._final
        try:
            final.partial_fit(Xt, y, classes=classes)
        except TypeError:
            final.partial_fit(Xt, y)
        return self

    def fit(self, X, y, classes=None) -> "Pipeline":
        return self.partial_fit(X, y, classes)

    def transform(self, X) -> np.ndarray:
        return self._forward(X, update=False)

    def predict(self, X) -> np.ndarray:
        return self._final.predict(self._forward(X, update=False))

    def predict_proba(self, X) -> np.ndarray:
        return self._final.predict_proba(self._forward(X, update=False))

    def score(self, X, y) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))