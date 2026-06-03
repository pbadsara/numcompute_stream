"""ensemble.py -- Streaming tree ensembles.
"""
from __future__ import annotations

import numpy as np

from .tree import HoeffdingTreeClassifier


class EnsembleClassifier:
    """N decision trees combined by bagging, random forest, or boosting.
    """

    def __init__(self, n_estimators=10, method="random_forest", max_features=None,
                 voting="soft", tree_params=None, random_state=None):
        if method not in {"bagging", "random_forest", "boosting"}:
            raise ValueError("method must be bagging, random_forest or boosting.")
        if voting not in {"soft", "hard"}:
            raise ValueError("voting must be 'soft' or 'hard'.")
        self.n_estimators = n_estimators
        self.method = method
        self.voting = voting
        self.tree_params = dict(tree_params or {})
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        if max_features is None:
            max_features = "sqrt" if method == "random_forest" else None
        self.max_features = max_features
        seeds = self._rng.integers(0, 2**31 - 1, size=n_estimators)
        self.estimators_ = [
            HoeffdingTreeClassifier(
                max_features=max_features, random_state=int(s), **self.tree_params
            )
            for s in seeds
        ]
        # boosting state
        self._lam_sc = np.zeros(n_estimators)  # sum of correctly-weighted lambda
        self._lam_sw = np.zeros(n_estimators)  # sum of incorrectly-weighted lambda
        self.classes_ = None
        self._cls_index: dict = {}

    def _register(self, y, classes):
        if classes is not None:
            for c in np.unique(classes):
                self._cls_index.setdefault(c.item(), len(self._cls_index))
        for c in np.unique(y):
            self._cls_index.setdefault(c.item(), len(self._cls_index))
        self.classes_ = np.array(
            [k for k, _ in sorted(self._cls_index.items(), key=lambda kv: kv[1])]
        )

    # -- training ------------------------------------------------------------
    def partial_fit(self, X, y, classes=None) -> "EnsembleClassifier":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different numbers of rows.")
        self._register(y, classes)
        all_classes = self.classes_
        if self.method == "boosting":
            return self._partial_fit_boosting(X, y, all_classes)
        # bagging / random forest
        for tree in self.estimators_:
            k = self._rng.poisson(1.0, size=X.shape[0])
            if k.sum() == 0:
                tree.partial_fit(X[:0], y[:0], classes=all_classes)
                continue
            Xr = np.repeat(X, k, axis=0)
            yr = np.repeat(y, k, axis=0)
            tree.partial_fit(Xr, yr, classes=all_classes)
        return self

    def _partial_fit_boosting(self, X, y, all_classes):
        for i in range(X.shape[0]):
            xi = X[i : i + 1]
            yi = y[i : i + 1]
            lam = 1.0
            for m, tree in enumerate(self.estimators_):
                k = self._rng.poisson(lam)
                if k > 0:
                    tree.partial_fit(np.repeat(xi, k, axis=0),
                                     np.repeat(yi, k, axis=0), classes=all_classes)
                pred = tree.predict(xi)[0] if tree.root is not None else None
                if pred == yi[0]:
                    self._lam_sc[m] += lam
                else:
                    self._lam_sw[m] += lam
                tot = self._lam_sc[m] + self._lam_sw[m]
                err = self._lam_sw[m] / tot if tot > 0 else 0.5
                err = min(max(err, 1e-6), 1 - 1e-6)
                lam *= (1.0 / (2 * err)) if pred != yi[0] else (1.0 / (2 * (1 - err)))
        return self

    def fit(self, X, y, classes=None) -> "EnsembleClassifier":
        return self.partial_fit(X, y, classes)

    # -- inference -----------------------------------------------------------
    def _aligned_proba(self, tree, X):
        """Map a tree's class-local probabilities onto the ensemble class order."""
        K = self.classes_.size
        out = np.zeros((X.shape[0], K))
        if tree.root is None or tree.classes_ is None:
            return out + 1.0 / K
        p = tree.predict_proba(X)
        idx = {c: i for i, c in enumerate(self.classes_)}
        for j, c in enumerate(tree.classes_):
            out[:, idx[c.item()]] = p[:, j]
        return out

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.method == "boosting":
            tot = self._lam_sc + self._lam_sw
            err = np.divide(self._lam_sw, tot, out=np.full_like(tot, 0.5),
                            where=tot > 0)
            err = np.clip(err, 1e-6, 1 - 1e-6)
            w = np.log((1 - err) / err)
            w = np.where(tot > 0, w, 0.0)
        else:
            w = np.ones(self.n_estimators)
        if w.sum() <= 0:
            w = np.ones(self.n_estimators)
        acc = np.zeros((X.shape[0], self.classes_.size))
        for m, tree in enumerate(self.estimators_):
            acc += w[m] * self._aligned_proba(tree, X)
        return acc / w.sum()

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.voting == "hard":
            votes = np.zeros((X.shape[0], self.classes_.size))
            idx = {c: i for i, c in enumerate(self.classes_)}
            for tree in self.estimators_:
                if tree.root is None:
                    continue
                pred = tree.predict(X)
                for r, c in enumerate(pred):
                    votes[r, idx[c.item()]] += 1
            return self.classes_[np.argmax(votes, axis=1)]
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))


# Convenience subclasses -------------------------------------------------------
class RandomForestClassifier(EnsembleClassifier):
    def __init__(self, n_estimators=10, max_features="sqrt", tree_params=None,
                 random_state=None, voting="soft"):
        super().__init__(n_estimators=n_estimators, method="random_forest",
                         max_features=max_features, voting=voting,
                         tree_params=tree_params, random_state=random_state)