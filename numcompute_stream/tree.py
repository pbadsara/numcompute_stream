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


class HoeffdingTreeClassifier:
    """Depth-limited streaming decision tree.
    """

    def __init__(self, max_depth=8, min_samples_split=60, max_features=None,
                 criterion="gini", delta=0.05, tau=0.05, n_candidates=10,
                 random_state=None):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.delta = delta
        self.tau = tau
        self.n_candidates = n_candidates
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.classes_ = None
        self._cls_index: dict = {}
        self.n_features_ = None
        self.root: _Leaf | _Node | None = None

    # -- class bookkeeping ---------------------------------------------------
    def _register_classes(self, y):
        for c in np.unique(y):
            c = c.item()
            if c not in self._cls_index:
                self._cls_index[c] = len(self._cls_index)
        self.classes_ = np.array(
            [k for k, _ in sorted(self._cls_index.items(), key=lambda kv: kv[1])]
        )

    def _encode(self, y):
        return np.array([self._cls_index[v.item()] for v in np.asarray(y)])

    def _n_classes(self):
        return len(self._cls_index)

    def _max_feats(self):
        d = self.n_features_
        if self.max_features is None:
            return d
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        return min(int(self.max_features), d)

    # -- routing -------------------------------------------------------------
    def _collect(self, X):
        """Partition row indices to their leaves (vectorised descent)."""
        out = []

        def rec(node, idx):
            if isinstance(node, _Leaf):
                out.append((node, idx))
                return
            xj = X[idx, node.feature]
            left = ~(xj > node.threshold)  # NaN -> left
            rec(node.left, idx[left])
            rec(node.right, idx[~left])

        rec(self.root, np.arange(X.shape[0]))
        return out

    # -- splitting -----------------------------------------------------------
    def _best_split(self, leaf: _Leaf):
        """Return (gain1, gain2, feature, threshold, left_counts, right_counts)."""
        counts = leaf.n_class
        K = counts.size
        N = counts.sum()
        parent_imp = _impurity(counts, self.criterion)
        feats = self._rng.choice(self.n_features_, size=self._max_feats(),
                                 replace=False)
        best = (-np.inf, None, None, None, None)
        second = -np.inf
        for j in feats:
            lo, hi = leaf.fmin[j], leaf.fmax[j]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            thr = np.linspace(lo, hi, self.n_candidates + 2)[1:-1]  # (T,)
            mean = leaf.stat_mean[:, j]  # (K,)
            var = np.divide(leaf.stat_M2[:, j], np.maximum(leaf.stat_n[:, j] - 1, 0),
                            out=np.zeros(K), where=leaf.stat_n[:, j] > 1)
            std = np.sqrt(np.maximum(var, 1e-12))
            # P(x <= t | class) for every threshold/class -> (T, K)
            z = (thr[:, None] - mean[None, :]) / std[None, :]
            cdf = _norm_cdf(z)
            left = cdf * counts[None, :]  # (T, K)
            right = counts[None, :] - left
            nL = left.sum(1)
            nR = right.sum(1)
            valid = (nL > 0) & (nR > 0)
            if not valid.any():
                continue
            impL = _impurity(left, self.criterion)
            impR = _impurity(right, self.criterion)
            gain = parent_imp - (nL / N * impL + nR / N * impR)
            gain = np.where(valid, gain, -np.inf)
            t_idx = int(np.argmax(gain))
            g = gain[t_idx]
            if g > best[0]:
                second = best[0] if best[0] > -np.inf else second
                best = (g, j, float(thr[t_idx]), left[t_idx], right[t_idx])
            elif g > second:
                second = g
        return best, second

    def _attempt_split(self, leaf: _Leaf):
        if leaf.depth >= self.max_depth:
            return None
        if leaf.total() < self.min_samples_split:
            return None
        leaf.seen_since_eval = 0.0
        (g1, j, thr, left_counts, right_counts), g2 = self._best_split(leaf)
        if j is None or g1 <= 0:
            return None
        n = leaf.total()
        R = 1.0 if self.criterion == "gini" else np.log(max(self._n_classes(), 2))
        eps = np.sqrt(R * R * np.log(1.0 / self.delta) / (2.0 * n))
        if (g1 - g2) > eps or eps < self.tau:
            K, d = self._n_classes(), self.n_features_
            lc = np.zeros(K)
            rc = np.zeros(K)
            lc[: left_counts.size] = np.round(left_counts)
            rc[: right_counts.size] = np.round(right_counts)
            return _Node(
                feature=j,
                threshold=thr,
                left=_Leaf(leaf.depth + 1, K, d, lc),
                right=_Leaf(leaf.depth + 1, K, d, rc),
            )
        return None

    def _grow(self):
        """Walk the tree and replace any splittable leaf with a decision node."""
        def rec(node):
            if isinstance(node, _Leaf):
                return self._attempt_split(node) or node
            node.left = rec(node.left)
            node.right = rec(node.right)
            return node

        self.root = rec(self.root)

    # -- public API ----------------------------------------------------------
    def partial_fit(self, X, y, classes=None) -> "HoeffdingTreeClassifier":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} labels."
            )
        if classes is not None:
            self._register_classes(np.asarray(classes))
        self._register_classes(y)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, got {X.shape[1]}."
            )
        # grow class-count arrays if new classes appeared since the last chunk
        if self.root is None:
            self.root = _Leaf(0, self._n_classes(), self.n_features_)
        self._resize_leaves(self.root)

        yc = self._encode(y)
        for leaf, idx in self._collect(X):
            if idx.size:
                leaf.update(X[idx], yc[idx])
        self._grow()
        return self

    def _resize_leaves(self, node):
        """Pad per-leaf class arrays after new classes are discovered."""
        K = self._n_classes()

        def rec(n):
            if isinstance(n, _Leaf):
                if n.n_class.size < K:
                    pad = K - n.n_class.size
                    n.n_class = np.concatenate([n.n_class, np.zeros(pad)])
                    n.stat_n = np.vstack([n.stat_n, np.zeros((pad, self.n_features_))])
                    n.stat_mean = np.vstack([n.stat_mean,
                                             np.zeros((pad, self.n_features_))])
                    n.stat_M2 = np.vstack([n.stat_M2,
                                           np.zeros((pad, self.n_features_))])
            else:
                rec(n.left)
                rec(n.right)

        rec(node)

    def fit(self, X, y, classes=None) -> "HoeffdingTreeClassifier":
        self.__init__(self.max_depth, self.min_samples_split, self.max_features,
                      self.criterion, self.delta, self.tau, self.n_candidates,
                      self.random_state)
        return self.partial_fit(X, y, classes)

    def predict_proba(self, X) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("tree is not fitted.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K = self._n_classes()
        proba = np.zeros((X.shape[0], K))
        for leaf, idx in self._collect(X):
            if not idx.size:
                continue
            counts = leaf.n_class[:K]
            tot = counts.sum()
            row = counts / tot if tot > 0 else np.full(K, 1.0 / K)
            proba[idx] = row
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    def n_leaves(self) -> int:
        def rec(n):
            return 1 if isinstance(n, _Leaf) else rec(n.left) + rec(n.right)

        return rec(self.root) if self.root is not None else 0

    def depth(self) -> int:
        def rec(n, d):
            return d if isinstance(n, _Leaf) else max(rec(n.left, d + 1),
                                                      rec(n.right, d + 1))

        return rec(self.root, 0) if self.root is not None else 0


# Public alias matching the specification's class name.
DecisionTreeClassifier = HoeffdingTreeClassifier