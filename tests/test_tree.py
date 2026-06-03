"""Tests for the streaming Hoeffding decision tree."""
import numpy as np
import pytest

from numcompute_stream import DecisionTreeClassifier, HoeffdingTreeClassifier


def _make_separable(n=1500, seed=0):
    """Two well-separated Gaussian blobs in 2-D."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X0 = rng.normal(loc=[-3, -3], scale=0.7, size=(n0, 2))
    X1 = rng.normal(loc=[3, 3], scale=0.7, size=(n - n0, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n - n0)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def test_alias_identity():
    assert DecisionTreeClassifier is HoeffdingTreeClassifier


def test_partial_fit_learns_separable():
    X, y = _make_separable()
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=40, delta=0.05,
                                 random_state=0)
    for i in range(0, len(X), 100):
        clf.partial_fit(X[i:i + 100], y[i:i + 100], classes=[0.0, 1.0])
    assert clf.score(X, y) > 0.95


def test_tree_actually_splits():
    X, y = _make_separable()
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=40, delta=0.05)
    clf.fit(X, y)
    assert clf.n_leaves() >= 2
    assert clf.depth() >= 1


def test_predict_proba_rows_sum_to_one():
    X, y = _make_separable()
    clf = DecisionTreeClassifier().fit(X, y)
    proba = clf.predict_proba(X[:50])
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_before_fit_raises():
    clf = DecisionTreeClassifier()
    with pytest.raises(Exception):
        clf.predict(np.zeros((2, 2)))


def test_nan_routes_without_error():
    X, y = _make_separable(n=600)
    X[::10, 0] = np.nan  # inject missing values
    clf = DecisionTreeClassifier(max_depth=4, min_samples_split=40)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (len(X),)


def test_max_depth_respected():
    X, y = _make_separable(n=3000)
    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=30, delta=0.05)
    clf.fit(X, y)
    assert clf.depth() <= 2


def test_entropy_criterion_runs():
    X, y = _make_separable()
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                 min_samples_split=40)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.9