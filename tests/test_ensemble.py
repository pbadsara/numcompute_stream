"""Tests for streaming ensembles (bagging / random forest / boosting)."""
import numpy as np
import pytest

from numcompute_stream import EnsembleClassifier, RandomForestClassifier


def _make_three_blobs(n=1800, seed=0):
    rng = np.random.default_rng(seed)
    per = n // 3
    centers = [[-4, 0], [4, 0], [0, 5]]
    Xs, ys = [], []
    for k, c in enumerate(centers):
        Xs.append(rng.normal(loc=c, scale=0.9, size=(per, 2)))
        ys.append(np.full(per, k))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


@pytest.mark.parametrize("method", ["bagging", "random_forest", "boosting"])
def test_ensemble_methods_learn(method):
    X, y = _make_three_blobs()
    clf = EnsembleClassifier(n_estimators=8, method=method, random_state=0)
    classes = np.unique(y)
    for i in range(0, len(X), 150):
        clf.partial_fit(X[i:i + 150], y[i:i + 150], classes=classes)
    assert clf.score(X, y) > 0.9


def test_random_forest_subclass_defaults():
    rf = RandomForestClassifier(n_estimators=6, random_state=0)
    assert rf.method == "random_forest"
    assert rf.max_features == "sqrt"


def test_predict_proba_shape_and_normalised():
    X, y = _make_three_blobs()
    rf = RandomForestClassifier(n_estimators=6, random_state=0).fit(X, y)
    proba = rf.predict_proba(X[:40])
    assert proba.shape == (40, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_hard_vs_soft_voting_both_work():
    X, y = _make_three_blobs()
    classes = np.unique(y)
    for voting in ("soft", "hard"):
        clf = EnsembleClassifier(n_estimators=6, method="random_forest",
                                 voting=voting, random_state=1)
        clf.partial_fit(X, y, classes=classes)
        assert clf.score(X, y) > 0.85


def test_forest_beats_single_member_on_noisy_data():
    # Forest of many shallow trees should outperform one shallow tree.
    X, y = _make_three_blobs(n=2400, seed=3)
    single = RandomForestClassifier(n_estimators=1, random_state=0,
                                    tree_params={"max_depth": 3}).fit(X, y)
    forest = RandomForestClassifier(n_estimators=15, random_state=0,
                                    tree_params={"max_depth": 3}).fit(X, y)
    assert forest.score(X, y) >= single.score(X, y)