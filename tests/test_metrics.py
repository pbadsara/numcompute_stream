"""Tests for streaming classification metrics."""
import numpy as np
import pytest

from numcompute_stream import (
    Accuracy, ConfusionMatrix, Precision, Recall, F1,
    BinaryAUC, RollingAccuracy, accuracy_score,
)


def test_accuracy_streaming_matches_batch():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 3, size=500)
    yp = yt.copy()
    yp[::7] = (yp[::7] + 1) % 3  # corrupt ~1/7
    acc = Accuracy()
    for i in range(0, 500, 50):
        acc.update(yt[i:i + 50], yp[i:i + 50])
    assert acc.result() == pytest.approx(np.mean(yt == yp))


def test_accuracy_reset():
    acc = Accuracy()
    acc.update([1, 1, 0], [1, 0, 0])
    acc.reset()
    assert np.isnan(acc.result())


def test_confusion_matrix_grows_labels():
    cm = ConfusionMatrix()
    cm.update([0, 1], [0, 1])
    cm.update([2, 2], [2, 0])  # new label 2 appears
    mat = cm.result()
    labels = cm.labels
    assert set(labels) == {0, 1, 2}
    assert mat.sum() == 4
    # diagonal counts correct predictions
    idx = {l: i for i, l in enumerate(labels)}
    assert mat[idx[0], idx[0]] == 1


def test_precision_recall_f1_perfect():
    p, r, f = Precision(), Recall(), F1()
    for m in (p, r, f):
        m.update([0, 1, 2, 1], [0, 1, 2, 1])
    assert p.result() == pytest.approx(1.0)
    assert r.result() == pytest.approx(1.0)
    assert f.result() == pytest.approx(1.0)


def test_precision_macro_vs_micro():
    yt = [0, 0, 1, 1, 1]
    yp = [0, 1, 1, 1, 0]
    macro = Precision(average="macro")
    micro = Precision(average="micro")
    macro.update(yt, yp)
    micro.update(yt, yp)
    # micro precision == accuracy for single-label multiclass
    assert micro.result() == pytest.approx(accuracy_score(yt, yp))
    assert 0.0 <= macro.result() <= 1.0


def test_binary_auc_separable():
    auc = BinaryAUC(positive_label=1)
    yt = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc.update(yt, scores)
    assert auc.result() == pytest.approx(1.0, abs=0.02)


def test_binary_auc_random_is_half():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 2, size=4000)
    scores = rng.random(4000)
    auc = BinaryAUC(positive_label=1)
    auc.update(yt, scores)
    assert auc.result() == pytest.approx(0.5, abs=0.05)


def test_rolling_accuracy_window():
    roll = RollingAccuracy(window=10)
    # first 20 all correct, then feed 10 wrong -> window should be ~0
    roll.update(np.ones(20), np.ones(20))
    assert roll.result() == pytest.approx(1.0)
    roll.update(np.ones(10), np.zeros(10))
    assert roll.result() == pytest.approx(0.0)


def test_accuracy_score_stateless():
    assert accuracy_score([1, 2, 3], [1, 2, 0]) == pytest.approx(2 / 3)