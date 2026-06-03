"""Tests for the NumPy/stdlib-only CSV IO layer."""
import numpy as np
import pytest

from numcompute_stream import load_csv, save_csv, iter_chunks, stream_csv


def test_save_then_load_roundtrip(tmp_path):
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    path = tmp_path / "d.csv"
    save_csv(str(path), X, y, feature_names=["a", "b"])
    X2, y2, meta = load_csv(str(path), target=-1, has_header=True,
                            categorical_target=False)
    assert np.allclose(X2, X)
    assert np.allclose(y2.astype(float), y)
    assert meta["feature_names"] == ["a", "b"]


def test_load_csv_missing_values_become_nan(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("a,b,label\n1,2,0\n,5,1\n7,,0\n")
    X, y, meta = load_csv(str(path), target=-1, has_header=True,
                          categorical_target=True)
    assert np.isnan(X[1, 0])
    assert np.isnan(X[2, 1])
    assert X.shape == (3, 2)


def test_categorical_target_encoded_to_ints(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("x,y\n1.0,cat\n2.0,dog\n3.0,cat\n")
    X, y, meta = load_csv(str(path), target=-1, has_header=True,
                          categorical_target=True)
    assert set(np.unique(y)) == {0, 1}
    assert "cat" in meta["classes"] and "dog" in meta["classes"]


def test_iter_chunks_partitions_all_rows():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)
    chunks = list(iter_chunks(X, y, chunk_size=3))
    assert sum(c[0].shape[0] for c in chunks) == 10
    assert chunks[0][0].shape == (3, 2)
    assert chunks[-1][0].shape[0] == 1  # remainder


def test_iter_chunks_shuffle_preserves_pairs():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0].copy()  # label == first feature for traceability
    chunks = list(iter_chunks(X, y, chunk_size=4, shuffle=True, random_state=0))
    for Xc, yc in chunks:
        assert np.allclose(Xc[:, 0], yc)


def test_stream_csv_yields_chunks(tmp_path):
    X = np.random.default_rng(0).normal(size=(25, 3))
    y = np.random.default_rng(1).integers(0, 2, size=25)
    path = tmp_path / "s.csv"
    save_csv(str(path), X, y, feature_names=["f0", "f1", "f2"])
    total = sum(Xc.shape[0] for Xc, _ in stream_csv(str(path), chunk_size=10,
                                                    categorical_target=True))
    assert total == 25