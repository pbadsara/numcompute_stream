"""Tests for the streaming Pipeline and StreamTrainer harness."""
import numpy as np
import pytest

from numcompute_stream import (
    Pipeline, Imputer, StandardScaler, DecisionTreeClassifier,
    RandomForestClassifier, StreamTrainer, model_size_bytes, iter_chunks,
)


def _data(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X = np.vstack([rng.normal(-2, 1, (n0, 3)), rng.normal(2, 1, (n - n0, 3))])
    y = np.concatenate([np.zeros(n0), np.ones(n - n0)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def test_pipeline_partial_fit_and_predict():
    X, y = _data()
    pipe = Pipeline([("sc", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=5,
                                                      min_samples_split=40))])
    classes = np.unique(y)
    for Xc, yc in iter_chunks(X, y, chunk_size=200):
        pipe.partial_fit(Xc, yc, classes=classes)
    assert pipe.score(X, y) > 0.95


def test_pipeline_named_steps_and_classes():
    X, y = _data()
    pipe = Pipeline([("imp", Imputer("mean")), ("sc", StandardScaler()),
                     ("model", RandomForestClassifier(n_estimators=5,
                                                      random_state=0))])
    pipe.fit(X, y)
    assert "model" in pipe.named_steps
    # classes_ must delegate to the final estimator
    assert set(pipe.classes_) == set(np.unique(y))


def test_pipeline_transform_chain_imputes_nan():
    X, y = _data(n=400)
    X[::5, 0] = np.nan
    pipe = Pipeline([("imp", Imputer("mean")), ("sc", StandardScaler()),
                     ("model", DecisionTreeClassifier())])
    pipe.fit(X, y)
    Xt = pipe.transform(X)
    assert not np.isnan(Xt).any()


def test_stream_trainer_prequential_logs():
    X, y = _data()
    pipe = Pipeline([("sc", StandardScaler()),
                     ("model", RandomForestClassifier(n_estimators=6,
                                                      random_state=0))])
    tr = StreamTrainer(pipe, prequential=True, rolling_window=100)
    tr.run(iter_chunks(X, y, chunk_size=200), classes=np.unique(y))
    s = tr.summary()
    assert s["n_chunks"] == 8
    assert s["n_samples"] == 1600
    assert 0.0 <= s["final_cumulative_accuracy"] <= 1.0
    # every history list should have one entry per chunk
    assert all(len(v) == 8 for v in tr.history.values())


def test_stream_trainer_first_chunk_accuracy_is_nan():
    # Prequential: first chunk is scored before any training -> nan accuracy.
    X, y = _data()
    pipe = Pipeline([("sc", StandardScaler()),
                     ("model", DecisionTreeClassifier())])
    tr = StreamTrainer(pipe, prequential=True)
    tr.run(iter_chunks(X, y, chunk_size=200), classes=np.unique(y))
    assert np.isnan(tr.history["chunk_accuracy"][0])
    assert not np.isnan(tr.history["chunk_accuracy"][-1])


def test_model_size_bytes_positive():
    clf = DecisionTreeClassifier().fit(*_data(n=200))
    assert model_size_bytes(clf) > 0


def test_stream_trainer_end_to_end_summary_valid():
    # Full pipeline + StreamTrainer.run() integration: summary() must return
    # sensible values after a complete multi-chunk run.
    X, y = _data(n=800)
    pipe = Pipeline([
        ("imp", Imputer("mean")),
        ("sc", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=5, random_state=1)),
    ])
    tr = StreamTrainer(pipe, prequential=True, rolling_window=100)
    tr.run(iter_chunks(X, y, chunk_size=100), classes=np.unique(y))
    s = tr.summary()
    assert s["n_chunks"] == 8
    assert s["n_samples"] == 800
    assert 0.0 <= s["final_cumulative_accuracy"] <= 1.0
    assert 0.0 <= s["final_rolling_accuracy"] <= 1.0
    assert s["total_fit_time_s"] > 0.0
    assert s["final_memory_bytes"] > 0


def test_stream_trainer_summary_empty_stream():
    # summary() on a trainer that has never processed a chunk must not raise
    # and must return NaN / -1 sentinels rather than index errors.
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    tr = StreamTrainer(pipe)
    s = tr.summary()
    assert s["n_chunks"] == 0
    assert s["n_samples"] == 0
    assert np.isnan(s["final_cumulative_accuracy"])
    assert np.isnan(s["final_rolling_accuracy"])
    assert s["final_memory_bytes"] == -1