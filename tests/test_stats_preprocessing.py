"""Tests for streaming statistics and preprocessing transformers."""
import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats, StreamingHistogram, P2Quantile
from numcompute_stream.preprocessing import StandardScaler, Imputer, OneHotEncoder


def test_streaming_stats_matches_numpy():
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 3.0, size=(1000, 4))
    s = StreamingStats()
    for i in range(0, 1000, 137):  # uneven chunks
        s.update_stats(X[i : i + 137])
    assert np.allclose(s.mean_, X.mean(0), atol=1e-9)
    assert np.allclose(s.variance, X.var(0, ddof=1), atol=1e-7)


def test_streaming_stats_ignores_nan():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    s = StreamingStats().update_stats(X)
    assert s.mean_[0] == pytest.approx(3.0)
    assert s.mean_[1] == pytest.approx(3.0)  # mean of [2, 4]
    assert s.count[1] == 2


def test_streaming_stats_zero_variance():
    s = StreamingStats().update_stats(np.full((50, 2), 7.0))
    assert np.allclose(s.variance, 0.0)


def test_streaming_stats_feature_mismatch_raises():
    s = StreamingStats().update_stats(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        s.update_stats(np.zeros((3, 3)))


def test_histogram_quantile_close_to_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=5000)
    h = StreamingHistogram(bins=128, value_range=(-5, 5))
    for i in range(0, 5000, 500):
        h.update(x[i : i + 500])
    assert h.quantile(0.5) == pytest.approx(np.median(x), abs=0.1)
    assert h.quantile(0.9) == pytest.approx(np.quantile(x, 0.9), abs=0.15)


def test_histogram_sliding_window():
    h = StreamingHistogram(bins=10, value_range=(0, 1), window=1)
    h.update(np.zeros(100))
    h.update(np.ones(100) * 0.95)  # only the last chunk should remain
    assert h.counts_.sum() == 100
    assert h.quantile(0.5) > 0.5


def test_p2_quantile_median():
    rng = np.random.default_rng(2)
    x = rng.normal(10, 2, size=10000)
    p = P2Quantile(0.5).update(x)
    assert p.result() == pytest.approx(np.median(x), abs=0.1)


def test_p2_quantile_invalid_p():
    with pytest.raises(ValueError):
        P2Quantile(1.5)


def test_standard_scaler_streaming_equivalence():
    rng = np.random.default_rng(3)
    X = rng.normal(5, 2, size=(600, 3))
    sc = StandardScaler()
    for i in range(0, 600, 100):
        sc.partial_fit(X[i : i + 100])
    Xt = sc.transform(X)
    assert np.allclose(Xt.mean(0), 0.0, atol=1e-6)
    assert np.allclose(Xt.std(0), 1.0, atol=1e-2)


def test_standard_scaler_zero_variance_no_nan():
    X = np.hstack([np.full((20, 1), 3.0), np.arange(20).reshape(-1, 1).astype(float)])
    sc = StandardScaler().fit(X)
    Xt = sc.transform(X)
    assert not np.isnan(Xt).any()  # constant column must not divide by zero


def test_standard_scaler_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.zeros((2, 2)))


def test_imputer_mean_fills_nan():
    X = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0]])
    imp = Imputer("mean").fit(X)
    out = imp.transform(X)
    assert not np.isnan(out).any()
    assert out[2, 0] == pytest.approx(2.0)  # mean of column 0 = (1+3)/2


def test_imputer_most_frequent():
    X = np.array([[1.0], [1.0], [2.0], [np.nan]])
    imp = Imputer("most_frequent").fit(X)
    assert imp.transform(X)[3, 0] == pytest.approx(1.0)


def test_onehot_grows_categories():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([["a"], ["b"]]))
    enc.partial_fit(np.array([["c"]]))  # new category appears later in stream
    assert enc.n_output_features_ == 3
    out = enc.transform(np.array([["c"]]))
    assert out.shape == (1, 3) and out.sum() == 1.0


def test_onehot_unknown_ignored():
    enc = OneHotEncoder(handle_unknown="ignore").fit(np.array([["x"], ["y"]]))
    out = enc.transform(np.array([["z"]]))  # unseen -> all zeros
    assert out.sum() == 0.0


def test_imputer_all_nan_column_does_not_crash():
    # A column that is entirely NaN across the fitting chunk should produce a
    # fill value of 0.0 (safe fallback) rather than raising or returning NaN
    # in the output — the scaler downstream must not blow up.
    X = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    imp = Imputer("mean").fit(X)
    # statistics_ for the all-NaN column should be 0.0 (Welford fallback)
    assert imp.statistics_[0] == pytest.approx(0.0)
    out = imp.transform(X)
    # no NaNs should survive into the transformed output
    assert not np.isnan(out).any()


def test_onehot_unknown_error_mode_raises():
    enc = OneHotEncoder(handle_unknown="error").fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="unknown category"):
        enc.transform(np.array([["c"]]))  