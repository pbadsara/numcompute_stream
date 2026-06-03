"""io.py -- Custom CSV input/output pipeline.

Only the Python standard-library ``csv`` module and NumPy are used; no pandas.
The loader returns a numeric feature matrix ``X`` (missing values encoded as
``np.nan``) and an integer-encoded label vector ``y`` 
"""
from __future__ import annotations

import csv
from typing import Iterator

import numpy as np

_DEFAULT_MISSING = ("", "na", "n/a", "nan", "null", "none", "?")


def _to_float(col: np.ndarray, missing: tuple[str, ...]) -> np.ndarray:
    """Convert an object column to ``float64`` with ``np.nan`` for missing cells.
    """
    out = np.empty(col.shape[0], dtype=float)
    for i, raw in enumerate(col):
        s = str(raw).strip()
        if s.lower() in missing:
            out[i] = np.nan
        else:
            try:
                out[i] = float(s)
            except ValueError:
                out[i] = np.nan
    return out


def _encode_labels(col: np.ndarray) -> tuple[np.ndarray, list]:
    """Integer-encode a categorical label column.
    """
    classes = sorted({str(v).strip() for v in col})
    lookup = {c: i for i, c in enumerate(classes)}
    y = np.array([lookup[str(v).strip()] for v in col], dtype=np.int64)
    return y, classes


def load_csv(
    path: str,
    target: int | str = -1,
    has_header: bool = True,
    missing_values: tuple[str, ...] = _DEFAULT_MISSING,
    categorical_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a CSV file into ``(X, y, meta)``.
    """
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        raise ValueError(f"{path!r} contains no data rows.")

    if has_header:
        header, data = rows[0], rows[1:]
    else:
        header = [f"f{i}" for i in range(len(rows[0]))]
        data = rows
    if not data:
        raise ValueError(f"{path!r} has a header but no samples.")

    ncols = len(header)
    tidx = header.index(target) if isinstance(target, str) else target % ncols
    feat_idx = [i for i in range(ncols) if i != tidx]

    raw = np.array(data, dtype=object)
    missing = tuple(m.lower() for m in missing_values)
    X = (
        np.column_stack([_to_float(raw[:, j], missing) for j in feat_idx])
        if feat_idx
        else np.empty((len(data), 0))
    )

    ycol = raw[:, tidx]
    if categorical_target:
        y, classes = _encode_labels(ycol)
    else:
        y, classes = _to_float(ycol, missing), None

    meta = {
        "feature_names": [header[i] for i in feat_idx],
        "target_name": header[tidx],
        "classes": classes,
    }
    return X, y, meta


def iter_chunks(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    shuffle: bool = False,
    random_state: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(X_chunk, y_chunk)`` slices to simulate a data stream.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    n = X.shape[0]
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(random_state).shuffle(order)
    for start in range(0, n, chunk_size):
        idx = order[start : start + chunk_size]
        yield X[idx], y[idx]


def stream_csv(
    path: str, chunk_size: int, **load_kwargs
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Convenience generator: ``load_csv`` then ``iter_chunks``."""
    X, y, _ = load_csv(path, **load_kwargs)
    yield from iter_chunks(X, y, chunk_size)


def save_csv(path: str, X: np.ndarray, y: np.ndarray, feature_names=None) -> None:
    """Persist a feature matrix and labels to CSV (utility for tests/demos)."""
    X = np.atleast_2d(X)
    d = X.shape[1]
    names = feature_names or [f"f{i}" for i in range(d)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(names) + ["target"])
        for row, label in zip(X, y):
            w.writerow([*row.tolist(), label])