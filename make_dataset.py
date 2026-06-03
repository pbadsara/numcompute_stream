"""make_dataset.py -- generating a synthetic classification CSV for the demo.

Creates a 3-class, multi-feature dataset with:
  * informative + redundant + noise features,
  * some missing values
"""
from __future__ import annotations

import argparse

import numpy as np

from numcompute_stream.io import save_csv


def make_classification(n=6000, n_informative=4, n_noise=2, n_classes=3,
                        missing_rate=0.02, drift=True, random_state=0):
    rng = np.random.default_rng(random_state)
    centres = rng.normal(0, 3.5, size=(n_classes, n_informative))
    y = rng.integers(0, n_classes, size=n)
    X_inf = centres[y] + rng.normal(0, 1.0, size=(n, n_informative))

    if drift:  # shift class centres for the second half of the stream
        half = n // 2
        shift = rng.normal(0, 1.5, size=(n_classes, n_informative))
        X_inf[half:] += shift[y[half:]]

    X_noise = rng.normal(0, 1.0, size=(n, n_noise))
    X_red = X_inf[:, :1] * 0.8 + rng.normal(0, 0.3, size=(n, 1))  # redundant feat
    X = np.hstack([X_inf, X_red, X_noise])

    if missing_rate > 0:
        mask = rng.random(X.shape) < missing_rate
        X[mask] = np.nan
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data/stream_data.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X, y = make_classification(n=args.n, random_state=args.seed)
    names = [f"feat_{i}" for i in range(X.shape[1])]
    save_csv(args.out, X, y, feature_names=names)
    print(f"wrote {args.out}: X={X.shape}, classes={sorted(set(y.tolist()))}")


if __name__ == "__main__":
    main()