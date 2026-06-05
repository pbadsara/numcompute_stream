"""benchmark.py -- Streaming benchmark: single tree vs. ensembles.

Running a prequential evaluation that compares a single Hoeffding tree against an online Random Forest and
an online boosting ensemble on three axes: predictive accuracy, training time,
and model memory footprint.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from numcompute_stream import (
    load_csv, iter_chunks, Pipeline, Imputer, StandardScaler,
    DecisionTreeClassifier, RandomForestClassifier, EnsembleClassifier,
    StreamTrainer,
)


def build_models(random_state: int = 0) -> dict:
    """Return the set of competitors keyed by display name."""
    tree_params = {"max_depth": 10, "min_samples_split": 60, "delta": 0.05}
    return {
        "SingleTree": lambda: DecisionTreeClassifier(**tree_params),
        "RandomForest(12)": lambda: RandomForestClassifier(
            n_estimators=12, max_features="sqrt",
            tree_params=tree_params, random_state=random_state),
        "Boosting(12)": lambda: EnsembleClassifier(
            n_estimators=12, method="boosting",
            tree_params=tree_params, random_state=random_state),
    }


def run_one(make_model, chunks, classes):
    """Run prequential evaluation for a single model; return (summary, history)."""
    pipe = Pipeline([
        ("impute", Imputer("mean")),
        ("scale", StandardScaler()),
        ("model", make_model()),
    ])
    trainer = StreamTrainer(pipe, prequential=True, rolling_window=200)
    t0 = time.perf_counter()
    trainer.run(chunks, classes=classes)
    wall = time.perf_counter() - t0
    summary = trainer.summary()
    summary["wall_time_s"] = wall
    return summary, trainer.history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/stream_data.csv")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--regenerate", action="store_true",
                        help="Rebuild the synthetic dataset before benchmarking.")
    parser.add_argument("--plots", action="store_true",
                        help="Save accuracy/memory curves to benchmark_out/.")
    parser.add_argument("--no-boosting", action="store_true",
                        help="Skip the online-boosting model (much faster; "
                             "boosting is sequential and ~100x slower than the "
                             "forest in pure NumPy).")
    args = parser.parse_args()

    if args.regenerate or not os.path.exists(args.data):
        from make_dataset import make_classification
        from numcompute_stream import save_csv
        X, y = make_classification(random_state=0)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
        save_csv(args.data, X, y, feature_names=names)
        print(f"[regenerated] wrote {args.data}  X={X.shape}")

    X, y, meta = load_csv(args.data, target=-1, has_header=True,
                          categorical_target=True)
    classes = np.unique(y)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(classes)} classes from {args.data}")
    print(f"Chunk size = {args.chunk_size}  "
          f"({int(np.ceil(X.shape[0] / args.chunk_size))} chunks)\n")

    histories = {}
    rows = []
    models = build_models()
    if args.no_boosting:
        models.pop("Boosting(12)", None)
    for name, maker in models.items():
        # fresh chunk list per model so each sees an identical stream
        chunks = list(iter_chunks(X, y, chunk_size=args.chunk_size, shuffle=False))
        summary, history = run_one(maker, chunks, classes)
        histories[name] = history
        rows.append((name, summary))

    header = f"{'model':18s} {'cum_acc':>8s} {'roll_acc':>9s} " \
             f"{'fit_s':>8s} {'wall_s':>8s} {'mem_KB':>8s}"
    print(header)
    print("-" * len(header))
    for name, s in rows:
        print(f"{name:18s} {s['final_cumulative_accuracy']:8.3f} "
              f"{s['final_rolling_accuracy']:9.3f} {s['total_fit_time_s']:8.3f} "
              f"{s['wall_time_s']:8.3f} {s['final_memory_bytes'] / 1024:8.1f}")

    if args.plots:
        from numcompute_stream import visualise
        os.makedirs("benchmark_out", exist_ok=True)
        curves = {n: h["cumulative_accuracy"] for n, h in histories.items()}
        fig = visualise.plot_metric_over_time(
            list(curves.values())[0], title="Cumulative accuracy over chunks",
            ylabel="accuracy", show=False)
        # overlay the rest manually for a multi-line comparison
        ax = fig.axes[0]
        for n, series in curves.items():
            ax.plot(range(len(series)), series, label=n)
        ax.legend()
        fig.savefig("benchmark_out/accuracy_over_time.png", dpi=120,
                    bbox_inches="tight")
        mem = {n: np.array(h["memory_bytes"]) / 1024 for n, h in histories.items()}
        fig2 = visualise.plot_metric_over_time(
            list(mem.values())[0], title="Model memory over chunks",
            ylabel="KB", show=False)
        ax2 = fig2.axes[0]
        for n, series in mem.items():
            ax2.plot(range(len(series)), series, label=n)
        ax2.legend()
        fig2.savefig("benchmark_out/memory_over_time.png", dpi=120,
                     bbox_inches="tight")
        print("\nSaved plots to benchmark_out/")


if __name__ == "__main__":
    main()