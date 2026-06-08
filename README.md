## NumCompute-Stream

A **NumPy-only** streaming machine-learning framework built around incremental,
tree-based classifiers. Every component — preprocessing, statistics, metrics,
trees, ensembles, pipelines — supports `partial_fit` / chunk-wise updates so
models can be trained over a data stream that never fully fits in memory.

> **Constraints honoured:** the package uses **only NumPy** for computation and
> the Python standard library (`csv`, `pickle`, `time`) for IO/utilities.
> **scikit-learn and pandas are *not* used anywhere.** Plotting uses Matplotlib,
> as permitted for the visualisation module.

---

### Why a Hoeffding tree?

A naive "streaming" tree would buffer data and periodically rebuild. Instead the
core learner is a **Hoeffding tree (VFDT)**: each leaf accumulates lightweight
per-class, per-feature Gaussian statistics, and a split is committed only once
the **Hoeffding bound** guarantees (with confidence `1 - delta`) that the best
candidate split really beats the runner-up. This gives genuinely incremental
growth with bounded per-example work — the principled streaming choice, and it
maps directly onto the required `max_depth` / `min_samples_split` (grace period)
/ `max_features` knobs.

---

### Installation

No build step. Requires Python 3.10+, NumPy, and Matplotlib.

```bash
pip install numpy matplotlib          # only hard dependencies
# optional, to run the notebook / tests:
pip install jupyter nbclient ipykernel pytest
```

Put the `numcompute_stream/` package on your import path (the repo root already
is).

---

### Quick start

```python
import numpy as np
from numcompute_stream import (
    load_csv, iter_chunks, Pipeline, Imputer, StandardScaler,
    RandomForestClassifier, StreamTrainer,
)

X, y, meta = load_csv("data/stream_data.csv", target=-1,
                      has_header=True, categorical_target=True)

pipe = Pipeline([
    ("impute", Imputer("mean")),
    ("scale",  StandardScaler()),
    ("model",  RandomForestClassifier(n_estimators=12, max_features="sqrt")),
])

trainer = StreamTrainer(pipe, prequential=True, rolling_window=200)
trainer.run(iter_chunks(X, y, chunk_size=300), classes=np.unique(y))

print(trainer.summary())
```

`StreamTrainer` runs **prequential (test-then-train)** evaluation: each chunk is
scored *before* it is used to update the model, then logged into
`trainer.history` (per-chunk / cumulative / rolling accuracy, fit time, memory).

---

### Package layout

| Module | Purpose |
|--------|---------|
| `io.py` | CSV loading/saving and chunk iterators (`load_csv`, `iter_chunks`, `stream_csv`, `save_csv`) — stdlib `csv` + NumPy only. |
| `stats.py` | Streaming statistics: `StreamingStats` (Welford mean/var), `StreamingHistogram` (optional sliding window), `P2Quantile`. |
| `preprocessing.py` | `StandardScaler` (Welford or EMA), `Imputer`, incremental `OneHotEncoder` — all with `partial_fit`. |
| `metrics.py` | Streaming `Accuracy`, `Precision`, `Recall`, `F1`, `ConfusionMatrix`, `BinaryAUC`, `RollingAccuracy` with `update`/`reset`/`result`. |
| `tree.py` | `HoeffdingTreeClassifier` (aliased `DecisionTreeClassifier`): incremental Gini/entropy tree. |
| `ensemble.py` | `EnsembleClassifier` (online bagging / random forest / Oza boosting) and `RandomForestClassifier`. |
| `pipeline.py` | `Pipeline` chaining streaming transformers + a final model under one `partial_fit`. |
| `stream.py` | `StreamTrainer` prequential harness + `model_size_bytes`. |
| `visualise.py` | Matplotlib helpers: `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`, `plot_confusion_matrix`. |

---

### Running the pieces

```bash
# unit tests (51 tests)
pytest tests/ -q

# regenerate the synthetic concept-drift dataset
python make_dataset.py

# head-to-head streaming benchmark
python benchmark.py --plots               # single tree vs forest vs boosting
python benchmark.py --no-boosting         # fast path (skips slow online boosting)

# demo notebook
jupyter notebook demo/stream_demo.ipynb
```

---

### Using your own data

Any CSV with feature columns followed by a label column works:

```python
X, y, meta = load_csv("your_data.csv", target=-1, has_header=True,
                      categorical_target=True)   # string labels are auto-encoded
```

Missing cells become `NaN` and are handled by the `Imputer` step (and the tree
routes `NaN` deterministically). For a true streaming source, `stream_csv` reads
the file chunk-by-chunk without loading it all into memory.

---

### Notes

* **Concept drift:** the bundled synthetic dataset injects a drift at its
  midpoint; the rolling-accuracy curve shows the models recovering.
* **Online boosting is intentionally sequential** (Oza) and therefore far slower
  than the vectorised forest in pure NumPy — see `benchmark.py --no-boosting`.
* All randomness is seedable via `random_state` for reproducibility.
