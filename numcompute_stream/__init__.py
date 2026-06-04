"""numcompute_stream -- a NumPy-only streaming, tree-based ML framework.
"""
from __future__ import annotations

from .io import load_csv, stream_csv, iter_chunks, save_csv
from .stats import StreamingStats, StreamingHistogram, P2Quantile
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .tree import DecisionTreeClassifier, HoeffdingTreeClassifier
from .ensemble import EnsembleClassifier, RandomForestClassifier
from .pipeline import Pipeline
from .stream import StreamTrainer, model_size_bytes
from .metrics import (
    Accuracy, Precision, Recall, F1, ConfusionMatrix,
    BinaryAUC, RollingAccuracy, accuracy_score, StreamingMetric,
)
from . import visualise

__version__ = "0.2.0"

__all__ = [
    "load_csv", "stream_csv", "iter_chunks", "save_csv",
    "StreamingStats", "StreamingHistogram", "P2Quantile",
    "StandardScaler", "Imputer", "OneHotEncoder",
    "DecisionTreeClassifier", "HoeffdingTreeClassifier",
    "EnsembleClassifier", "RandomForestClassifier",
    "Pipeline", "StreamTrainer", "model_size_bytes",
    "Accuracy", "Precision", "Recall", "F1", "ConfusionMatrix",
    "BinaryAUC", "RollingAccuracy", "accuracy_score", "StreamingMetric",
    "visualise", "__version__",
]