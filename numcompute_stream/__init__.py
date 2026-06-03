"""numcompute_stream -- NumPy-only streaming ML framework."""
from __future__ import annotations

from .io import load_csv, stream_csv, iter_chunks, save_csv
from .stats import StreamingStats, StreamingHistogram, P2Quantile
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .tree import DecisionTreeClassifier, HoeffdingTreeClassifier
from .ensemble import EnsembleClassifier, RandomForestClassifier

__version__ = "0.1.3"

__all__ = [
    "load_csv", "stream_csv", "iter_chunks", "save_csv",
    "StreamingStats", "StreamingHistogram", "P2Quantile",
    "StandardScaler", "Imputer", "OneHotEncoder",
    "DecisionTreeClassifier", "HoeffdingTreeClassifier",
    "EnsembleClassifier", "RandomForestClassifier",
    "__version__",
]