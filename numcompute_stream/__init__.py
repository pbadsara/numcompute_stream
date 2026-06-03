"""numcompute_stream -- NumPy-only streaming ML framework."""
from __future__ import annotations

from .io import load_csv, stream_csv, iter_chunks, save_csv
from .stats import StreamingStats, StreamingHistogram, P2Quantile
from .preprocessing import StandardScaler, Imputer, OneHotEncoder

__version__ = "0.1.2"

__all__ = [
    "load_csv", "stream_csv", "iter_chunks", "save_csv",
    "StreamingStats", "StreamingHistogram", "P2Quantile",
    "StandardScaler", "Imputer", "OneHotEncoder",
    "__version__",
]