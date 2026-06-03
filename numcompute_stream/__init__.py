"""numcompute_stream -- NumPy-only streaming ML framework.
"""
from __future__ import annotations

from .io import load_csv, stream_csv, iter_chunks, save_csv

__version__ = "0.1.1"

__all__ = ["load_csv", "stream_csv", "iter_chunks", "save_csv", "__version__"]