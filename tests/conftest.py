"""Pytest configuration and process-wide test env defaults."""

from __future__ import annotations

import os

# Avoid OpenMP / BLAS duplicate-library aborts (e.g. FAISS + NumPy on macOS)
# when running the full suite after torch/spacy import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
