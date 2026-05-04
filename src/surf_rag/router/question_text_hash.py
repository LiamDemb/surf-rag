"""Canonical question text normalization and hashing for embedding caches.

Matches ``scripts/ingest_data.normalize_question`` + ``surf_rag.core.schemas.sha256_text``.
"""

from __future__ import annotations

from surf_rag.core.schemas import sha256_text


def normalize_benchmark_question_text(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace (same as ingest_data)."""
    return " ".join(text.lower().strip().split())


def canonical_question_text_hash(text: str) -> str:
    """SHA-256 hex digest of normalized question text (ingest-compatible)."""
    return sha256_text(normalize_benchmark_question_text(text))
