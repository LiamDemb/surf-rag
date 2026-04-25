"""Batch query embedding using the project embedder."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from surf_rag.core.embedder import SentenceTransformersEmbedder


def embed_queries(
    questions: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Return float32 array ``(len(questions), dim)``, L2-normalized per vector."""
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    # Batch encode for throughput
    texts = [str(q or "") for q in questions]
    model = embedder.model
    all_vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(all_vecs, dtype=np.float32)
