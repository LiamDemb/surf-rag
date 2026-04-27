"""Batch query embedding using the project embedder."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from surf_rag.core.embedder import SentenceTransformersEmbedder


def embed_queries(
    questions: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 32,
    embedder: Optional[SentenceTransformersEmbedder] = None,
    embedder_factory: Optional[Callable[[], SentenceTransformersEmbedder]] = None,
) -> np.ndarray:
    """Return float32 array ``(len(questions), dim)``, L2-normalized per vector.

    ``embedder_factory`` is invoked only when ``embedder`` is ``None`` (lazy init).
    Callers should pass ``embedder_factory=ictx.query_embedder`` to reuse a shared
    embedder without eagerly constructing it when ``embed_queries`` is patched in tests.
    """
    if embedder is not None:
        e = embedder
    elif embedder_factory is not None:
        e = embedder_factory()
    else:
        e = SentenceTransformersEmbedder(model_name=model_name)
    texts = [str(q or "") for q in questions]
    model = e.model
    all_vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(all_vecs, dtype=np.float32)
