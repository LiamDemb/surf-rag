"""Batch query embedding using the project embedder or OpenAI."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from surf_rag.core.embedder import SentenceTransformersEmbedder
from surf_rag.router.embedding_config import EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS
from surf_rag.router.embedding_providers import embed_texts


def embed_queries(
    questions: Sequence[str],
    *,
    model_name: str,
    provider: str = EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
    batch_size: int = 32,
    embedder: Optional[SentenceTransformersEmbedder] = None,
    embedder_factory: Optional[Callable[[], SentenceTransformersEmbedder]] = None,
    openai_client: Any | None = None,
    openai_dimensions: int | None = None,
) -> np.ndarray:
    """Return float32 array ``(len(questions), dim)``, L2-normalized per vector.

    ``embedder_factory`` is used only for SentenceTransformers when ``embedder`` is
    ``None`` (lazy init). For OpenAI, pass ``openai_client`` in tests.
    """
    return embed_texts(
        questions,
        provider=provider,
        model=model_name,
        batch_size=batch_size,
        embedder=embedder,
        embedder_factory=embedder_factory,
        openai_client=openai_client,
        openai_dimensions=openai_dimensions,
    )
