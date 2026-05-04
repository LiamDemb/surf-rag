"""Provider-specific batch text embedding (SentenceTransformers, OpenAI)."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import numpy as np

from surf_rag.core.embedder import SentenceTransformersEmbedder
from surf_rag.router.embedding_config import (
    EMBEDDING_PROVIDER_OPENAI,
    EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
    parse_embedding_provider,
)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, np.float32(1e-12))
    return (x / norms).astype(np.float32)


def _embed_sentence_transformers(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    embedder: Optional[SentenceTransformersEmbedder] = None,
    embedder_factory: Optional[Callable[[], SentenceTransformersEmbedder]] = None,
) -> np.ndarray:
    if embedder is not None:
        e = embedder
    elif embedder_factory is not None:
        e = embedder_factory()
    else:
        e = SentenceTransformersEmbedder(model_name=model_name)
    model = e.model
    all_vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(all_vecs, dtype=np.float32)


def _embed_openai(
    texts: list[str],
    *,
    model: str,
    batch_size: int,
    client: Any | None = None,
    dimensions: int | None = None,
) -> np.ndarray:
    from openai import OpenAI

    c = client if client is not None else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out_rows: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        kwargs: dict[str, Any] = {"model": model, "input": batch}
        if dimensions is not None:
            kwargs["dimensions"] = int(dimensions)
        resp = c.embeddings.create(**kwargs)
        data = sorted(resp.data, key=lambda d: int(d.index))
        if len(data) != len(batch):
            raise RuntimeError(
                f"OpenAI embeddings response size mismatch: got {len(data)}, expected {len(batch)}"
            )
        for d in data:
            out_rows.append(np.asarray(d.embedding, dtype=np.float32))
    stacked = (
        np.stack(out_rows, axis=0) if out_rows else np.zeros((0, 0), dtype=np.float32)
    )
    return _l2_normalize_rows(stacked)


def embed_texts(
    texts: Sequence[str],
    *,
    provider: str,
    model: str,
    batch_size: int = 32,
    embedder: Optional[SentenceTransformersEmbedder] = None,
    embedder_factory: Optional[Callable[[], SentenceTransformersEmbedder]] = None,
    openai_client: Any | None = None,
    openai_dimensions: int | None = None,
) -> np.ndarray:
    """Return ``(N, D)`` float32; L2-normalized rows for both providers."""
    prov = parse_embedding_provider(provider)
    tlist = [str(x or "") for x in texts]
    if prov == EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS:
        return _embed_sentence_transformers(
            tlist,
            model_name=model,
            batch_size=batch_size,
            embedder=embedder,
            embedder_factory=embedder_factory,
        )
    if prov == EMBEDDING_PROVIDER_OPENAI:
        return _embed_openai(
            tlist,
            model=model,
            batch_size=batch_size,
            client=openai_client,
            dimensions=openai_dimensions,
        )
    raise ValueError(f"unknown embedding provider: {provider!r}")
