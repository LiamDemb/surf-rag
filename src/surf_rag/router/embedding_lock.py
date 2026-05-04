"""Validate router checkpoint embedding settings vs dataset manifest."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from surf_rag.router.embedding_config import (
    EMBEDDING_PROVIDER_OPENAI,
    EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
    parse_embedding_provider,
)


def infer_embedding_provider_from_model(model: str) -> str:
    m = (model or "").strip().lower()
    if m.startswith("text-embedding") or m.startswith("text-embedding-"):
        return EMBEDDING_PROVIDER_OPENAI
    return EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS


def validate_router_embedding_compatibility(
    *,
    model_manifest: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` if trained router embedding stack disagrees with dataset."""
    mm = dict(model_manifest.get("model") or {})
    dm_emb_model = str(dataset_manifest.get("embedding_model") or "").strip()
    dm_prov_raw = dataset_manifest.get("embedding_provider")
    dm_prov = (
        parse_embedding_provider(str(dm_prov_raw))
        if str(dm_prov_raw or "").strip()
        else infer_embedding_provider_from_model(dm_emb_model)
    )
    m_model = str(mm.get("embedding_model") or "").strip()
    if dm_emb_model and m_model and dm_emb_model != m_model:
        raise ValueError(
            "Router embedding_model mismatch between model manifest and dataset manifest: "
            f"model={m_model!r} dataset={dm_emb_model!r}. Retrain the router on this dataset."
        )
    m_prov_raw = mm.get("embedding_provider")
    m_prov = (
        parse_embedding_provider(str(m_prov_raw))
        if str(m_prov_raw or "").strip()
        else infer_embedding_provider_from_model(m_model)
    )
    if m_prov != dm_prov:
        raise ValueError(
            "Router embedding_provider mismatch: "
            f"model_manifest={m_prov!r} dataset_manifest={dm_prov!r}. "
            "Use the same embedding provider as the router dataset."
        )
    m_dim = mm.get("embedding_dim")
    arch = mm.get("architecture") or {}
    if m_dim is None and isinstance(arch, dict):
        m_dim = arch.get("embedding_dim")
    d_dim = dataset_manifest.get("embedding_dim")
    if d_dim is None:
        d_dim = (dataset_manifest.get("embedding_cache") or {}).get("dim")
    if m_dim is not None and d_dim is not None and int(m_dim) != int(d_dim):
        raise ValueError(
            f"Router embedding_dim mismatch: model={int(m_dim)} dataset_cache={int(d_dim)}"
        )
