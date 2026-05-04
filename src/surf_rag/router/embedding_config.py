"""Embedding provider / cache mode parsing and defaults."""

from __future__ import annotations

EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS = "sentence-transformers"
EMBEDDING_PROVIDER_OPENAI = "openai"

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_CACHE_OFF = "off"
EMBEDDING_CACHE_PREFER = "prefer"
EMBEDDING_CACHE_REQUIRED = "required"
EMBEDDING_CACHE_BUILD = "build"
EMBEDDING_CACHE_AUTO = "auto"


def parse_embedding_provider(value: str | None) -> str:
    s = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "": EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
        "st": EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
        "sentence-transformers": EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
        "sentence_transformers": EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
        "huggingface": EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
        "hf": EMBEDDING_PROVIDER_SENTENCE_TRANSFORMERS,
        "openai": EMBEDDING_PROVIDER_OPENAI,
    }
    if s not in aliases:
        raise ValueError(
            "embedding_provider must be one of "
            f"{sorted({v for v in aliases.values() if v})!r} (aliases: st, openai), got {value!r}"
        )
    return aliases[s]


def resolve_embedding_cache_mode_for_dataset(provider: str, mode: str | None) -> str:
    """Resolve auto/off/prefer/required/build for router dataset build."""
    m = (mode or "").strip().lower().replace("_", "-")
    if m in ("", "auto"):
        if parse_embedding_provider(provider) == EMBEDDING_PROVIDER_OPENAI:
            return EMBEDDING_CACHE_REQUIRED
        return EMBEDDING_CACHE_OFF
    return parse_embedding_cache_mode(m)


def resolve_router_e2e_embedding_cache_mode(
    router_dataset_provider: str, mode: str | None
) -> str:
    """E2E cache mode: auto follows dataset (OpenAI -> required)."""
    m = (mode or "").strip().lower().replace("_", "-")
    if m in ("", "auto"):
        if (
            parse_embedding_provider(router_dataset_provider)
            == EMBEDDING_PROVIDER_OPENAI
        ):
            return EMBEDDING_CACHE_REQUIRED
        return EMBEDDING_CACHE_OFF
    # e2e does not use build mode; treat as off if someone passes build
    resolved = parse_embedding_cache_mode(m)
    if resolved == EMBEDDING_CACHE_BUILD:
        return EMBEDDING_CACHE_OFF
    return resolved


def parse_embedding_cache_mode(value: str | None) -> str:
    s = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "": EMBEDDING_CACHE_OFF,
        "off": EMBEDDING_CACHE_OFF,
        "none": EMBEDDING_CACHE_OFF,
        "prefer": EMBEDDING_CACHE_PREFER,
        "required": EMBEDDING_CACHE_REQUIRED,
        "build": EMBEDDING_CACHE_BUILD,
    }
    if s not in aliases:
        raise ValueError(
            "embedding_cache_mode must be one of "
            f"{EMBEDDING_CACHE_OFF!r}, {EMBEDDING_CACHE_PREFER!r}, "
            f"{EMBEDDING_CACHE_REQUIRED!r}, {EMBEDDING_CACHE_BUILD!r}, got {value!r}"
        )
    return aliases[s]


def resolve_embedding_model_for_provider(provider: str, embedding_model: str) -> str:
    """When provider is OpenAI, default model to text-embedding-3-large if unset or ST default."""
    p = parse_embedding_provider(provider)
    m = (embedding_model or "").strip()
    if p == EMBEDDING_PROVIDER_OPENAI:
        if not m or m == "all-MiniLM-L6-v2":
            return DEFAULT_OPENAI_EMBEDDING_MODEL
        return m
    return m or "all-MiniLM-L6-v2"
