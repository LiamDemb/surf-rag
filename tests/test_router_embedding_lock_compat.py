from __future__ import annotations

import pytest

from surf_rag.router.embedding_lock import validate_router_embedding_compatibility


def test_provider_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="embedding_provider mismatch"):
        validate_router_embedding_compatibility(
            model_manifest={
                "model": {
                    "embedding_model": "text-embedding-3-large",
                    "embedding_provider": "openai",
                }
            },
            dataset_manifest={
                "embedding_model": "text-embedding-3-large",
                "embedding_provider": "sentence-transformers",
            },
        )


def test_model_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="embedding_model mismatch"):
        validate_router_embedding_compatibility(
            model_manifest={
                "model": {
                    "embedding_model": "text-embedding-3-small",
                    "embedding_provider": "openai",
                }
            },
            dataset_manifest={
                "embedding_model": "text-embedding-3-large",
                "embedding_provider": "openai",
            },
        )


def test_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="embedding_dim mismatch"):
        validate_router_embedding_compatibility(
            model_manifest={"model": {"embedding_dim": 3072, "embedding_model": "m"}},
            dataset_manifest={
                "embedding_model": "m",
                "embedding_dim": 1024,
            },
        )


def test_compatible_ok() -> None:
    validate_router_embedding_compatibility(
        model_manifest={
            "model": {
                "embedding_model": "text-embedding-3-large",
                "embedding_provider": "openai",
                "embedding_dim": 3072,
            }
        },
        dataset_manifest={
            "embedding_model": "text-embedding-3-large",
            "embedding_provider": "openai",
            "embedding_dim": 3072,
        },
    )
