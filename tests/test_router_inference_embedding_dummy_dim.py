"""Placeholder feature width for embedding-only router input mode (e.g. MLP v2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("torch")

from surf_rag.router.embedding_config import EMBEDDING_CACHE_OFF
from surf_rag.router.inference import LoadedRouter
from surf_rag.router.inference_inputs import (
    RouterInferenceContext,
    _dummy_feature_vector_dim_for_embedding_mode,
    compute_query_tensors_for_router_batch,
)
from surf_rag.router.model import (
    ROUTER_TASK_CLASSIFICATION,
    RouterMLP,
    RouterMLPConfig,
    RouterMLPv2,
    RouterMLPv2Config,
)


def _minimal_ictx(router: LoadedRouter) -> RouterInferenceContext:
    return RouterInferenceContext(
        router=router,
        normalizer=MagicMock(),
        embedding_model="stub-model",
        input_mode="embedding",
        feature_context=MagicMock(entity_pipeline=None, nlp=None),
        embedding_cache_mode=EMBEDDING_CACHE_OFF,
    )


def test_dummy_dim_mlp_v2_falls_back_to_v1_feature_width() -> None:
    cfg = RouterMLPv2Config(
        embedding_dim=16,
        task_type=ROUTER_TASK_CLASSIFICATION,
    )
    assert not hasattr(cfg, "feature_dim")
    model = RouterMLPv2(cfg)
    router = LoadedRouter(
        model=model,
        config=cfg,
        architecture="mlp-v2",
        architecture_kwargs={},
        weight_grid=np.array([0.0, 1.0], dtype=np.float64),
        device="cpu",
        manifest={},
        task_type="classification",
    )
    from surf_rag.router.query_features import V1_FEATURE_NAMES

    assert _dummy_feature_vector_dim_for_embedding_mode(router) == len(V1_FEATURE_NAMES)


def test_dummy_dim_uses_config_when_feature_dim_present() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=8,
        feature_dim=5,
        task_type=ROUTER_TASK_CLASSIFICATION,
    )
    router = LoadedRouter(
        model=MagicMock(),
        config=cfg,
        architecture="mlp-v1",
        architecture_kwargs={},
        weight_grid=np.array([0.0], dtype=np.float64),
        device="cpu",
        manifest={},
        task_type="classification",
    )
    assert _dummy_feature_vector_dim_for_embedding_mode(router) == 5


def test_compute_query_tensors_embedding_mlp_v2_shapes() -> None:
    cfg = RouterMLPv2Config(
        embedding_dim=16,
        task_type=ROUTER_TASK_CLASSIFICATION,
    )
    model = RouterMLPv2(cfg)
    router = LoadedRouter(
        model=model,
        config=cfg,
        architecture="mlp-v2",
        architecture_kwargs={},
        weight_grid=np.array([0.0, 1.0], dtype=np.float64),
        device="cpu",
        manifest={},
        task_type="classification",
    )
    ictx = _minimal_ictx(router)
    from surf_rag.router.query_features import V1_FEATURE_NAMES

    with patch(
        "surf_rag.router.inference_inputs.embed_queries",
        return_value=np.ones((2, 16), dtype=np.float32),
    ):
        qe, qf = compute_query_tensors_for_router_batch(
            ["a", "b"], ictx, st_batch_size=8
        )
    assert qe.shape == (2, 16)
    assert qf.shape == (2, len(V1_FEATURE_NAMES))
    assert qf.dtype == np.float32
    assert np.all(qf == 0.0)


def test_predict_class_id_batch_mlp_v2_with_dummy_features() -> None:
    """End-to-end: tensors from compute path must work with batched predict."""
    import torch

    from surf_rag.router.inference import predict_class_id_batch

    cfg = RouterMLPv2Config(
        embedding_dim=4,
        hidden_dim_1=8,
        hidden_dim_2=4,
        task_type=ROUTER_TASK_CLASSIFICATION,
    )
    model = RouterMLPv2(cfg)
    router = LoadedRouter(
        model=model,
        config=cfg,
        architecture="mlp-v2",
        architecture_kwargs={},
        weight_grid=np.array([0.0, 1.0], dtype=np.float64),
        device="cpu",
        manifest={},
        task_type="classification",
    )
    ictx = _minimal_ictx(router)
    with patch(
        "surf_rag.router.inference_inputs.embed_queries",
        return_value=np.random.default_rng(0)
        .standard_normal((1, 4))
        .astype(np.float32),
    ):
        qe, qf = compute_query_tensors_for_router_batch(["q"], ictx)
    out = predict_class_id_batch(router, qe, qf)
    assert out.shape == (1,)
    assert int(out[0]) in (0, 1)


def test_compute_query_tensors_embedding_mlp_v1_respects_feature_dim() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=12,
        feature_dim=3,
        task_type=ROUTER_TASK_CLASSIFICATION,
    )
    model = RouterMLP(cfg)
    router = LoadedRouter(
        model=model,
        config=cfg,
        architecture="mlp-v1",
        architecture_kwargs={},
        weight_grid=np.array([0.0], dtype=np.float64),
        device="cpu",
        manifest={},
        task_type="classification",
    )
    ictx = _minimal_ictx(router)
    with patch(
        "surf_rag.router.inference_inputs.embed_queries",
        return_value=np.zeros((1, 12), dtype=np.float32),
    ):
        qe, qf = compute_query_tensors_for_router_batch(["x"], ictx)
    assert qe.shape == (1, 12)
    assert qf.shape == (1, 3)
