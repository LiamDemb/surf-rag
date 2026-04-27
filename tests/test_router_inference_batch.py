"""Batched router query tensors reuse a single SentenceTransformersEmbedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from surf_rag.router.inference_inputs import (
    RouterInferenceContext,
    compute_query_tensors_for_router_batch,
)
from surf_rag.router.query_features import V1_FEATURE_NAMES


@pytest.fixture
def fake_router_ctx() -> RouterInferenceContext:
    router = MagicMock()
    router.config.embedding_dim = 4
    router.config.feature_dim = 2
    normalizer = MagicMock()
    feat_ctx = MagicMock()
    feat_ctx.nlp = None
    feat_ctx.entity_pipeline = None
    return RouterInferenceContext(
        router=router,
        normalizer=normalizer,
        embedding_model="dummy-embed",
        input_mode="both",
        feature_context=feat_ctx,
    )


@patch("surf_rag.router.inference_inputs.embed_queries")
@patch("surf_rag.router.inference_inputs.extract_features_v1")
@patch("surf_rag.router.inference_inputs.transform_row", side_effect=lambda f, _n: f)
def test_batch_calls_embed_queries_once_per_batch(
    mock_transform_row: MagicMock,
    mock_extract_features: MagicMock,
    mock_embed_queries: MagicMock,
    fake_router_ctx: RouterInferenceContext,
) -> None:
    zero = {n: 0.0 for n in V1_FEATURE_NAMES}
    mock_extract_features.return_value = zero
    mock_embed_queries.return_value = np.zeros((2, 4), dtype=np.float32)
    compute_query_tensors_for_router_batch(
        ["q1", "q2"], fake_router_ctx, st_batch_size=32
    )
    assert mock_embed_queries.call_count == 1
    assert mock_extract_features.call_count == 2
