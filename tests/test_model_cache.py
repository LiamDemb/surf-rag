"""Shared SentenceTransformer / CrossEncoder cache."""

from unittest.mock import MagicMock, patch

import pytest

from surf_rag.core.model_cache import (
    clear_model_caches,
    get_cross_encoder,
    get_sentence_transformer,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_model_caches()
    yield
    clear_model_caches()


@patch("sentence_transformers.SentenceTransformer")
def test_sentence_transformer_reuses_instance(mock_st: MagicMock) -> None:
    mock_st.return_value = MagicMock(name="model")
    a = get_sentence_transformer("dummy-st")
    b = get_sentence_transformer("dummy-st")
    assert a is b
    assert mock_st.call_count == 1


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reuses_instance(mock_ce: MagicMock) -> None:
    mock_ce.return_value = MagicMock(name="ce")
    a = get_cross_encoder("dummy-ce")
    b = get_cross_encoder("dummy-ce")
    assert a is b
    assert mock_ce.call_count == 1
