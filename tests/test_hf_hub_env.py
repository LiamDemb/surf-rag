"""Tests for Hugging Face Hub token env normalization."""

from __future__ import annotations

import os

import pytest

from surf_rag.config.env import (
    HF_TOKEN_ENV,
    HUGGING_FACE_HUB_TOKEN_ENV,
    sync_hf_hub_token_env,
)


@pytest.fixture
def clean_hf_tokens() -> None:
    old_hf = os.environ.pop(HF_TOKEN_ENV, None)
    old_hub = os.environ.pop(HUGGING_FACE_HUB_TOKEN_ENV, None)
    yield
    if old_hf is not None:
        os.environ[HF_TOKEN_ENV] = old_hf
    else:
        os.environ.pop(HF_TOKEN_ENV, None)
    if old_hub is not None:
        os.environ[HUGGING_FACE_HUB_TOKEN_ENV] = old_hub
    else:
        os.environ.pop(HUGGING_FACE_HUB_TOKEN_ENV, None)


def test_sync_copies_hf_token_to_hub_key(clean_hf_tokens: None) -> None:
    os.environ[HF_TOKEN_ENV] = "hf_unit_test_value"
    sync_hf_hub_token_env()
    assert os.environ[HUGGING_FACE_HUB_TOKEN_ENV] == "hf_unit_test_value"


def test_sync_copies_hub_key_to_hf_token(clean_hf_tokens: None) -> None:
    os.environ[HUGGING_FACE_HUB_TOKEN_ENV] = "hf_from_hub_env"
    sync_hf_hub_token_env()
    assert os.environ[HF_TOKEN_ENV] == "hf_from_hub_env"


def test_sync_strips_whitespace_for_emptiness(clean_hf_tokens: None) -> None:
    os.environ[HF_TOKEN_ENV] = "   "
    sync_hf_hub_token_env()
    assert (
        HUGGING_FACE_HUB_TOKEN_ENV not in os.environ
        or not os.environ.get(HUGGING_FACE_HUB_TOKEN_ENV, "").strip()
    )


def test_sync_does_not_overwrite_when_both_set(clean_hf_tokens: None) -> None:
    os.environ[HF_TOKEN_ENV] = "a"
    os.environ[HUGGING_FACE_HUB_TOKEN_ENV] = "b"
    sync_hf_hub_token_env()
    assert os.environ[HF_TOKEN_ENV] == "a"
    assert os.environ[HUGGING_FACE_HUB_TOKEN_ENV] == "b"
