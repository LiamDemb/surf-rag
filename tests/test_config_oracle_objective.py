"""Tests for oracle objective config parsing."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from surf_rag.config.loader import pipeline_config_from_dict


def test_pipeline_config_parses_oracle_metric_and_k() -> None:
    cfg = pipeline_config_from_dict(
        {
            "oracle": {
                "oracle_metric": "recall",
                "oracle_metric_k": 20,
            }
        }
    )
    assert cfg.oracle.oracle_metric == "recall"
    assert cfg.oracle.oracle_metric_k == 20


def test_pipeline_config_router_task_defaults_and_overrides() -> None:
    cfg = pipeline_config_from_dict(
        {
            "router": {"train": {"task_type": "classification"}},
            "e2e": {"router_task_type": "classification"},
        }
    )
    assert cfg.router.train.task_type == "classification"
    assert cfg.e2e.router_task_type == "classification"
