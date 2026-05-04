"""Tests for oracle objective config parsing."""

from __future__ import annotations

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
