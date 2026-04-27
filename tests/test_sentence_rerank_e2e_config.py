"""E2E named config for sentence-window rerank experiment."""

from __future__ import annotations

from pathlib import Path

from surf_rag.config.loader import load_pipeline_config

_REPO = Path(__file__).resolve().parents[1]
CONFIG = (
    _REPO / "configs" / "e2e" / "surf-bench" / "200-test" / "sentence-rerank-test.yaml"
)


def test_sentence_rerank_config_is_sentence_window_and_prompt() -> None:
    cfg = load_pipeline_config(CONFIG)
    assert str(cfg.e2e.reranker).lower() in ("sentence_window",)
    assert "generator_sentence_windows" in str(cfg.generation.prompt_file)
    assert int(cfg.e2e.branch_top_k) == 20
    assert int(cfg.e2e.fusion_keep_k) == 20
    assert int(cfg.e2e.sentence_window_max_windows) == 15
    assert int(cfg.e2e.sentence_window_min_windows) == 8
    assert int(cfg.e2e.sentence_window_max_words) == 1280
    assert int(cfg.e2e.sentence_window_max_subwindow_words) == 180
