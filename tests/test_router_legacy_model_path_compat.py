from __future__ import annotations

from pathlib import Path

from surf_rag.evaluation.router_model_artifacts import build_router_model_root


def test_build_router_model_root_legacy_default() -> None:
    root = build_router_model_root(Path("/data/router"), "rid", "embedding")
    assert root == Path("/data/router/rid/model/embedding")
