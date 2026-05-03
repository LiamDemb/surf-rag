from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from surf_rag.evaluation.router_dataset_artifacts import RouterDatasetPaths
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.router.inference_inputs import load_router_inference_context
from surf_rag.router.model import RouterMLP, RouterMLPConfig


def _write_dataset_side(router_base: Path, router_id: str, corpus_dir: Path) -> None:
    ds = RouterDatasetPaths(run_root=router_base / router_id / "dataset")
    ds.ensure_dirs()
    ds.feature_stats.write_text(
        json.dumps({"version": "1", "means": {}, "stds": {}}), encoding="utf-8"
    )
    ds.manifest.write_text(
        json.dumps(
            {
                "embedding_model": "all-MiniLM-L6-v2",
                "source_corpus": {"retrieval_asset_dir": str(corpus_dir)},
            }
        ),
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    cfg = RouterMLPConfig(
        embedding_dim=4,
        feature_dim=2,
        input_mode="both",
        embed_proj_dim=2,
        feat_proj_dim=2,
        hidden_dim=4,
        dropout=0.0,
    )
    model = RouterMLP(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "mlp-v1",
            "architecture_kwargs": {},
            "config": cfg.to_json(),
        },
        path,
    )


def test_auto_resolve_single_architecture_dir(tmp_path: Path) -> None:
    router_base = tmp_path / "router"
    router_id = "r1"
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_dataset_side(router_base, router_id, corpus)
    mp = make_router_model_paths_for_cli(
        router_id,
        router_base=router_base,
        input_mode="both",
        router_architecture_id="mlp-v1-default",
    )
    _write_checkpoint(mp.checkpoint)
    mp.manifest.write_text(
        json.dumps({"model": {"weight_grid": [0.0, 0.5, 1.0]}}), encoding="utf-8"
    )
    ctx = load_router_inference_context(
        router_id,
        input_mode="both",
        router_architecture_id=None,
        router_base=router_base,
        retrieval_asset_dir=corpus,
        device="cpu",
    )
    assert ctx.router.architecture == "mlp-v1"


def test_auto_resolve_fails_for_multiple_architecture_dirs(tmp_path: Path) -> None:
    router_base = tmp_path / "router"
    router_id = "r2"
    corpus = tmp_path / "corpus2"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_dataset_side(router_base, router_id, corpus)
    (router_base / router_id / "models" / "a").mkdir(parents=True, exist_ok=True)
    (router_base / router_id / "models" / "b").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="router_architecture_id is required"):
        load_router_inference_context(
            router_id,
            input_mode="both",
            router_architecture_id=None,
            router_base=router_base,
            retrieval_asset_dir=corpus,
            device="cpu",
        )


def test_fallback_to_legacy_model_dir_when_models_missing(tmp_path: Path) -> None:
    router_base = tmp_path / "router"
    router_id = "r3"
    corpus = tmp_path / "corpus3"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_dataset_side(router_base, router_id, corpus)
    legacy = make_router_model_paths_for_cli(
        router_id,
        router_base=router_base,
        input_mode="both",
        router_architecture_id=None,
    )
    _write_checkpoint(legacy.checkpoint)
    legacy.manifest.write_text(
        json.dumps({"model": {"weight_grid": [0.0, 0.5, 1.0]}}), encoding="utf-8"
    )
    ctx = load_router_inference_context(
        router_id,
        input_mode="both",
        router_architecture_id=None,
        router_base=router_base,
        retrieval_asset_dir=corpus,
        device="cpu",
    )
    assert ctx.router.architecture == "mlp-v1"
