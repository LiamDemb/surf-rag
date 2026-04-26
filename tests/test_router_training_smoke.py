"""Smoke test for router training on a tiny synthetic Parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.evaluation.router_dataset_artifacts import (
    RouterDatasetPaths,
    write_router_dataset_manifest,
)
from surf_rag.router.training import (
    RouterTrainConfig,
    save_checkpoint,
    train_router,
)
from surf_rag.evaluation.router_model_artifacts import (
    make_router_model_paths_for_cli,
    read_json,
)
from surf_rag.router.model import RouterMLPConfig

pytest.importorskip("torch")
import torch  # noqa: E402


def _row(qid: str, split: str) -> dict:
    w = [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    d = [1.0 / 11.0] * 11
    return {
        "question_id": qid,
        "split": split,
        "embedding_dim": 4,
        "query_embedding": [0.1, 0.2, 0.3, 0.4],
        "feature_vector_norm": [0.0] * 14,
        "distribution": d,
        "weight_grid": w,
        "feature_set_version": "1",
        "embedding_model": "m",
    }


def test_train_smoke(tmp_path: Path) -> None:
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r.parquet"
    df.to_parquet(pq, index=False)
    mdir = tmp_path / "d"
    mdir.mkdir()
    dpaths = RouterDatasetPaths(run_root=mdir)
    dpaths.ensure_dirs()
    write_router_dataset_manifest(
        dpaths,
        router_id="t",
        source_benchmark_name="m",
        source_benchmark_id="v",
        benchmark_path="/b.jsonl",
        retrieval_asset_dir="/c",
        oracle_run_root="/o",
        labels_selected_path="/l",
        selected_beta=1.0,
        feature_set_version="1",
        embedding_model="m",
        split_seed=0,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    out_model = make_router_model_paths_for_cli("t", router_base=tmp_path, input_mode="both")
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t",
        output_dir=out_model.run_root,
        epochs=3,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        input_mode="both",
    )
    result = train_router(cfg)
    assert result.history
    mcfg: RouterMLPConfig = result.model.config
    save_checkpoint(out_model.checkpoint, result.model, mcfg)
    assert out_model.checkpoint.is_file()
    try:
        pack = torch.load(out_model.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(out_model.checkpoint, map_location="cpu")
    assert "state_dict" in pack
    mpath = out_model.metrics
    mpath.write_text(json.dumps({"splits": result.metrics}), encoding="utf-8")
    assert read_json(mpath).get("splits") is not None


def test_train_smoke_query_features(tmp_path: Path) -> None:
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r2.parquet"
    df.to_parquet(pq, index=False)
    mdir = tmp_path / "d2"
    mdir.mkdir()
    dpaths = RouterDatasetPaths(run_root=mdir)
    dpaths.ensure_dirs()
    write_router_dataset_manifest(
        dpaths,
        router_id="t2",
        source_benchmark_name="m",
        source_benchmark_id="v",
        benchmark_path="/b.jsonl",
        retrieval_asset_dir="/c",
        oracle_run_root="/o",
        labels_selected_path="/l",
        selected_beta=1.0,
        feature_set_version="1",
        embedding_model="m",
        split_seed=0,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    out_model = make_router_model_paths_for_cli(
        "t2", router_base=tmp_path, input_mode="query-features"
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t2",
        output_dir=out_model.run_root,
        epochs=3,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        input_mode="query-features",
    )
    result = train_router(cfg)
    mcfg: RouterMLPConfig = result.model.config
    assert mcfg.input_mode == "query-features"
    save_checkpoint(out_model.checkpoint, result.model, mcfg)
    assert out_model.checkpoint.is_file()
    assert (tmp_path / "t2" / "model" / "query-features" / "model.pt").is_file()
