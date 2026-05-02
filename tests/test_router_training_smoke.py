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

pytest.importorskip("torch")
import torch  # noqa: E402


def _row(qid: str, split: str, *, valid: bool = True) -> dict:
    w = [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    c = [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    return {
        "question_id": qid,
        "split": split,
        "dataset_source": "nq",
        "embedding_dim": 4,
        "query_embedding": [0.1, 0.2, 0.3, 0.4],
        "feature_vector_norm": [0.0] * 14,
        "oracle_curve": c,
        "oracle_best_weight": 1.0,
        "oracle_best_score": 1.0,
        "oracle_curve_std": 0.3,
        "is_valid_for_router_training": valid,
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
        router_labels_path="/l",
        feature_set_version="1",
        embedding_model="m",
        split_seed=0,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    out_model = make_router_model_paths_for_cli(
        "t",
        router_base=tmp_path,
        input_mode="both",
        router_architecture_id="mlp-v1-default",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t",
        output_dir=out_model.run_root,
        epochs=3,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="both",
    )
    result = train_router(cfg)
    assert result.history
    assert result.history[-1]["train_loss"] == pytest.approx(
        result.history[-1]["train_regret"], rel=0, abs=1e-5
    )
    assert result.loss_effective == "regret"
    assert not result.loss_fallback
    mcfg = result.model.config
    save_checkpoint(
        out_model.checkpoint,
        result.model,
        mcfg,
        architecture="mlp-v1",
        architecture_kwargs={},
    )
    assert out_model.checkpoint.is_file()
    try:
        pack = torch.load(out_model.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(out_model.checkpoint, map_location="cpu")
    assert "state_dict" in pack
    assert pack["architecture"] == "mlp-v1"
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
        router_labels_path="/l",
        feature_set_version="1",
        embedding_model="m",
        split_seed=0,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    out_model = make_router_model_paths_for_cli(
        "t2",
        router_base=tmp_path,
        input_mode="query-features",
        router_architecture_id="mlp-v1-qf",
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
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="query-features",
    )
    result = train_router(cfg)
    mcfg = result.model.config
    assert mcfg.input_mode == "query-features"
    save_checkpoint(
        out_model.checkpoint,
        result.model,
        mcfg,
        architecture="mlp-v1",
        architecture_kwargs={},
    )
    assert out_model.checkpoint.is_file()
    assert (
        tmp_path / "t2" / "models" / "mlp-v1-qf" / "query-features" / "model.pt"
    ).is_file()


def test_train_smoke_polyreg_v1(tmp_path: Path) -> None:
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r_poly.parquet"
    df.to_parquet(pq, index=False)
    out_model = make_router_model_paths_for_cli(
        "t_poly",
        router_base=tmp_path,
        input_mode="query-features",
        router_architecture_id="polyreg-v1-smoke",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t_poly",
        output_dir=out_model.run_root,
        epochs=3,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="polyreg-v1",
        architecture_kwargs={"degree": 2},
        input_mode="query-features",
    )
    result = train_router(cfg)
    assert result.model.config.degree == 2
    save_checkpoint(
        out_model.checkpoint,
        result.model,
        result.model.config,
        architecture="polyreg-v1",
        architecture_kwargs=cfg.architecture_kwargs or {},
    )
    assert out_model.checkpoint.is_file()


def test_train_ignores_invalid_rows_in_metrics(tmp_path: Path) -> None:
    rows = [
        _row("a", "train", valid=True),
        _row("b", "train", valid=False),
        _row("c", "dev", valid=False),
        _row("d", "test", valid=True),
        _row("e", "test", valid=False),
    ]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r3.parquet"
    df.to_parquet(pq, index=False)
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t3",
        output_dir=tmp_path / "out3",
        epochs=2,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="both",
    )
    result = train_router(cfg)
    tr = result.metrics["train"]
    dv = result.metrics["dev"]
    te = result.metrics["test"]
    assert tr["num_rows_total"] == 2.0
    assert tr["num_rows_router_eligible"] == 1.0
    assert tr["num_rows_router_ignored_all_zero"] == 1.0
    assert tr["num_rows"] == 1.0
    assert dv["num_rows_total"] == 1.0
    assert dv["num_rows_router_eligible"] == 0.0
    assert dv["num_rows_router_ignored_all_zero"] == 1.0
    assert dv["num_rows"] == 0.0
    assert te["num_rows_total"] == 2.0
    assert te["num_rows_router_eligible"] == 1.0
    assert te["num_rows_router_ignored_all_zero"] == 1.0
    assert result.metrics["router_quality_filtering"]["num_rows_total"] == 5.0
    assert result.metrics["router_quality_filtering"]["num_rows_router_eligible"] == 2.0


def test_train_fails_when_train_split_has_no_eligible_rows(tmp_path: Path) -> None:
    rows = [
        _row("a", "train", valid=False),
        _row("b", "train", valid=False),
        _row("c", "dev", valid=True),
    ]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r4.parquet"
    df.to_parquet(pq, index=False)
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t4",
        output_dir=tmp_path / "out4",
        epochs=2,
        batch_size=2,
        device="cpu",
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="both",
    )
    with pytest.raises(ValueError, match="No router-eligible rows in train split"):
        train_router(cfg)


def test_train_smoke_hinge_squared_regret(tmp_path: Path) -> None:
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r_hinge.parquet"
    df.to_parquet(pq, index=False)
    out_model = make_router_model_paths_for_cli(
        "t_hinge",
        router_base=tmp_path,
        input_mode="both",
        router_architecture_id="mlp-v1-hinge",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t_hinge",
        output_dir=out_model.run_root,
        epochs=3,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="both",
        loss="hinge_squared_regret",
        loss_kwargs={"epsilon": 0.01},
    )
    result = train_router(cfg)
    assert result.loss_effective == "hinge_squared_regret"
    assert not result.loss_fallback
    h = result.history[-1]
    assert "train_loss" in h and "train_regret" in h
    assert "dev_loss" in h and "dev_regret" in h


def test_train_smoke_boundary_magnet(tmp_path: Path) -> None:
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r_magnet.parquet"
    df.to_parquet(pq, index=False)
    out_model = make_router_model_paths_for_cli(
        "t_magnet",
        router_base=tmp_path,
        input_mode="both",
        router_architecture_id="mlp-v1-boundary-magnet",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t_magnet",
        output_dir=out_model.run_root,
        epochs=3,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="both",
        loss="boundary_magnet",
        loss_kwargs={"regret_threshold": 0.05, "magnet_alpha": 0.02},
    )
    result = train_router(cfg)
    assert result.loss_effective == "boundary_magnet"
    assert not result.loss_fallback


def test_train_smoke_logreg_v1(tmp_path: Path) -> None:
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r5.parquet"
    df.to_parquet(pq, index=False)
    out_model = make_router_model_paths_for_cli(
        "t5",
        router_base=tmp_path,
        input_mode="embedding",
        router_architecture_id="logreg-v1-baseline",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t5",
        output_dir=out_model.run_root,
        epochs=2,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="logreg-v1",
        architecture_kwargs={},
        input_mode="embedding",
    )
    result = train_router(cfg)
    save_checkpoint(
        out_model.checkpoint,
        result.model,
        result.model.config,
        architecture="logreg-v1",
        architecture_kwargs={},
    )
    assert out_model.checkpoint.is_file()


def test_train_smoke_tower_v01(tmp_path: Path) -> None:
    """tower_v01 with shrunk kwargs so 4-d synthetic embeddings fit the tower."""
    rows = [_row("a", "train"), _row("b", "train"), _row("c", "dev")]
    df = pd.DataFrame(rows)
    pq = tmp_path / "r6.parquet"
    df.to_parquet(pq, index=False)
    out_model = make_router_model_paths_for_cli(
        "t6",
        router_base=tmp_path,
        input_mode="both",
        router_architecture_id="tower-v01-smoke",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="t6",
        output_dir=out_model.run_root,
        epochs=2,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="tower_v01",
        architecture_kwargs={
            "embed_dims": [3, 3, 3],
            "feat_hidden": 8,
            "dropout": 0.0,
        },
        input_mode="both",
    )
    result = train_router(cfg)
    save_checkpoint(
        out_model.checkpoint,
        result.model,
        result.model.config,
        architecture="tower_v01",
        architecture_kwargs=cfg.architecture_kwargs or {},
    )
    assert out_model.checkpoint.is_file()
    assert result.model.config.feature_dim == 14
    assert result.model.config.embedding_dim == 4


def _train_row_for_midpoint_group(qid: str, split: str, group_index: int) -> dict:
    w = [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
    c = [0.0] * len(w)
    if group_index == 0:
        c[0] = 1.0
    elif group_index == 1:
        c[3] = 1.0
    elif group_index == 2:
        c = [1.0] * len(w)
    elif group_index == 3:
        c[7] = 1.0
    else:
        c[10] = 1.0
    row = _row(qid, split)
    row["oracle_curve"] = c
    row["weight_grid"] = w
    return row


def test_train_smoke_midpoint_balance_masking(tmp_path: Path) -> None:
    """Two train rows per midpoint group (10 train); mask to min=2 per group."""
    rows: list[dict] = []
    for g in range(5):
        rows.append(_train_row_for_midpoint_group(f"t{g}a", "train", g))
        rows.append(_train_row_for_midpoint_group(f"t{g}b", "train", g))
    rows.append(_row("dev1", "dev"))
    df = pd.DataFrame(rows)
    pq = tmp_path / "r_mid.parquet"
    df.to_parquet(pq, index=False)
    mdir = tmp_path / "d_mid"
    mdir.mkdir()
    dpaths = RouterDatasetPaths(run_root=mdir)
    dpaths.ensure_dirs()
    write_router_dataset_manifest(
        dpaths,
        router_id="tmid",
        source_benchmark_name="m",
        source_benchmark_id="v",
        benchmark_path="/b.jsonl",
        retrieval_asset_dir="/c",
        oracle_run_root="/o",
        router_labels_path="/l",
        feature_set_version="1",
        embedding_model="m",
        split_seed=0,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    out_model = make_router_model_paths_for_cli(
        "tmid",
        router_base=tmp_path,
        input_mode="both",
        router_architecture_id="mlp-v1-midmask",
    )
    out_model.ensure_dirs()
    cfg = RouterTrainConfig(
        parquet_path=pq,
        router_id="tmid",
        output_dir=out_model.run_root,
        epochs=2,
        batch_size=2,
        early_stopping_patience=100,
        device="cpu",
        architecture="mlp-v1",
        architecture_kwargs={},
        input_mode="both",
        midpoint_balance_masking=True,
        midpoint_balance_epsilon=1e-6,
        seed=42,
    )
    result = train_router(cfg)
    rep = result.midpoint_balance_report
    assert rep is not None
    assert rep["enabled"] is True
    assert rep["train_rows_before"] == 10
    assert rep["train_rows_after"] == 10
    assert rep["target_per_group"] == 2
    assert rep["counts_after_per_group"] == [2, 2, 2, 2, 2]
