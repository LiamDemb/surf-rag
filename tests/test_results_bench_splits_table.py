from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.answerability_layout import answerability_mask_path
from surf_rag.evaluation.answerability_types import build_mask_document
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.tables.bench_splits import build_bench_splits


def _minimal_cfg(tmp_path: Path) -> tuple:
    bench_root = tmp_path / "benchmarks" / "bench" / "v1"
    bench_jsonl = bench_root / "benchmark" / "benchmark.jsonl"
    bench_jsonl.parent.mkdir(parents=True, exist_ok=True)
    bench_jsonl.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"question_id": "n1", "dataset_source": "nq"},
                {"question_id": "n2", "dataset_source": "nq"},
                {"question_id": "w1", "dataset_source": "2wiki"},
                {"question_id": "w2", "dataset_source": "2wiki"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    router = tmp_path / "router" / "rid"
    ds_dir = router / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps(
            {
                "train": ["n1", "w1"],
                "dev": ["n2"],
                "test": ["w2"],
            }
        ),
        encoding="utf-8",
    )
    oracle_dir = router / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (oracle_dir / "oracle_scores.jsonl").write_text(
        '{"question_id":"n1"}\n', encoding="utf-8"
    )
    (oracle_dir / "retrieval_graph.jsonl").write_text(
        '{"question_id":"n1"}\n', encoding="utf-8"
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
paths:
  data_base: {tmp_path}
  benchmark_base: {tmp_path / "benchmarks"}
  router_base: {tmp_path / "router"}
  benchmark_name: bench
  benchmark_id: v1
  router_id: rid
results:
  bundle_id: t
  output_root: {tmp_path / "out"}
  split: all
  policies:
    dense-only:
      run_id: run1
""",
        encoding="utf-8",
    )
    run_dir = bench_root / "evaluations" / "dense-only" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"per_question":[]}', encoding="utf-8")
    cfg = load_pipeline_config(cfg_path)
    return cfg, cfg_path


def test_bench_splits_e2e_and_orchestrator_columns(tmp_path: Path) -> None:
    cfg, _ = _minimal_cfg(tmp_path)
    bundle = ResultsBundle.from_config(cfg)
    mask_doc = build_mask_document(
        audit_entries=[{"question_id": "n2", "reason": "audit"}],
        balance_entries=[],
    )
    mp = answerability_mask_path(bundle.resolved.benchmark_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(mask_doc), encoding="utf-8")

    df, _ = build_bench_splits(bundle)

    assert list(df.columns) == ["dataset_source", "e2e_split", "orchestrator_split"]
    by_src = {r["dataset_source"]: r for _, r in df.iterrows()}
    assert by_src["nq"]["e2e_split"] == "1/1/0"
    assert by_src["nq"]["orchestrator_split"] == "1/0/0"
    assert by_src["2wiki"]["e2e_split"] == "1/0/1"
    assert by_src["2wiki"]["orchestrator_split"] == "1/0/1"
    assert by_src["total"]["e2e_split"] == "2/1/1"
    assert by_src["total"]["orchestrator_split"] == "2/0/1"


def test_bench_splits_without_mask_matches_e2e(tmp_path: Path) -> None:
    cfg, _ = _minimal_cfg(tmp_path)
    bundle = ResultsBundle.from_config(cfg)
    df, _ = build_bench_splits(bundle)
    for _, row in df.iterrows():
        assert row["e2e_split"] == row["orchestrator_split"]
    assert any("mask.json not found" in w for w in bundle.warnings)


def test_bench_splits_writes_csv(tmp_path: Path) -> None:
    cfg, cfg_path = _minimal_cfg(tmp_path)
    bundle = ResultsBundle.from_config(cfg, config_path=cfg_path)
    build_bench_splits(bundle)
    csv_path = bundle.output_dir / "tables" / "bench_splits.csv"
    assert csv_path.is_file()
    loaded = pd.read_csv(csv_path)
    assert "orchestrator_split" in loaded.columns
