from __future__ import annotations

import json
from pathlib import Path

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.tables.pairwise_wilcoxon import build_pairwise_wilcoxon


def _metrics_row(qid: str, ndcg: float, *, recall: float = 0.5) -> dict:
    return {
        "question_id": qid,
        "retrieval_before_ce": {
            "retrieval": {
                "10": {"ndcg": ndcg, "hit": 1.0, "recall": recall},
            }
        },
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
    router = tmp_path / "router" / "rid"
    oracle_dir = router / "oracle"
    ds_dir = router / "dataset"
    (bench / "benchmark").mkdir(parents=True, exist_ok=True)
    (bench / "benchmark" / "benchmark.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"question_id": "q1", "dataset_source": "nq"},
                {"question_id": "q2", "dataset_source": "2wiki"},
                {"question_id": "q3", "dataset_source": "nq"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps({"train": [], "dev": [], "test": ["q1", "q2", "q3"]}),
        encoding="utf-8",
    )
    (oracle_dir / "oracle_scores.jsonl").write_text(
        '{"question_id":"q1"}\n', encoding="utf-8"
    )
    (oracle_dir / "retrieval_graph.jsonl").write_text(
        '{"question_id":"q1"}\n', encoding="utf-8"
    )

    policies = {
        "learned-soft": [0.9, 0.5, 0.8],
        "rrf": [0.7, 0.6, 0.7],
        "hard-routing": [0.85, 0.4, 0.75],
        "50-50": [0.8, 0.55, 0.72],
    }
    for policy, ndcgs in policies.items():
        run_dir = bench / "evaluations" / policy / "run1"
        run_dir.mkdir(parents=True)
        rows = [
            _metrics_row(qid, v)
            for qid, v in zip(["q1", "q2", "q3"], ndcgs, strict=True)
        ]
        (run_dir / "metrics.json").write_text(
            json.dumps({"per_question": rows}), encoding="utf-8"
        )

    cfg_path = tmp_path / "results.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_base": str(tmp_path),
                    "benchmark_base": str(tmp_path / "benchmarks"),
                    "router_base": str(tmp_path / "router"),
                    "benchmark_name": "bench",
                    "benchmark_id": "v1",
                    "router_id": "rid",
                },
                "results": {
                    "bundle_id": "t",
                    "output_root": str(tmp_path / "out"),
                    "split": "test",
                    "policies": {p: {"run_id": "run1"} for p in policies},
                    "oracle": {"metric": "stateful_ndcg", "k": 10},
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path, tmp_path


def test_pairwise_wilcoxon_three_comparisons(tmp_path: Path) -> None:
    cfg_path, _ = _write_fixture(tmp_path)
    cfg = load_pipeline_config(cfg_path)
    bundle = ResultsBundle.from_config(cfg)
    spec = ResultsArtifactSpec(id="pairwise_wilcoxon", kind="table", k=10)
    df, _ = build_pairwise_wilcoxon(bundle, spec)

    assert set(df["comparison_policy"]) == {"rrf", "hard-routing", "50-50"}
    assert (df["baseline_policy"] == "learned-soft").all()
    all_rows = df[df["dataset_source"] == "all"]
    assert len(all_rows) == 3
    rrf_row = all_rows[all_rows["comparison_policy"] == "rrf"].iloc[0]
    assert rrf_row["n"] == 3
    assert (
        float(rrf_row["mean_difference"])
        == (0.9 + 0.5 + 0.8) / 3 - (0.7 + 0.6 + 0.7) / 3
    )
    assert float(rrf_row["p_value"]) >= 0.0
    assert float(rrf_row["p_value"]) <= 1.0
    assert int(rrf_row["n_ndcg_win"]) == 2
    assert int(rrf_row["n_ndcg_loss"]) == 1
    assert int(rrf_row["n_ndcg_tie"]) == 0
    assert float(rrf_row["pct_ndcg_win"]) == 100.0 * 2 / 3
    assert int(rrf_row["n_recall_perfect_baseline_only"]) == 0
    assert int(rrf_row["n_recall_perfect_comparison_only"]) == 0
    assert int(rrf_row["n_recall_perfect_both"]) == 0
    assert int(rrf_row["n_recall_perfect_neither"]) == 3


def test_pairwise_wilcoxon_recall_coverage(tmp_path: Path) -> None:
    cfg_path, root = _write_fixture(tmp_path)
    bench = root / "benchmarks" / "bench" / "v1" / "evaluations"
    ls_rows = [
        _metrics_row("q1", 0.5, recall=1.0),
        _metrics_row("q2", 0.5, recall=0.5),
        _metrics_row("q3", 0.5, recall=1.0),
    ]
    rrf_rows = [
        _metrics_row("q1", 0.5, recall=0.5),
        _metrics_row("q2", 0.5, recall=1.0),
        _metrics_row("q3", 0.5, recall=0.5),
    ]
    (bench / "learned-soft" / "run1" / "metrics.json").write_text(
        json.dumps({"per_question": ls_rows}), encoding="utf-8"
    )
    (bench / "rrf" / "run1" / "metrics.json").write_text(
        json.dumps({"per_question": rrf_rows}), encoding="utf-8"
    )

    cfg = load_pipeline_config(cfg_path)
    bundle = ResultsBundle.from_config(cfg)
    spec = ResultsArtifactSpec(id="pairwise_wilcoxon", kind="table", k=10)
    df, _ = build_pairwise_wilcoxon(bundle, spec)

    rrf_row = df[
        (df["dataset_source"] == "all") & (df["comparison_policy"] == "rrf")
    ].iloc[0]
    assert int(rrf_row["n_recall_perfect_baseline_only"]) == 2
    assert int(rrf_row["n_recall_perfect_comparison_only"]) == 1
    assert int(rrf_row["n_recall_perfect_both"]) == 0
    assert int(rrf_row["n_recall_perfect_neither"]) == 0
