from __future__ import annotations

import json
from pathlib import Path

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.tables.oracle_policy_comparison import (
    ORACLE_CLASSIFICATION,
    ORACLE_UPPER_BOUND,
    build_oracle_policy_comparison,
)
from surf_rag.results.tables.pipeline_answers import build_pipeline_answers


def _metrics_row(qid: str, ndcg: float, hit: float, recall: float) -> dict:
    return {
        "question_id": qid,
        "retrieval_before_ce": {
            "retrieval": {
                "10": {"ndcg": ndcg, "hit": hit, "recall": recall},
            }
        },
    }


def _write_fixture(tmp_path: Path) -> Path:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
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
    router = tmp_path / "router" / "rid"
    ds_dir = router / "dataset"
    oracle_dir = router / "oracle"
    ds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps({"train": [], "dev": [], "test": ["q1", "q2", "q3"]}),
        encoding="utf-8",
    )
    (oracle_dir / "oracle_scores.jsonl").write_text('{"question_id":"q1"}\n')
    (oracle_dir / "retrieval_graph.jsonl").write_text('{"question_id":"q1"}\n')

    oracle_scores = {
        ORACLE_UPPER_BOUND: [(0.9, 1.0, 0.8), (0.8, 1.0, 0.7), (0.85, 1.0, 0.75)],
        ORACLE_CLASSIFICATION: [(0.7, 0.8, 0.6), (0.75, 0.9, 0.65), (0.8, 1.0, 0.7)],
    }
    policies = {
        "dense-only": [(0.5, 0.5, 0.5)] * 3,
        ORACLE_UPPER_BOUND: oracle_scores[ORACLE_UPPER_BOUND],
        ORACLE_CLASSIFICATION: oracle_scores[ORACLE_CLASSIFICATION],
    }
    for policy, triples in policies.items():
        run_dir = bench / "evaluations" / policy / "run1"
        run_dir.mkdir(parents=True)
        rows = [
            _metrics_row(qid, ndcg, hit, recall)
            for qid, (ndcg, hit, recall) in zip(
                ["q1", "q2", "q3"], triples, strict=True
            )
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
                    "oracle": {
                        "metric": "stateful_ndcg",
                        "k": 10,
                        "diagnostic_ks": [10],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_oracle_policy_comparison_metrics_and_slices(tmp_path: Path) -> None:
    cfg = load_pipeline_config(_write_fixture(tmp_path))
    bundle = ResultsBundle.from_config(cfg)
    spec = ResultsArtifactSpec(id="oracle_policy_comparison", kind="table")
    df, _ = build_oracle_policy_comparison(bundle, spec)

    assert set(df["metric"]) == {"ndcg", "hit", "recall"}
    assert set(df["dataset_source"]) <= {"all", "nq", "2wiki"}
    all_ndcg = df[(df["dataset_source"] == "all") & (df["metric"] == "ndcg")].iloc[0]
    assert float(all_ndcg["oracle_upper_bound_mean"]) == (0.9 + 0.8 + 0.85) / 3
    assert float(all_ndcg["oracle_classification_mean"]) == (0.7 + 0.75 + 0.8) / 3
    assert float(all_ndcg["mean_difference"]) > 0
    assert float(all_ndcg["p_value"]) >= 0.0


def test_pipeline_answers_excludes_classification_oracle(tmp_path: Path) -> None:
    cfg = load_pipeline_config(_write_fixture(tmp_path))
    bundle = ResultsBundle.from_config(cfg)
    df, _ = build_pipeline_answers(bundle)
    policies = set(df["policy"].unique()) if len(df) else set()
    assert ORACLE_CLASSIFICATION not in policies
