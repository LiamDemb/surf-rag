from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import math

from surf_rag.config.schema import (
    PipelineConfig,
    ResultsArtifactSpec,
    ResultsOracleConfig,
    ResultsPolicyEntry,
    ResultsSection,
)
from surf_rag.results.bundle import PolicyRun, ResultsBundle
from surf_rag.results.tables.pipeline_retrieval_perfect_rates import (
    build_pipeline_retrieval_perfect_rates,
)


def _metrics_json(tmp_path: Path) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(
        """{
  "per_question": [
    {
      "question_id": "q1",
      "retrieval_before_ce": {
        "retrieval": {
          "10": {"hit": 1.0, "recall": 1.0, "ndcg": 1.0}
        }
      },
      "qa_llm_judge": {"correct": true}
    },
    {
      "question_id": "q2",
      "retrieval_before_ce": {
        "retrieval": {
          "10": {"hit": 1.0, "recall": 0.5, "ndcg": 0.7}
        }
      },
      "qa_llm_judge": {"correct": false}
    }
  ]
}""",
        encoding="utf-8",
    )
    return path


def _minimal_bundle(tmp_path: Path, metrics_path: Path) -> ResultsBundle:
    results = ResultsSection(
        bundle_id="t",
        output_root=str(tmp_path / "out"),
        split="test",
        oracle=ResultsOracleConfig(metric="stateful_ndcg", k=10, diagnostic_ks=[10]),
        policies={"dense-only": ResultsPolicyEntry(run_id="r1")},
    )
    cfg = replace(PipelineConfig(), results=results)
    bundle = ResultsBundle(
        cfg=cfg,
        resolved=None,  # type: ignore[arg-type]
        results=results,
        output_dir=tmp_path / "out" / "t",
        split_qids={"q1", "q2"},
        qid_to_source={"q1": "nq", "q2": "2wiki"},
        policies={
            "dense-only": PolicyRun(
                policy="dense-only",
                run_id="r1",
                router_role=None,
                metrics_path=metrics_path,
                run_dir=metrics_path.parent,
                resolved_config_path=None,
            )
        },
        oracle_scores_path=tmp_path / "oracle.jsonl",
        split_question_ids_path=tmp_path / "split.json",
        retrieval_graph_path=tmp_path / "graph.jsonl",
        answerability_path=tmp_path / "ans.jsonl",
        image_format="png",
    )
    bundle.output_dir.mkdir(parents=True, exist_ok=True)
    return bundle


def test_pipeline_retrieval_perfect_rates(tmp_path: Path) -> None:
    bundle = _minimal_bundle(tmp_path, _metrics_json(tmp_path))
    spec = ResultsArtifactSpec(
        id="pipeline_retrieval_perfect_rates",
        kind="table",
        k=10,
    )
    df, _ = build_pipeline_retrieval_perfect_rates(bundle, spec)
    all_row = df.loc[df["dataset_source"] == "all"].iloc[0]
    assert int(all_row["n"]) == 2
    assert int(all_row["n_perfect_ndcg"]) == 1
    assert float(all_row["pct_perfect_ndcg"]) == 50.0
    assert int(all_row["n_qa_perfect_ndcg"]) == 1
    assert float(all_row["qa_accuracy_perfect_ndcg"]) == 1.0
    assert float(all_row["pct_qa_accuracy_perfect_ndcg"]) == 100.0
    assert int(all_row["n_perfect_recall"]) == 1
    assert float(all_row["pct_perfect_recall"]) == 50.0
    assert int(all_row["n_qa_perfect_recall"]) == 1
    assert float(all_row["qa_accuracy_perfect_recall"]) == 1.0

    nq = df.loc[df["dataset_source"] == "nq"].iloc[0]
    assert float(nq["pct_perfect_ndcg"]) == 100.0
    assert float(nq["pct_perfect_recall"]) == 100.0
    assert float(nq["qa_accuracy_perfect_ndcg"]) == 1.0


def test_pipeline_retrieval_perfect_rates_no_judge(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        """{
  "per_question": [
    {
      "question_id": "q1",
      "retrieval_before_ce": {
        "retrieval": {"10": {"hit": 1.0, "recall": 1.0, "ndcg": 1.0}}
      }
    }
  ]
}""",
        encoding="utf-8",
    )
    bundle = _minimal_bundle(tmp_path, path)
    bundle.split_qids = {"q1"}
    df, _ = build_pipeline_retrieval_perfect_rates(
        bundle,
        ResultsArtifactSpec(id="pipeline_retrieval_perfect_rates", kind="table", k=10),
    )
    row = df.iloc[0]
    assert int(row["n_perfect_ndcg"]) == 1
    assert int(row["n_qa_perfect_ndcg"]) == 0
    assert math.isnan(row["qa_accuracy_perfect_ndcg"])
