from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from surf_rag.config.schema import (
    PipelineConfig,
    ResultsArtifactSpec,
    ResultsOracleConfig,
    ResultsPolicyEntry,
    ResultsSection,
)
from surf_rag.results.bundle import PolicyRun, ResultsBundle
from surf_rag.results.figures.pipeline_qa_calibration import (
    _N_BINS,
    _assign_score_bin,
    render_pipeline_qa_calibration,
    render_pipeline_qa_calibration_recall,
)
from surf_rag.viz.theme import apply_theme


def test_assign_score_bin_edges() -> None:
    assert _N_BINS == 5
    assert _assign_score_bin(0.0) == 0
    assert _assign_score_bin(0.1) == 1
    assert _assign_score_bin(0.25) == 1
    assert _assign_score_bin(0.26) == 2
    assert _assign_score_bin(0.5) == 2
    assert _assign_score_bin(0.9) == 3
    assert _assign_score_bin(1.0) == 4


def _metrics_json(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"per_question": rows}),
        encoding="utf-8",
    )


def _bundle(tmp_path: Path) -> ResultsBundle:
    out = tmp_path / "out" / "t"
    out.mkdir(parents=True, exist_ok=True)
    results = ResultsSection(
        bundle_id="t",
        output_root=str(tmp_path / "out"),
        split="test",
        oracle=ResultsOracleConfig(metric="stateful_ndcg", k=10),
        policies={
            "dense-only": ResultsPolicyEntry(run_id="r1"),
            "learned-soft": ResultsPolicyEntry(run_id="r1"),
            "oracle-upper-bound": ResultsPolicyEntry(run_id="r1"),
        },
    )
    cfg = replace(PipelineConfig(), results=results)
    policies: dict[str, PolicyRun] = {}
    row_nq = {
        "question_id": "q_nq",
        "retrieval_before_ce": {
            "retrieval": {"10": {"ndcg": 0.0, "hit": 0.0, "recall": 0.0}}
        },
        "qa_llm_judge": {"correct": False},
    }
    row_2wiki = {
        "question_id": "q_2wiki",
        "retrieval_before_ce": {
            "retrieval": {"10": {"ndcg": 1.0, "hit": 1.0, "recall": 1.0}}
        },
        "qa_llm_judge": {"correct": True},
    }
    for name in ("dense-only", "learned-soft", "oracle-upper-bound"):
        mp = tmp_path / name / "metrics.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        _metrics_json(mp, [row_nq, row_2wiki])
        policies[name] = PolicyRun(
            policy=name,
            run_id="r1",
            router_role=None,
            metrics_path=mp,
            run_dir=mp.parent,
            resolved_config_path=None,
        )
    return ResultsBundle(
        cfg=cfg,
        resolved=None,  # type: ignore[arg-type]
        results=results,
        output_dir=out,
        split_qids={"q_nq", "q_2wiki"},
        qid_to_source={"q_nq": "nq", "q_2wiki": "2wiki"},
        policies=policies,
        oracle_scores_path=tmp_path / "oracle.jsonl",
        split_question_ids_path=tmp_path / "split.json",
        retrieval_graph_path=tmp_path / "graph.jsonl",
        answerability_path=tmp_path / "ans.jsonl",
        image_format="png",
    )


def test_pipeline_qa_calibration_renders(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    bundle = _bundle(tmp_path)
    paths = render_pipeline_qa_calibration(
        bundle,
        ResultsArtifactSpec(
            id="pipeline_qa_calibration",
            kind="figure",
            figure="pipeline_qa_calibration",
        ),
    )
    assert Path(paths["image"]).is_file()
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    assert meta["dataset_source"] == "all"
    assert meta["highlight_policy"] == "learned-soft"
    assert len(meta["series"]) == 3
    assert meta["metric"] in ("stateful_ndcg", "ndcg")


def test_pipeline_qa_calibration_recall_renders(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    bundle = _bundle(tmp_path)
    for mp in bundle.policies.values():
        data = json.loads(mp.metrics_path.read_text(encoding="utf-8"))
        for row in data["per_question"]:
            block = row["retrieval_before_ce"]["retrieval"]["10"]
            block["recall"] = block.pop("ndcg", 0.0)
        mp.metrics_path.write_text(json.dumps(data), encoding="utf-8")

    paths = render_pipeline_qa_calibration_recall(
        bundle,
        ResultsArtifactSpec(
            id="pipeline_qa_calibration_recall",
            kind="figure",
            figure="pipeline_qa_calibration_recall",
            metric="recall",
            k=10,
        ),
    )
    assert Path(paths["image"]).name == "pipeline_qa_calibration_recall.png"
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    assert meta["metric"] == "recall"
    assert meta["k"] == 10
