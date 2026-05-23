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
from surf_rag.results.figures.pipeline_retrieval_qa_scatter import (
    render_pipeline_retrieval_qa_scatter,
)
from surf_rag.viz.theme import apply_theme


def _metrics_json(path: Path, *, ndcg: float, correct: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "per_question": [
                    {
                        "question_id": "q1",
                        "retrieval_before_ce": {
                            "retrieval": {
                                "10": {"hit": 1.0, "recall": 0.8, "ndcg": ndcg},
                            }
                        },
                        "qa_llm_judge": {"correct": correct},
                    }
                ]
            }
        ),
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
            "oracle-upper-bound": ResultsPolicyEntry(run_id="r1"),
        },
    )
    cfg = replace(PipelineConfig(), results=results)
    policies: dict[str, PolicyRun] = {}
    for name, ndcg, correct in (
        ("dense-only", 0.5, False),
        ("oracle-upper-bound", 0.9, True),
    ):
        mp = tmp_path / name / "metrics.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        _metrics_json(mp, ndcg=ndcg, correct=correct)
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
        split_qids={"q1"},
        qid_to_source={"q1": "nq"},
        policies=policies,
        oracle_scores_path=tmp_path / "oracle.jsonl",
        split_question_ids_path=tmp_path / "split.json",
        retrieval_graph_path=tmp_path / "graph.jsonl",
        answerability_path=tmp_path / "ans.jsonl",
        image_format="png",
    )


def test_pipeline_retrieval_qa_scatter_renders(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    bundle = _bundle(tmp_path)
    paths = render_pipeline_retrieval_qa_scatter(
        bundle,
        ResultsArtifactSpec(
            id="pipeline_retrieval_qa_scatter",
            kind="figure",
            figure="pipeline_retrieval_qa_scatter",
        ),
    )
    assert Path(paths["image"]).is_file()
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    assert len(meta["policies"]) == 2
    oracle = next(p for p in meta["policies"] if p["policy"] == "oracle-upper-bound")
    assert oracle["ndcg"] == pytest.approx(0.9)
    assert oracle["accuracy"] == pytest.approx(1.0)
