from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from surf_rag.config.schema import (
    PipelineConfig,
    ResultsArtifactSpec,
    ResultsOracleConfig,
    ResultsPolicyEntry,
    ResultsSection,
)
from surf_rag.evaluation.latency_metrics import bootstrap_mean_ci95
from surf_rag.results.bundle import PolicyRun, ResultsBundle
from surf_rag.results.figures.retrieval_by_policy_ci import (
    _DEFAULT_YLIM_MAX,
    _DEFAULT_YLIM_MIN,
    _mean_and_ci95,
    _resolve_ylim,
    render_retrieval_by_policy_ci,
)
from surf_rag.viz.theme import BAR_ALPHA, PALETTE, bar_style


def test_bar_style_uses_light_blue_and_alpha() -> None:
    style = bar_style()
    assert style["color"] == PALETTE["light-blue"]
    assert style["alpha"] == BAR_ALPHA
    assert style["edgecolor"] == "none"
    assert style["linewidth"] == 0


def test_mean_and_ci95_bootstrap_on_questions() -> None:
    vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean, lo, hi, n = _mean_and_ci95(vals)
    assert n == 5
    assert lo <= mean <= hi
    boot_lo, boot_hi = bootstrap_mean_ci95(vals, samples=5000, seed=42)
    assert abs(lo - boot_lo) < 1e-9
    assert abs(hi - boot_hi) < 1e-9


def test_resolve_ylim_defaults() -> None:
    spec = ResultsArtifactSpec(id="x", kind="figure")
    assert _resolve_ylim(spec) == (_DEFAULT_YLIM_MIN, _DEFAULT_YLIM_MAX)


def test_resolve_ylim_override() -> None:
    spec = ResultsArtifactSpec(id="x", kind="figure", ylim_min=0.4, ylim_max=0.95)
    assert _resolve_ylim(spec) == (0.4, 0.95)


def test_render_retrieval_by_policy_ci(tmp_path: Path) -> None:
    metrics = {
        "per_question": [
            {
                "question_id": "q1",
                "retrieval_before_ce": {
                    "retrieval": {"10": {"ndcg": 0.9, "hit": 1.0, "recall": 0.8}}
                },
            },
            {
                "question_id": "q2",
                "retrieval_before_ce": {
                    "retrieval": {"10": {"ndcg": 0.7, "hit": 1.0, "recall": 0.6}}
                },
            },
        ]
    }
    mpath = tmp_path / "metrics.json"
    mpath.write_text(__import__("json").dumps(metrics), encoding="utf-8")

    results = ResultsSection(
        bundle_id="t",
        output_root=str(tmp_path / "out"),
        split="test",
        oracle=ResultsOracleConfig(metric="ndcg", k=10),
        policies={"dense-only": ResultsPolicyEntry(run_id="r1")},
    )
    bundle = ResultsBundle(
        cfg=replace(PipelineConfig(), results=results),
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
                metrics_path=mpath,
                run_dir=mpath.parent,
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
    spec = ResultsArtifactSpec(id="retrieval_by_policy_ci", kind="figure")
    paths = render_retrieval_by_policy_ci(bundle, spec)
    assert Path(paths["image"]).is_file()
    assert Path(paths["meta"]).is_file()
