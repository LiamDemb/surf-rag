from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from surf_rag.evaluation.discrepancy_debug import (
    ComparisonCounts,
    SelectionRule,
    assert_reranker_none_or_allow_ce,
    extract_interesting_rows,
    load_answers_by_qid,
    load_benchmark_index,
    load_retrieval_by_qid,
    read_e2e_manifest_block,
    write_discrepancy_bundle,
)
from surf_rag.evaluation.run_artifacts import RunArtifactPaths


def _load_dd_script_module():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "discrepancy_debug_e2e.py"
    )
    spec = importlib.util.spec_from_file_location("_discrepancy_debug_e2e", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _bench_row() -> dict:
    return {
        "question_id": "q1",
        "question": "Who developed the theory of relativity?",
        "gold_answers": ["Einstein"],
        "gold_support_sentences": [
            "Albert Einstein developed the theory of relativity."
        ],
        "dataset_source": "2wikimultihopqa",
    }


def _retrieval_row(*, qid: str, chunks: list[dict]) -> dict:
    return {
        "question_id": qid,
        "query": "ignored",
        "retriever_name": "Fused",
        "status": "OK",
        "chunks": chunks,
        "latency_ms": {},
        "error": None,
        "debug_info": None,
    }


def _chunk(text: str, rank: int, cid: str) -> dict:
    return {
        "chunk_id": cid,
        "text": text,
        "score": 1.0 - rank * 0.01,
        "rank": rank,
        "metadata": {},
    }


def _setup_run(
    tmp: Path,
    *,
    retrieval_chunks: list[dict],
    answer_text: str,
    manifest: dict | None = None,
) -> Path:
    run = RunArtifactPaths(run_root=tmp / "run")
    run.ensure_dirs()
    _write_jsonl(
        run.retrieval_results_jsonl(),
        [_retrieval_row(qid="q1", chunks=retrieval_chunks)],
    )
    row = dict(_bench_row())
    row["answer"] = answer_text
    _write_jsonl(run.generation_answers_jsonl(), [row])
    if manifest is not None:
        run.manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return run.run_root


def test_interesting_when_ndcg_improves_but_f1_flat(tmp_path: Path) -> None:
    bench_p = tmp_path / "bench.jsonl"
    _write_jsonl(bench_p, [_bench_row()])
    bench = load_benchmark_index(bench_p)

    root_a = _setup_run(
        tmp_path,
        retrieval_chunks=[
            _chunk("The moon is mostly rock.", 0, "c0"),
            _chunk("Albert Einstein developed the theory of relativity.", 1, "c1"),
        ],
        answer_text="Moon",
    )
    root_b = _setup_run(
        tmp_path / "b",
        retrieval_chunks=[
            _chunk("Albert Einstein developed the theory of relativity.", 0, "c1"),
            _chunk("The moon is mostly rock.", 1, "c0"),
        ],
        answer_text="Moon",
    )

    retr_a = load_retrieval_by_qid(root_a / "retrieval" / "retrieval_results.jsonl")
    retr_b = load_retrieval_by_qid(root_b / "retrieval" / "retrieval_results.jsonl")
    ans_a = load_answers_by_qid(root_a / "generation" / "answers.jsonl")
    ans_b = load_answers_by_qid(root_b / "generation" / "answers.jsonl")

    rows, counts = extract_interesting_rows(
        bench_by_qid=bench,
        retr_a=retr_a,
        retr_b=retr_b,
        ans_a=ans_a,
        ans_b=ans_b,
        restrict_qids=None,
        rule=SelectionRule(epsilon_ndcg=1e-9, delta_f1=0.0),
        top_k_chunks=10,
        chunk_preview_chars=80,
    )
    assert counts.joined_question_ids == 1
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["delta_ndcg_at_10"] > 1e-6
    assert rows[0]["delta_f1"] == 0.0
    assert rows[0]["run_a"]["prediction"] == rows[0]["run_b"]["prediction"] == "Moon"


def test_assert_reranker_rejects_cross_encoder(tmp_path: Path) -> None:
    good = _setup_run(
        tmp_path / "g",
        retrieval_chunks=[_chunk("x", 0, "a")],
        answer_text="y",
        manifest={"e2e": {"reranker": "none"}},
    )
    bad = _setup_run(
        tmp_path / "h",
        retrieval_chunks=[_chunk("x", 0, "a")],
        answer_text="y",
        manifest={"e2e": {"reranker": "cross_encoder", "rerank_top_k": 10}},
    )
    assert_reranker_none_or_allow_ce(good, good, allow_cross_encoder=False)
    with pytest.raises(ValueError, match="reranker"):
        assert_reranker_none_or_allow_ce(good, bad, allow_cross_encoder=False)
    assert_reranker_none_or_allow_ce(good, bad, allow_cross_encoder=True)


def test_write_bundle_manifest_schema(tmp_path: Path) -> None:
    root_a = _setup_run(
        tmp_path / "wa",
        retrieval_chunks=[_chunk("x", 0, "a")],
        answer_text="z",
        manifest={"e2e": {"router_id": "r1", "benchmark_id": "main"}},
    )
    root_b = _setup_run(
        tmp_path / "wb",
        retrieval_chunks=[_chunk("x", 0, "a")],
        answer_text="z",
        manifest={"e2e": {"router_id": "r1", "benchmark_id": "main"}},
    )
    out = tmp_path / "out" / "t1"
    rule = SelectionRule(epsilon_ndcg=1e-9, delta_f1=0.0)
    interesting: list[dict] = [
        {
            "question_id": "q1",
            "dataset_source": "2wikimultihopqa",
            "question": "Q?",
            "gold_answers": ["a"],
            "run_a": {
                "ndcg_at_10": 0.0,
                "f1": 0.0,
                "em": 0.0,
                "prediction": "",
                "retrieval_status": "OK",
                "top_chunks": [],
            },
            "run_b": {
                "ndcg_at_10": 0.5,
                "f1": 0.0,
                "em": 0.0,
                "prediction": "",
                "retrieval_status": "OK",
                "top_chunks": [],
            },
            "delta_ndcg_at_10": 0.5,
            "delta_f1": 0.0,
            "extras": {"top10_chunk_id_jaccard": 1.0},
        }
    ]
    counts = ComparisonCounts(
        joined_question_ids=1,
        interesting=1,
        skipped_missing_retrieval_a=0,
        skipped_missing_retrieval_b=0,
        skipped_missing_answer_a=0,
        skipped_missing_answer_b=0,
        skipped_restrict_filter=0,
    )
    write_discrepancy_bundle(
        out,
        test_id="t1",
        benchmark_path=tmp_path / "bench.jsonl",
        run_root_a=root_a,
        run_root_b=root_b,
        e2e_a=read_e2e_manifest_block(root_a),
        e2e_b=read_e2e_manifest_block(root_b),
        rule=rule,
        interesting_rows=interesting,
        counts=counts,
        markdown_max_rows=10,
    )
    mf = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert mf["schema_version"] == "surf-rag/discrepancy-debug/v1"
    assert mf["test_id"] == "t1"
    assert "selection_rule" in mf
    assert mf["counts"]["interesting"] == 1
    assert (out / "interesting.jsonl").is_file()
    assert (out / "interesting.md").is_file()


def test_cli_smoke_invocation(tmp_path: Path) -> None:
    bench_p = tmp_path / "bench.jsonl"
    _write_jsonl(bench_p, [_bench_row()])
    root_a = _setup_run(
        tmp_path / "ca",
        retrieval_chunks=[
            _chunk("The moon is mostly rock.", 0, "c0"),
            _chunk("Albert Einstein developed the theory of relativity.", 1, "c1"),
        ],
        answer_text="Moon",
        manifest={"e2e": {"reranker": "none"}},
    )
    root_b = _setup_run(
        tmp_path / "cb",
        retrieval_chunks=[
            _chunk("Albert Einstein developed the theory of relativity.", 0, "c1"),
            _chunk("The moon is mostly rock.", 1, "c0"),
        ],
        answer_text="Moon",
        manifest={"e2e": {"reranker": "none"}},
    )
    out_root = tmp_path / "reports"
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "discrepancy_debug_e2e.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--benchmark-path",
            str(bench_p),
            "--run-root-a",
            str(root_a),
            "--run-root-b",
            str(root_b),
            "--test-id",
            "99",
            "--output-root",
            str(out_root),
            "-q",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wrote = Path(proc.stdout.strip())
    assert wrote == out_root / "99"
    mf = json.loads((wrote / "manifest.json").read_text(encoding="utf-8"))
    assert mf["counts"]["interesting"] == 1
    cmp_res = mf.get("inputs", {}).get("comparison_resolution")
    assert cmp_res == {"mode": "explicit_run_roots"}


def test_yaml_config_derives_benchmark_and_test_id(monkeypatch, tmp_path) -> None:
    mod = _load_dd_script_module()
    bb = tmp_path / "benchmarks" / "surf-bench" / "main"
    bm = bb / "benchmark"
    bm.mkdir(parents=True)
    bp = bm / "benchmark.jsonl"
    bp.write_text(
        '{"question_id":"q1","question":"","gold_answers":[]}\n', encoding="utf-8"
    )
    yaml_path = tmp_path / "d.yaml"
    base_repr = json.dumps(str(tmp_path / "benchmarks"))
    yaml_path.write_text(
        "\n".join(
            [
                "benchmark_path: null",
                f"benchmark_base: {base_repr}",
                "benchmark_name: surf-bench",
                "benchmark_id: main",
                "policy: learned-soft",
                "run_id_a: r1",
                "run_id_b: r2",
                "test_id: '042'",
                "run_root_a: null",
                "run_root_b: null",
                "quiet: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(yaml_path)])
    args = mod.build_parser().parse_args()
    mod.merge_discrepancy_yaml_into_args(
        args, mod._load_yaml_mapping(Path(args.config).expanduser().resolve())
    )
    mod.finalize_benchmark_and_requirements(args)
    assert args.benchmark_path.resolve() == bp.resolve()
    assert args.test_id == "042"


def test_cli_overrides_yaml_test_id(monkeypatch, tmp_path) -> None:
    mod = _load_dd_script_module()
    bp = tmp_path / "b.jsonl"
    bp.write_text("{}", encoding="utf-8")
    yaml_path = tmp_path / "d2.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"benchmark_path: {json.dumps(str(bp))}",
                "test_id: '001'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", "--config", str(yaml_path), "--test-id", "88"]
    )
    args = mod.build_parser().parse_args()
    mod.merge_discrepancy_yaml_into_args(
        args, mod._load_yaml_mapping(Path(args.config).expanduser().resolve())
    )
    mod.finalize_benchmark_and_requirements(args)
    assert args.test_id == "88"


def test_resolve_run_roots_uses_distinct_policies(tmp_path) -> None:
    mod = _load_dd_script_module()
    bb = tmp_path / "bm"
    args = argparse.Namespace(
        run_root_a=None,
        run_root_b=None,
        benchmark_base=bb,
        benchmark_name="surf-bench",
        benchmark_id="main",
        policy=None,
        policy_a="dense-only",
        policy_b="learned-soft",
        run_id_a="r1",
        run_id_b="r2",
    )
    root_a, root_b = mod._resolve_run_roots(args)
    sa = root_a.as_posix()
    sb = root_b.as_posix()
    assert "evaluations/dense-only/r1" in sa or "/dense-only/r1" in sa
    assert "evaluations/learned-soft/r2" in sb or "/learned-soft/r2" in sb
    meta = mod._comparison_resolution_meta(args)
    assert meta["mode"] == "evaluations_layout"
    assert meta["run_a"]["routing_policy"] == "dense-only"
    assert meta["run_b"]["routing_policy"] == "learned-soft"
    assert meta["same_policy_folder"] is False


def test_yaml_nested_discrepancy_debug_block(tmp_path) -> None:
    mod = _load_dd_script_module()
    p = tmp_path / "nested.yaml"
    p.write_text(
        "other_keys: 1\n"
        "discrepancy_debug:\n"
        "  test_id: '3'\n"
        "  epsilon_ndcg: 0.01\n",
        encoding="utf-8",
    )
    data = mod._load_yaml_mapping(p)
    assert data["test_id"] == "3"
    assert data["epsilon_ndcg"] == 0.01
