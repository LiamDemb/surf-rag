"""Run artifact paths and batch custom_id encoding."""

from pathlib import Path

from surf_rag.evaluation.run_artifacts import (
    RunArtifactPaths,
    build_run_root,
    default_evaluation_base,
    make_generation_custom_id,
    parse_generation_custom_id,
)


def test_custom_id_roundtrip():
    cid = make_generation_custom_id("r1", "nq", "dev", "dense", "q-42")
    meta = parse_generation_custom_id(cid)
    assert meta is not None
    assert meta["run_id"] == "r1"
    assert meta["benchmark"] == "nq"
    assert meta["split"] == "dev"
    assert meta["pipeline_name"] == "dense"
    assert meta["question_id"] == "q-42"


def test_custom_id_escapes_colons_in_question_id():
    cid = make_generation_custom_id("r", "b", "s", "graph", "a::b")
    meta = parse_generation_custom_id(cid)
    assert meta is not None
    assert meta["question_id"] == "a::b"


def test_build_run_root_and_paths():
    root = build_run_root(Path("/tmp/eval"), "nq", "test", "dense", "run-001")
    assert str(root).endswith("nq/test/dense/run-001")
    p = RunArtifactPaths(root)
    assert p.run_root.is_absolute()
    assert p.retrieval_results_jsonl().name == "retrieval_results.jsonl"
    assert p.generation_answers_jsonl().parts[-2:] == ("generation", "answers.jsonl")
    assert p.retrieval_results_jsonl().is_relative_to(p.run_root)


def test_run_artifact_paths_resolves_relative_root(tmp_path):
    rel = tmp_path / "nested" / "run1"
    p = RunArtifactPaths(rel)
    assert p.run_root.is_absolute()
    assert p.run_root == rel.resolve()
    assert p.retrieval_results_jsonl().relative_to(p.run_root).as_posix() == (
        "retrieval/retrieval_results.jsonl"
    )
    assert p.prompt_evidence_jsonl().relative_to(p.run_root).as_posix() == (
        "retrieval/prompt_evidence.jsonl"
    )


def test_default_evaluation_base_is_relative():
    assert default_evaluation_base() == Path("data/evaluation")
