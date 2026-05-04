from __future__ import annotations

from pathlib import Path

from surf_rag.evaluation.answerability_layout import (
    answerability_audit_dir,
    answerability_manifest_path,
    answerability_mask_path,
    answerability_verdicts_path,
    bundle_root_from_benchmark_jsonl,
)


def test_bundle_root_from_benchmark_jsonl(tmp_path: Path) -> None:
    bench = tmp_path / "surf" / "main" / "benchmark" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.touch()
    root = bundle_root_from_benchmark_jsonl(bench)
    assert root == tmp_path / "surf" / "main"


def test_answerability_paths_under_audit(tmp_path: Path) -> None:
    bench = tmp_path / "b" / "v1" / "benchmark" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bundle = bundle_root_from_benchmark_jsonl(bench)
    assert (
        answerability_audit_dir(bundle)
        == tmp_path / "b" / "v1" / "audit" / "answerability"
    )
    assert (
        answerability_mask_path(bench)
        == bundle / "audit" / "answerability" / "mask.json"
    )
    assert (
        answerability_manifest_path(bench)
        == bundle / "audit" / "answerability" / "manifest.json"
    )
    assert (
        answerability_verdicts_path(bench)
        == bundle / "audit" / "answerability" / "verdicts.jsonl"
    )
