"""Canonical on-disk layout for benchmark answerability audit artifacts."""

from __future__ import annotations

from pathlib import Path


def bundle_root_from_benchmark_jsonl(benchmark_path: Path) -> Path:
    """Bundle root for ``.../<name>/<id>/benchmark/benchmark.jsonl``."""
    p = benchmark_path.resolve()
    return p.parent.parent


def answerability_audit_dir(bundle_root: Path) -> Path:
    return bundle_root / "audit" / "answerability"


def answerability_mask_path(benchmark_path: Path) -> Path:
    return (
        answerability_audit_dir(bundle_root_from_benchmark_jsonl(benchmark_path))
        / "mask.json"
    )


def answerability_manifest_path(benchmark_path: Path) -> Path:
    return (
        answerability_audit_dir(bundle_root_from_benchmark_jsonl(benchmark_path))
        / "manifest.json"
    )


def answerability_verdicts_path(benchmark_path: Path) -> Path:
    return (
        answerability_audit_dir(bundle_root_from_benchmark_jsonl(benchmark_path))
        / "verdicts.jsonl"
    )


def answerability_batch_state_path(benchmark_path: Path) -> Path:
    return (
        answerability_audit_dir(bundle_root_from_benchmark_jsonl(benchmark_path))
        / "batch_state.json"
    )
