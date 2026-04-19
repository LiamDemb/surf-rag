"""Run-scoped paths under data/evaluation/<benchmark>/<split>/<pipeline>/<run_id>/."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def default_evaluation_base() -> Path:
    return Path(os.getenv("EVALUATION_BASE", "data/evaluation"))


def build_run_root(
    evaluation_base: Path,
    benchmark: str,
    split: str,
    pipeline_name: str,
    run_id: str,
) -> Path:
    """Root directory for one evaluation run."""
    return evaluation_base / benchmark / split / pipeline_name / run_id


@dataclass(frozen=True)
class RunArtifactPaths:
    """Standard subpaths for retrieval, batch, and generation stages."""

    run_root: Path

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def retrieval_dir(self) -> Path:
        return self.run_root / "retrieval"

    @property
    def batch_dir(self) -> Path:
        return self.run_root / "batch"

    @property
    def generation_dir(self) -> Path:
        return self.run_root / "generation"

    def retrieval_results_jsonl(self) -> Path:
        return self.retrieval_dir / "retrieval_results.jsonl"

    def batch_input_jsonl(self) -> Path:
        return self.batch_dir / "batch_input.jsonl"

    def batch_state_json(self) -> Path:
        return self.batch_dir / "batch_state.json"

    def batch_shard_pattern(self) -> str:
        return "batch_input_shard_{idx:03d}.jsonl"

    def batch_shard_path(self, idx: int) -> Path:
        return self.batch_dir / f"batch_input_shard_{idx:03d}.jsonl"

    def batch_output_raw_jsonl(self) -> Path:
        return self.batch_dir / "batch_output_raw.jsonl"

    def batch_output_parsed_jsonl(self) -> Path:
        return self.batch_dir / "batch_output_parsed.jsonl"

    def generation_answers_jsonl(self) -> Path:
        return self.generation_dir / "answers.jsonl"

    def ensure_dirs(self) -> None:
        self.retrieval_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.generation_dir.mkdir(parents=True, exist_ok=True)


def make_generation_custom_id(
    run_id: str,
    benchmark: str,
    split: str,
    pipeline_name: str,
    question_id: str,
) -> str:
    """OpenAI Batch custom_id. Escapes :: inside question_id."""
    safe = question_id.replace("::", "__COLON__")
    return f"{run_id}::{benchmark}::{split}::{pipeline_name}::{safe}"


def parse_generation_custom_id(custom_id: str) -> Optional[Dict[str, str]]:
    """Parse custom_id from make_generation_custom_id. Returns None if invalid."""
    parts = custom_id.split("::")
    if len(parts) < 5:
        return None
    run_id, bench, spl, pipe = parts[0], parts[1], parts[2], parts[3]
    qid = "::".join(parts[4:]).replace("__COLON__", "::")
    return {
        "run_id": run_id,
        "benchmark": bench,
        "split": spl,
        "pipeline_name": pipe,
        "question_id": qid,
    }
