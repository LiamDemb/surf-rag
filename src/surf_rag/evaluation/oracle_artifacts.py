"""Oracle run directories, manifests, and JSONL helpers."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from surf_rag.evaluation.artifact_paths import default_data_base, default_router_base
from surf_rag.evaluation.retrieval_jsonl import retrieval_result_to_dict
from surf_rag.evaluation.run_artifacts import as_resolved_path
from surf_rag.evaluation.weight_grid import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def default_oracle_base() -> Path:
    """Return the base directory for router bundles."""
    return default_router_base()


def build_oracle_run_root(router_base: Path, router_id: str) -> Path:
    return router_base / router_id / "oracle"


@dataclass(frozen=True)
class OracleRunPaths:
    """Standard subpaths inside one oracle run root."""

    run_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", as_resolved_path(self.run_root))

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def provenance(self) -> Path:
        return self.run_root / "provenance.json"

    @property
    def summary(self) -> Path:
        return self.run_root / "summary.json"

    @property
    def questions_snapshot(self) -> Path:
        return self.run_root / "questions_snapshot.jsonl"

    @property
    def retrieval_dense(self) -> Path:
        return self.run_root / "retrieval_dense.jsonl"

    @property
    def retrieval_graph(self) -> Path:
        return self.run_root / "retrieval_graph.jsonl"

    @property
    def oracle_scores(self) -> Path:
        return self.run_root / "oracle_scores.jsonl"

    @property
    def router_labels(self) -> Path:
        return self.run_root / "router_labels.jsonl"

    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    def ensure_dirs(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_paths_for_cli(
    router_id: str, *, router_base: Optional[Path] = None
) -> OracleRunPaths:
    base = router_base if router_base is not None else default_oracle_base()
    return OracleRunPaths(run_root=build_oracle_run_root(base, router_id))


def make_run_paths_for_legacy(
    benchmark: str,
    split: str,
    oracle_run_id: str,
    oracle_base: Optional[Path] = None,
) -> OracleRunPaths:
    warnings.warn(
        "make_run_paths_for_legacy and data/oracle/.../split/.../ paths are "
        "deprecated. Use data/router/<router_id>/oracle/ via make_run_paths_for_cli().",
        DeprecationWarning,
        stacklevel=2,
    )
    base = oracle_base if oracle_base is not None else default_data_base() / "oracle"
    return OracleRunPaths(run_root=base / benchmark / split / oracle_run_id)


def write_provenance(
    paths: OracleRunPaths,
    *,
    router_id: str,
    benchmark_name: str,
    benchmark_id: str,
    benchmark_path: str,
    retrieval_asset_dir: str,
) -> None:
    """Write machine-readable source bundle/corpus references for the oracle run."""
    paths.ensure_dirs()
    data: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "router_id": router_id,
        "source_benchmark_bundle": {
            "name": benchmark_name,
            "id": benchmark_id,
            "benchmark_jsonl": benchmark_path,
        },
        "source_corpus": {
            "retrieval_asset_dir": retrieval_asset_dir,
        },
    }
    paths.provenance.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    paths: OracleRunPaths,
    *,
    router_id: str,
    benchmark_name: str,
    benchmark_id: str,
    benchmark_path: str,
    retrieval_asset_dir: str,
    weight_grid: Iterable[float],
    branch_top_k: int,
    fusion_keep_k: int,
    oracle_metric: str,
    oracle_metric_k: int,
    diagnostic_metric_ks: Iterable[int],
    source_split: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``manifest.json`` for a router-scoped oracle run."""
    paths.ensure_dirs()
    data: Dict[str, Any] = {
        "schema_version": 3,
        "created_at": utc_now_iso(),
        "router_id": router_id,
        "benchmark_name": benchmark_name,
        "benchmark_id": benchmark_id,
        # Back-compat aliases
        "oracle_run_id": router_id,
        "benchmark": benchmark_name,
        "benchmark_path": benchmark_path,
        "retrieval_asset_dir": retrieval_asset_dir,
        "weight_grid": [float(w) for w in weight_grid],
        "branch_top_k": int(branch_top_k),
        "fusion_keep_k": int(fusion_keep_k),
        "oracle_metric": oracle_metric,
        "oracle_metric_k": int(oracle_metric_k),
        "diagnostic_metric_ks": [int(k) for k in diagnostic_metric_ks],
        "artifacts": {
            "provenance": paths.provenance.name,
            "questions_snapshot": paths.questions_snapshot.name,
            "retrieval_dense": paths.retrieval_dense.name,
            "retrieval_graph": paths.retrieval_graph.name,
            "oracle_scores": paths.oracle_scores.name,
            "router_labels": paths.router_labels.name,
            "reports_dir": paths.reports_dir.name,
        },
    }
    if source_split is not None:
        data["source_split"] = source_split
    if extra:
        data.update(extra)
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_manifest(paths: OracleRunPaths) -> Dict[str, Any]:
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def update_manifest(paths: OracleRunPaths, updates: Dict[str, Any]) -> None:
    data = read_manifest(paths)
    data.update(updates)
    data["updated_at"] = utc_now_iso()
    paths.manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_summary(paths: OracleRunPaths, summary: Dict[str, Any]) -> None:
    paths.ensure_dirs()
    data = dict(summary)
    data["updated_at"] = utc_now_iso()
    paths.summary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_summary(paths: OracleRunPaths) -> Dict[str, Any]:
    if not paths.summary.is_file():
        return {}
    return json.loads(paths.summary.read_text(encoding="utf-8"))


# JSON row helpers
def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Overwrite ``path`` with ``rows`` (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(_read_jsonl(path))


def read_question_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in _read_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid:
            ids.add(qid)
    return ids


# Question snapshot
def write_questions_snapshot(
    paths: OracleRunPaths, rows: Iterable[Dict[str, Any]]
) -> int:
    """Freeze the exact benchmark rows used for this oracle run.

    Returns the number of rows written.
    """
    paths.ensure_dirs()
    count = 0
    with paths.questions_snapshot.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# Retrieval cache helpers
def append_retrieval_line(
    path: Path,
    result: RetrievalResult,
    question_id: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one retrieval row to ``path`` (dense or graph cache)."""
    row = retrieval_result_to_dict(result, question_id, extra=extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_retrieval_cache(path: Path) -> Dict[str, RetrievalResult]:
    """Load a retrieval cache JSONL into ``{question_id: RetrievalResult}``."""
    out: Dict[str, RetrievalResult] = {}
    for row in _read_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if not qid:
            continue
        chunks = [
            RetrievedChunk(
                chunk_id=str(c.get("chunk_id", "")),
                text=str(c.get("text", "")),
                score=float(c.get("score", 0.0)),
                rank=int(c.get("rank", 0)),
                metadata=dict(c.get("metadata") or {}),
            )
            for c in row.get("chunks", [])
            if c.get("chunk_id")
        ]
        status = row.get("status", "NO_CONTEXT")
        # RetrievalResult.__post_init__ enforces invariants. When rehydrating
        # an OK row whose chunks list happens to be empty we downgrade to
        # NO_CONTEXT for safety.
        if status == "OK" and not chunks:
            status = "NO_CONTEXT"
        out[qid] = RetrievalResult(
            query=str(row.get("query", "")),
            retriever_name=str(row.get("retriever_name", "")),
            status=status,
            chunks=chunks,
            latency_ms=dict(row.get("latency_ms") or {}),
            error=row.get("error"),
        )
    return out


# Oracle score rows
@dataclass
class WeightBinScore:
    """Per-weight-bin diagnostics and fused shortlist."""

    dense_weight: float
    graph_weight: float
    ndcg_primary: float
    oracle_objective_value: float = 0.0
    diagnostic_ndcg: Dict[int, float] = field(default_factory=dict)
    diagnostic_hit: Dict[int, float] = field(default_factory=dict)
    diagnostic_recall: Dict[int, float] = field(default_factory=dict)
    fused_chunk_ids: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "dense_weight": float(self.dense_weight),
            "graph_weight": float(self.graph_weight),
            "ndcg_primary": float(self.ndcg_primary),
            "oracle_objective_value": float(self.oracle_objective_value),
            "diagnostic_ndcg": {
                str(k): float(v) for k, v in self.diagnostic_ndcg.items()
            },
            "diagnostic_hit": {
                str(k): float(v) for k, v in self.diagnostic_hit.items()
            },
            "diagnostic_recall": {
                str(k): float(v) for k, v in self.diagnostic_recall.items()
            },
            "fused_chunk_ids": list(self.fused_chunk_ids),
        }


@dataclass
class OracleScoreRow:
    """One question's full oracle sweep row (stored in oracle_scores.jsonl)."""

    question_id: str
    question: str
    dataset_source: str
    weight_grid: List[float]
    oracle_metric: str
    oracle_metric_k: int
    scores: List[WeightBinScore]
    best_bin_index: int
    best_dense_weight: float
    best_score: float
    dense_status: str
    graph_status: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "dataset_source": self.dataset_source,
            "weight_grid": [float(w) for w in self.weight_grid],
            "oracle_metric": self.oracle_metric,
            "oracle_metric_k": int(self.oracle_metric_k),
            "scores": [s.to_json() for s in self.scores],
            "best_bin_index": int(self.best_bin_index),
            "best_dense_weight": float(self.best_dense_weight),
            "best_score": float(self.best_score),
            "dense_status": self.dense_status,
            "graph_status": self.graph_status,
        }


def append_oracle_score_rows(
    paths: OracleRunPaths, rows: Iterable[OracleScoreRow]
) -> None:
    _append_jsonl(paths.oracle_scores, (r.to_json() for r in rows))


def overwrite_oracle_score_rows(
    paths: OracleRunPaths, rows: Iterable[OracleScoreRow]
) -> None:
    write_jsonl(paths.oracle_scores, (r.to_json() for r in rows))


def read_oracle_score_rows(paths: OracleRunPaths) -> List[Dict[str, Any]]:
    """Return raw oracle score rows as dicts."""
    return list(_read_jsonl(paths.oracle_scores))
