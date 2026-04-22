"""Oracle score generation pipeline.

High-level flow for one oracle run (see ``docs/dev/oracle-pipeline.md``):

1. Snapshot the benchmark rows used for this run.
2. Populate the dense retrieval cache for any missing question_id.
3. Populate the graph retrieval cache for any missing question_id.
4. For each question still missing an oracle_scores row, sweep the fixed
   11-bin dense-weight grid over the cached branch retrievals, fuse, and
   compute stateful NDCG@k (plus diagnostic Hit@k / Recall@k).
5. Pick the best dense weight under the primary NDCG@k for each question.
6. Write a ``summary.json`` overview.

Retrieval is the expensive part and is append-only, so adding new
questions later only drives retrieval for the newly added ``question_id``
values. Changing ``beta`` or the labeling utility never triggers this
stage again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from surf_rag.evaluation.oracle_artifacts import (
    DEFAULT_DENSE_WEIGHT_GRID,
    OracleRunPaths,
    OracleScoreRow,
    WeightBinScore,
    append_oracle_score_rows,
    append_retrieval_line,
    read_question_ids,
    read_retrieval_cache,
    write_manifest,
    write_questions_snapshot,
    write_summary,
)
from surf_rag.evaluation.retrieval_metrics import (
    DEFAULT_NDCG_KS,
    PRIMARY_NDCG_K,
    compute_metric_suite,
)
from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.fusion import fuse_cached_results
from surf_rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


ProgressFn = Optional[Callable[[str, int, int], None]]


@dataclass
class OracleRunConfig:
    """Configuration for one oracle run."""

    benchmark: str
    split: str
    oracle_run_id: str
    benchmark_path: Path
    retrieval_asset_dir: Path
    branch_top_k: int = 25
    fusion_keep_k: int = 25
    weight_grid: Tuple[float, ...] = DEFAULT_DENSE_WEIGHT_GRID
    oracle_metric: str = "stateful_ndcg"
    oracle_metric_k: int = PRIMARY_NDCG_K
    diagnostic_metric_ks: Tuple[int, ...] = DEFAULT_NDCG_KS


def _load_benchmark_rows(
    path: Path, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    import json

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _filter_rows_missing_qids(
    rows: Sequence[Dict[str, Any]], existing: set[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid in existing:
            continue
        out.append(row)
    return out


def populate_retrieval_cache(
    rows: Sequence[Dict[str, Any]],
    cache_path: Path,
    retriever: BranchRetriever,
    *,
    progress: ProgressFn = None,
    label: str = "retrieval",
) -> int:
    """Append retrieval results for any question_id not already cached.

    Returns the number of newly retrieved questions.
    """
    existing = read_question_ids(cache_path)
    pending = _filter_rows_missing_qids(rows, existing)
    total = len(pending)
    if total == 0:
        return 0

    for idx, row in enumerate(pending, start=1):
        qid = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        if not qid or not question:
            continue
        try:
            result = retriever.retrieve(question)
        except Exception as exc:  # defensive: never corrupt the cache
            logger.exception("Retriever %s raised on qid=%s", label, qid)
            result = RetrievalResult(
                query=question,
                retriever_name=retriever.__class__.__name__,
                status="ERROR",
                chunks=[],
                latency_ms={},
                error=f"{type(exc).__name__}: {exc}",
            )
        append_retrieval_line(cache_path, result, question_id=qid)
        if progress is not None:
            progress(label, idx, total)
    return total


def sweep_weights_for_question(
    row: Dict[str, Any],
    dense: RetrievalResult,
    graph: RetrievalResult,
    *,
    weight_grid: Sequence[float],
    fusion_keep_k: int,
    oracle_metric_k: int,
    diagnostic_metric_ks: Sequence[int],
) -> OracleScoreRow:
    """Fuse at every dense weight and score the fused rankings.

    The metric suite is evaluated over the fused shortlist; diagnostic
    metrics at other k values use the same shortlist so evaluation stays
    cheap and self-consistent with fusion.
    """
    gold = list(row.get("gold_support_sentences") or [])
    dataset_source = str(row.get("dataset_source") or "").strip()
    question = str(row.get("question") or "")
    qid = str(row.get("question_id") or "")

    bin_scores: List[WeightBinScore] = []
    all_ks = sorted(set([*diagnostic_metric_ks, oracle_metric_k]))

    for w in weight_grid:
        fused = fuse_cached_results(
            query=question,
            dense=dense,
            graph=graph,
            dense_weight=float(w),
            fusion_keep_k=fusion_keep_k,
        )
        if fused.status != "OK":
            bin_scores.append(
                WeightBinScore(
                    dense_weight=float(w),
                    graph_weight=1.0 - float(w),
                    ndcg_primary=0.0,
                    diagnostic_ndcg={k: 0.0 for k in diagnostic_metric_ks},
                    diagnostic_hit={k: 0.0 for k in diagnostic_metric_ks},
                    diagnostic_recall={k: 0.0 for k in diagnostic_metric_ks},
                    fused_chunk_ids=[],
                )
            )
            continue

        suites = compute_metric_suite(
            fused.chunks,
            gold,
            dataset_source=dataset_source,
            ks=all_ks,
        )
        by_k = {s.k: s for s in suites}
        primary_ndcg = by_k[oracle_metric_k].ndcg
        bin_scores.append(
            WeightBinScore(
                dense_weight=float(w),
                graph_weight=1.0 - float(w),
                ndcg_primary=float(primary_ndcg),
                diagnostic_ndcg={k: by_k[k].ndcg for k in diagnostic_metric_ks},
                diagnostic_hit={k: by_k[k].hit for k in diagnostic_metric_ks},
                diagnostic_recall={k: by_k[k].recall for k in diagnostic_metric_ks},
                fused_chunk_ids=[c.chunk_id for c in fused.chunks],
            )
        )

    best_idx = max(
        range(len(bin_scores)),
        key=lambda i: (
            bin_scores[i].ndcg_primary,
            -abs(bin_scores[i].dense_weight - 0.5),
        ),
        default=0,
    )
    best = bin_scores[best_idx] if bin_scores else None

    return OracleScoreRow(
        question_id=qid,
        question=question,
        dataset_source=dataset_source,
        weight_grid=[float(w) for w in weight_grid],
        oracle_metric="stateful_ndcg",
        oracle_metric_k=int(oracle_metric_k),
        scores=bin_scores,
        best_bin_index=int(best_idx if best else 0),
        best_dense_weight=float(best.dense_weight) if best else 0.0,
        best_score=float(best.ndcg_primary) if best else 0.0,
        dense_status=dense.status,
        graph_status=graph.status,
    )


def sweep_missing_oracle_scores(
    rows: Sequence[Dict[str, Any]],
    paths: OracleRunPaths,
    *,
    weight_grid: Sequence[float],
    fusion_keep_k: int,
    oracle_metric_k: int,
    diagnostic_metric_ks: Sequence[int],
    progress: ProgressFn = None,
) -> int:
    """Compute oracle score rows for any question_id missing from oracle_scores.

    Both retrieval caches are read once, then the sweep runs in-memory over
    cached branch results.
    """
    existing = read_question_ids(paths.oracle_scores)
    pending = _filter_rows_missing_qids(rows, existing)
    if not pending:
        return 0

    dense_cache = read_retrieval_cache(paths.retrieval_dense)
    graph_cache = read_retrieval_cache(paths.retrieval_graph)

    new_rows: List[OracleScoreRow] = []
    total = len(pending)
    for idx, row in enumerate(pending, start=1):
        qid = str(row.get("question_id", "")).strip()
        dense = dense_cache.get(qid)
        graph = graph_cache.get(qid)
        if dense is None or graph is None:
            logger.warning(
                "Skipping oracle sweep for qid=%s: " "dense_cached=%s, graph_cached=%s",
                qid,
                dense is not None,
                graph is not None,
            )
            continue

        new_rows.append(
            sweep_weights_for_question(
                row,
                dense,
                graph,
                weight_grid=weight_grid,
                fusion_keep_k=fusion_keep_k,
                oracle_metric_k=oracle_metric_k,
                diagnostic_metric_ks=diagnostic_metric_ks,
            )
        )
        if progress is not None:
            progress("oracle_sweep", idx, total)

    append_oracle_score_rows(paths, new_rows)
    return len(new_rows)


def build_summary(
    paths: OracleRunPaths,
    cfg: OracleRunConfig,
    *,
    snapshot_count: int,
    newly_retrieved_dense: int,
    newly_retrieved_graph: int,
    newly_scored: int,
) -> Dict[str, Any]:
    dense_qids = read_question_ids(paths.retrieval_dense)
    graph_qids = read_question_ids(paths.retrieval_graph)
    score_qids = read_question_ids(paths.oracle_scores)
    return {
        "oracle_run_id": cfg.oracle_run_id,
        "benchmark": cfg.benchmark,
        "split": cfg.split,
        "questions_snapshot": snapshot_count,
        "dense_cached": len(dense_qids),
        "graph_cached": len(graph_qids),
        "oracle_scored": len(score_qids),
        "newly_retrieved_dense": newly_retrieved_dense,
        "newly_retrieved_graph": newly_retrieved_graph,
        "newly_scored": newly_scored,
        "weight_grid": list(cfg.weight_grid),
        "branch_top_k": cfg.branch_top_k,
        "fusion_keep_k": cfg.fusion_keep_k,
        "oracle_metric": cfg.oracle_metric,
        "oracle_metric_k": cfg.oracle_metric_k,
        "diagnostic_metric_ks": list(cfg.diagnostic_metric_ks),
        "artifacts": {
            "manifest": paths.manifest.name,
            "questions_snapshot": paths.questions_snapshot.name,
            "retrieval_dense": paths.retrieval_dense.name,
            "retrieval_graph": paths.retrieval_graph.name,
            "oracle_scores": paths.oracle_scores.name,
            "labels_dir": paths.labels_dir.name,
        },
    }


def prepare_oracle_run(
    cfg: OracleRunConfig,
    paths: OracleRunPaths,
    dense_retriever_factory: Callable[[], BranchRetriever],
    graph_retriever_factory: Callable[[], BranchRetriever],
    *,
    limit: Optional[int] = None,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    """Top-level orchestrator used by :mod:`scripts.prepare_oracle_run`.

    Retriever construction is deferred via factories so this function is
    directly unit-testable with in-memory fakes.
    """
    paths.ensure_dirs()

    write_manifest(
        paths,
        oracle_run_id=cfg.oracle_run_id,
        benchmark=cfg.benchmark,
        split=cfg.split,
        benchmark_path=str(cfg.benchmark_path),
        retrieval_asset_dir=str(cfg.retrieval_asset_dir),
        weight_grid=cfg.weight_grid,
        branch_top_k=cfg.branch_top_k,
        fusion_keep_k=cfg.fusion_keep_k,
        oracle_metric=cfg.oracle_metric,
        oracle_metric_k=cfg.oracle_metric_k,
        diagnostic_metric_ks=cfg.diagnostic_metric_ks,
    )

    rows = _load_benchmark_rows(cfg.benchmark_path, limit=limit)
    snapshot_count = write_questions_snapshot(paths, rows)

    dense_retriever = dense_retriever_factory()
    newly_dense = populate_retrieval_cache(
        rows,
        paths.retrieval_dense,
        dense_retriever,
        progress=progress,
        label="dense_retrieval",
    )

    graph_retriever = graph_retriever_factory()
    newly_graph = populate_retrieval_cache(
        rows,
        paths.retrieval_graph,
        graph_retriever,
        progress=progress,
        label="graph_retrieval",
    )

    newly_scored = sweep_missing_oracle_scores(
        rows,
        paths,
        weight_grid=cfg.weight_grid,
        fusion_keep_k=cfg.fusion_keep_k,
        oracle_metric_k=cfg.oracle_metric_k,
        diagnostic_metric_ks=cfg.diagnostic_metric_ks,
        progress=progress,
    )

    summary = build_summary(
        paths,
        cfg,
        snapshot_count=snapshot_count,
        newly_retrieved_dense=newly_dense,
        newly_retrieved_graph=newly_graph,
        newly_scored=newly_scored,
    )
    write_summary(paths, summary)
    return summary
