"""Merge LLM-as-judge verdicts into metrics.json (pool-aware rollups)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from surf_rag.evaluation.router_overlap import (
    RouterOverlapSplit,
    RouterSplitSets,
    router_overlap_split,
)

OVERLAP_KEYS = ("all", "train", "dev", "test", "unseen")


def _mean_correct(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    vals = [
        r["qa_llm_judge"]["correct"]
        for r in rows
        if isinstance(r.get("qa_llm_judge"), dict) and "correct" in r["qa_llm_judge"]
    ]
    if not vals:
        return {"accuracy": 0.0, "n": 0}
    ok = sum(1 for v in vals if v is True)
    return {"accuracy": ok / len(vals), "n": len(vals)}


def _bucket_rows_for_judge(
    per_question: list[dict[str, Any]],
    split_sets: Optional[RouterSplitSets],
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in OVERLAP_KEYS}
    buckets["all"] = list(per_question)
    if split_sets is None:
        return buckets
    for row in per_question:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid:
            continue
        cat: RouterOverlapSplit = router_overlap_split(qid, split_sets)
        buckets[str(cat)].append(row)
    return buckets


def _attach_judge_rollups(
    overlap_block: Mapping[str, Any],
    per_question: list[dict[str, Any]],
    split_sets: Optional[RouterSplitSets],
) -> None:
    """Mutate each overlap key block in-place with qa_llm_judge summary."""
    buckets = _bucket_rows_for_judge(per_question, split_sets)
    for key in OVERLAP_KEYS:
        block = overlap_block.get(key)
        if not isinstance(block, dict):
            continue
        rows = buckets.get(key, [])
        block["qa_llm_judge"] = _mean_correct(rows)


def merge_llm_judge_verdicts_into_metrics(
    metrics: dict[str, Any],
    verdicts_by_qid: Mapping[str, bool],
    *,
    split_sets: Optional[RouterSplitSets] = None,
) -> dict[str, Any]:
    """Attach ``qa_llm_judge`` per row and rollups under ``overlap_breakdown``.

    If ``overlap_breakdown_primary`` is present (e.g. legacy metrics), rollups are also attached there
    using rows with ``audit.in_primary_eval`` when applicable.

    ``verdicts_by_qid`` maps question_id -> correct (bool). Missing qids are skipped for judge fields.
    """
    per = metrics.get("per_question")
    if not isinstance(per, list):
        return metrics

    split = split_sets
    if split is None:
        spath = metrics.get("split_question_ids")
        if isinstance(spath, str) and spath.strip():
            p = Path(spath)
            if p.is_file():
                split = RouterSplitSets.from_json_path(p)

    for row in per:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id", "") or "").strip()
        if not qid or qid not in verdicts_by_qid:
            continue
        row["qa_llm_judge"] = {"correct": bool(verdicts_by_qid[qid])}

    ob = metrics.get("overlap_breakdown")
    if isinstance(ob, dict):
        _attach_judge_rollups(ob, per, split)

    obp = metrics.get("overlap_breakdown_primary")
    if isinstance(obp, dict):
        primary_rows = [
            r
            for r in per
            if isinstance(r, dict)
            and (
                not isinstance(r.get("audit"), dict)
                or r["audit"].get("in_primary_eval") is True
            )
        ]
        _attach_judge_rollups(obp, primary_rows, split)

    metrics["llm_judge"] = {
        "merged": True,
        "n_verdicts": len(verdicts_by_qid),
    }
    return metrics
