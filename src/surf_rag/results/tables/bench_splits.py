"""Benchmark split composition: E2E (raw) vs orchestrator (answerability-masked)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from surf_rag.evaluation.answerability_layout import answerability_mask_path
from surf_rag.evaluation.answerability_types import MaskReason, load_mask_json_path
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.tables.writer import write_table

_SPLITS = ("train", "dev", "test")
_SOURCES = ("nq", "2wiki", "all")


def _empty_counts() -> dict[str, dict[str, int]]:
    return {src: {sp: 0 for sp in _SPLITS} for src in _SOURCES}


def _accumulate_split_counts(
    payload: Mapping[str, object],
    qid_to_source: Mapping[str, str],
    mask_by_qid: Mapping[str, MaskReason],
    *,
    apply_mask: bool,
) -> dict[str, dict[str, int]]:
    counts = _empty_counts()
    for split in _SPLITS:
        for qid in payload.get(split) or []:
            q = str(qid).strip()
            if not q:
                continue
            if apply_mask and q in mask_by_qid:
                continue
            src = qid_to_source.get(q, "unknown")
            counts["all"][split] += 1
            if src in counts:
                counts[src][split] += 1
    return counts


def _format_split_triple(counts: dict[str, dict[str, int]], source: str) -> str:
    c = counts[source]
    return f"{c['train']}/{c['dev']}/{c['test']}"


def _load_mask_optional(
    benchmark_path: Path,
) -> tuple[dict[str, MaskReason], Path | None]:
    mask_path = answerability_mask_path(benchmark_path)
    if not mask_path.is_file():
        return {}, None
    return load_mask_json_path(mask_path), mask_path


def build_bench_splits(bundle: ResultsBundle) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(bundle.split_question_ids_path.read_text(encoding="utf-8"))
    mask_by_qid, mask_path = _load_mask_optional(bundle.resolved.benchmark_path)
    mask_applied = bool(mask_by_qid)
    if mask_path is None:
        bundle.warnings.append(
            "bench_splits: answerability mask.json not found; "
            "orchestrator_split equals e2e_split (no exclusions)."
        )

    e2e_counts = _accumulate_split_counts(
        payload, bundle.qid_to_source, mask_by_qid, apply_mask=False
    )
    orch_counts = _accumulate_split_counts(
        payload, bundle.qid_to_source, mask_by_qid, apply_mask=True
    )

    rows: list[dict[str, str]] = []
    for src in _SOURCES:
        label = {"nq": "nq", "2wiki": "2wiki", "all": "total"}[src]
        rows.append(
            {
                "dataset_source": label,
                "e2e_split": _format_split_triple(e2e_counts, src),
                "orchestrator_split": _format_split_triple(orch_counts, src),
            }
        )

    df = pd.DataFrame(rows)
    meta = {
        "split_filter": bundle.results.split,
        "mask_path": str(mask_path) if mask_path else None,
        "mask_applied": mask_applied,
        "n_masked_in_splits": sum(
            1
            for sp in _SPLITS
            for qid in payload.get(sp) or []
            if str(qid).strip() in mask_by_qid
        ),
    }
    paths = write_table(bundle, "bench_splits", df, meta)
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
