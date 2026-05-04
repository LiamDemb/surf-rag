"""Answerability audit: mask schema, balance logic, manifest document building."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

MASK_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

MaskReason = str  # "audit" | "balance"


def load_mask_index(mask_doc: Mapping[str, Any]) -> dict[str, MaskReason]:
    """question_id -> reason (audit overrides balance if duplicate)."""
    out: dict[str, MaskReason] = {}
    for ent in mask_doc.get("entries") or []:
        if not isinstance(ent, Mapping):
            continue
        qid = str(ent.get("question_id", "") or "").strip()
        reason = str(ent.get("reason", "") or "").strip()
        if not qid or reason not in ("audit", "balance"):
            continue
        prev = out.get(qid)
        if prev == "audit":
            continue
        if reason == "audit":
            out[qid] = "audit"
        elif prev != "audit":
            out[qid] = reason
    return out


def load_mask_json_path(path: Path) -> dict[str, MaskReason]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"mask.json must be an object: {path}")
    return load_mask_index(data)


def in_primary_eval(qid: str, mask_by_qid: Mapping[str, str]) -> bool:
    return str(qid).strip() not in mask_by_qid


def audit_entries_from_verdicts(
    verdict_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """One entry per unanswerable verdict (reason=audit)."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in verdict_rows:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        if row.get("answerable") is False:
            entries.append({"question_id": qid, "reason": "audit"})
    return entries


def build_balance_mask(
    verdict_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    policy: str = "equal_per_source_min",
) -> list[dict[str, str]]:
    """Return mask entries with reason=balance to equalize per-source answerable counts.

    Policy ``equal_per_source_min``: among rows with ``answerable is True``, let
    ``k = min(count by dataset_source)``. For each source with more than ``k``
    answerable question_ids, randomly remove ``count - k`` (seeded) from primary eval.
    """
    if policy != "equal_per_source_min":
        raise ValueError(f"Unsupported balance policy: {policy!r}")

    by_source: dict[str, list[str]] = defaultdict(list)
    seen_answerable: set[str] = set()
    for row in verdict_rows:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid or qid in seen_answerable:
            continue
        if row.get("answerable") is not True:
            continue
        seen_answerable.add(qid)
        src = str(row.get("dataset_source", "") or "").strip() or "unknown"
        by_source[src].append(qid)

    if not by_source:
        return []

    k = min(len(v) for v in by_source.values())
    rng = Random(seed)
    balance: list[dict[str, str]] = []
    for src, qids in by_source.items():
        if len(qids) <= k:
            continue
        pool = list(qids)
        rng.shuffle(pool)
        for qid in pool[k:]:
            balance.append({"question_id": qid, "reason": "balance"})
    return balance


def build_mask_document(
    *,
    audit_entries: Sequence[Mapping[str, str]],
    balance_entries: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    merged: dict[str, dict[str, str]] = {}
    for ent in balance_entries:
        qid = str(ent.get("question_id", "") or "").strip()
        if qid:
            merged[qid] = {"question_id": qid, "reason": "balance"}
    for ent in audit_entries:
        qid = str(ent.get("question_id", "") or "").strip()
        if qid:
            merged[qid] = {"question_id": qid, "reason": "audit"}
    entries = sorted(merged.values(), key=lambda e: (e["reason"], e["question_id"]))
    return {"schema_version": MASK_SCHEMA_VERSION, "entries": entries}


def build_manifest_document(
    *,
    benchmark_path: Path,
    audit_model: str,
    prompt_id: str,
    verdict_rows: Sequence[Mapping[str, Any]],
    mask_entries: Sequence[Mapping[str, str]],
    balance_enabled: bool,
    balance_policy: str,
    balance_seed: int | None,
) -> dict[str, Any]:
    """Full manifest.json payload for operators."""
    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"answerable": 0, "unanswerable": 0}
    )
    seen: set[str] = set()
    for row in verdict_rows:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        src = str(row.get("dataset_source", "") or "").strip() or "unknown"
        if row.get("answerable") is True:
            by_source[src]["answerable"] += 1
        elif row.get("answerable") is False:
            by_source[src]["unanswerable"] += 1

    mask_by_reason: dict[str, list[str]] = {"audit": [], "balance": []}
    for ent in mask_entries:
        qid = str(ent.get("question_id", "") or "").strip()
        reason = str(ent.get("reason", "") or "").strip()
        if qid and reason in mask_by_reason:
            mask_by_reason[reason].append(qid)

    n_audit = len(mask_by_reason["audit"])
    n_balance = len(mask_by_reason["balance"])

    primary_by_source: dict[str, int] = defaultdict(int)
    verdict_by_qid = {str(r.get("question_id", "")).strip(): r for r in verdict_rows}
    mask_qids = set(mask_by_reason["audit"]) | set(mask_by_reason["balance"])
    for qid, row in verdict_by_qid.items():
        if not qid or qid in mask_qids:
            continue
        src = str(row.get("dataset_source", "") or "").strip() or "unknown"
        if row.get("answerable") is True:
            primary_by_source[src] += 1

    balance_removals_by_source: dict[str, int] = defaultdict(int)
    for qid in mask_by_reason["balance"]:
        row = verdict_by_qid.get(qid) or {}
        src = str(row.get("dataset_source", "") or "").strip() or "unknown"
        balance_removals_by_source[src] += 1

    totals = {
        "questions_in_verdicts": len(seen),
        "answerable": sum(b["answerable"] for b in by_source.values()),
        "unanswerable": sum(b["unanswerable"] for b in by_source.values()),
    }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_path": str(benchmark_path.resolve()),
        "audit_model": audit_model,
        "prompt_id": prompt_id,
        "totals": totals,
        "audit_by_source": dict(by_source),
        "balance": {
            "enabled": balance_enabled,
            "policy": balance_policy,
            "seed": balance_seed,
        },
        "balance_removals_by_source": dict(balance_removals_by_source),
        "primary_eval": {
            "n_total": sum(primary_by_source.values()),
            "by_source": dict(primary_by_source),
        },
        "excluded": {
            "n_audit": n_audit,
            "n_balance": n_balance,
            "n_total_excluded_from_primary": n_audit + n_balance,
        },
    }


def iter_verdicts_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
