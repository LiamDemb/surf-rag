"""Load frozen artefacts for results builds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    read_jsonl,
    read_retrieval_cache,
)
from surf_rag.router.splits import normalize_dataset_source


def load_split_qids(path: Path, split: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [split] if split != "all" else ["train", "dev", "test"]
    out: set[str] = set()
    for name in names:
        for qid in payload.get(name) or []:
            s = str(qid).strip()
            if s:
                out.add(s)
    return out


def load_benchmark_sources(benchmark_jsonl: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not benchmark_jsonl.is_file():
        return out
    for line in benchmark_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id", "")).strip()
        if qid:
            out[qid] = normalize_dataset_source(str(row.get("dataset_source", "")))
    return out


def load_oracle_rows(oracle_scores: Path) -> list[dict[str, Any]]:
    if not oracle_scores.is_file():
        raise FileNotFoundError(f"Oracle scores not found: {oracle_scores}")
    return list(read_jsonl(oracle_scores))


def load_policy_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"E2E metrics not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_answerability(verdicts_path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not verdicts_path.is_file():
        return out
    for line in verdicts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id", "")).strip()
        if qid:
            out[qid] = bool(row.get("answerable"))
    return out


def load_graph_no_context_rates(
    retrieval_graph: Path,
    qid_to_source: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Return per-source and 'all' NO_CONTEXT rates over the full benchmark."""
    counts: dict[str, dict[str, int]] = {
        "all": {"total": 0, "no_context": 0},
        "nq": {"total": 0, "no_context": 0},
        "2wiki": {"total": 0, "no_context": 0},
    }
    if not retrieval_graph.is_file():
        raise FileNotFoundError(f"Graph retrieval cache not found: {retrieval_graph}")
    cache = read_retrieval_cache(retrieval_graph)
    for qid, result in cache.items():
        src = qid_to_source.get(qid, "unknown")
        keys = ["all"]
        if src in ("nq", "2wiki"):
            keys.append(src)
        for key in keys:
            counts[key]["total"] += 1
            if getattr(result, "status", None) == "NO_CONTEXT":
                counts[key]["no_context"] += 1
    rates: dict[str, dict[str, float]] = {}
    for key, c in counts.items():
        total = c["total"]
        rates[key] = {
            "no_context_rate": (c["no_context"] / total) if total else 0.0,
            "n": float(total),
        }
    return rates


def iter_predictions_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")
    yield from read_jsonl(path)


def split_question_ids_path(router_dataset_dir: Path) -> Path:
    return router_dataset_dir / "split_question_ids.json"


def oracle_paths_from_dir(router_oracle_dir: Path) -> OracleRunPaths:
    return OracleRunPaths(run_root=router_oracle_dir)
