"""Compare two E2E runs: flag questions where nDCG@10 improves but QA F1 does not."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from surf_rag.evaluation.e2e_aggregation import (
    aggregate_per_question,
    load_benchmark_index,
)
from surf_rag.evaluation.manifest import utc_now_iso
from surf_rag.evaluation.retrieval_jsonl import dict_to_retrieval_result
from surf_rag.evaluation.retrieval_metrics import RankedMetricSuite
from surf_rag.evaluation.run_artifacts import as_resolved_path
from surf_rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "surf-rag/discrepancy-debug/v1"
METRIC_K = 10


def load_retrieval_by_qid(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "") or "").strip()
            if qid:
                out[qid] = row
    return out


def load_answers_by_qid(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "") or "").strip()
            if qid:
                out[qid] = row
    return out


def read_e2e_manifest_block(run_root: Path) -> dict[str, Any]:
    mf = Path(run_root) / "manifest.json"
    if not mf.is_file():
        return {}
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    blk = data.get("e2e")
    return dict(blk) if isinstance(blk, dict) else {}


def normalize_reranker_kind(kind: object) -> str:
    return str(kind or "").strip().lower()


def reranker_is_none_or_disabled(e2e: Mapping[str, Any]) -> bool:
    """True when reranker is absent or explicitly ``none`` (matches evaluate_e2e_run)."""
    return normalize_reranker_kind(e2e.get("reranker", "")) in ("", "none")


def surf_rag_version_string() -> str | None:
    try:
        from importlib.metadata import version

        return version("surf_rag")
    except Exception:
        return None


def _missing_retrieval_result(query: str) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        retriever_name="missing",
        status="EMPTY",
        chunks=[],
        latency_ms={},
        error="no_retrieval_row",
    )


def _retrieval_from_row(
    sample: Mapping[str, Any], retrieval_row: dict | None
) -> RetrievalResult:
    q = str(sample.get("question", "") or "")
    if retrieval_row is None:
        return _missing_retrieval_result(q)
    return dict_to_retrieval_result(retrieval_row)


def _suite_at_k(suites: Sequence[RankedMetricSuite], k: int) -> RankedMetricSuite:
    for s in suites:
        if int(s.k) == k:
            return s
    raise ValueError(f"No RankedMetricSuite for k={k} in {[x.k for x in suites]}")


def score_question_for_metrics(
    sample: Mapping[str, Any],
    retrieval_row: dict | None,
    answer_row: dict | None,
    *,
    question_id: str,
    ks: Tuple[int, ...] = (METRIC_K,),
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    str,
    RetrievalResult,
]:
    """Returns (ndcg_k, hit_k, recall_k, f1, em, prediction str, retrieval status, result)."""
    gold_sents = list(sample.get("gold_support_sentences") or [])
    ds = sample.get("dataset_source")
    gold_ans = list(sample.get("gold_answers") or [])
    pred = str((answer_row or {}).get("answer", "") or "")

    rr = _retrieval_from_row(sample, retrieval_row)

    pe = aggregate_per_question(
        question_id,
        result=rr,
        gold_support_sentences=gold_sents,
        dataset_source=str(ds) if ds else None,
        gold_answers=gold_ans,
        prediction=pred,
        ks=ks,
    )
    suite = _suite_at_k(pe.retrieval_suites, METRIC_K)
    return (
        suite.ndcg,
        suite.hit,
        suite.recall,
        pe.f1,
        pe.em,
        pred,
        str(rr.status or ""),
        rr,
    )


def top_chunks_preview(
    result: RetrievalResult,
    *,
    top_k: int,
    preview_chars: int,
) -> List[Dict[str, Any]]:
    chunks = list(result.chunks)[:top_k]
    out: List[Dict[str, Any]] = []
    for c in chunks:
        text = str(c.text or "")
        preview = text
        if preview_chars > 0 and len(text) > preview_chars:
            preview = text[:preview_chars] + "…"
        out.append(
            {
                "rank": int(c.rank),
                "chunk_id": str(c.chunk_id or ""),
                "score": float(c.score),
                "text_preview": preview,
            }
        )
    return out


def chunk_ids_top_k(result: RetrievalResult, top_k: int) -> List[str]:
    ids: List[str] = []
    for c in result.chunks[:top_k]:
        cid = str(c.chunk_id or "")
        ids.append(cid)
    return ids


def jaccard_chunk_ids(ids_a: Sequence[str], ids_b: Sequence[str]) -> float:
    sa = set(x for x in ids_a if x)
    sb = set(x for x in ids_b if x)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0


@dataclass(frozen=True)
class SelectionRule:
    epsilon_ndcg: float
    delta_f1: float
    baseline_label: str = "run_a"


@dataclass
class ComparisonCounts:
    joined_question_ids: int = 0
    interesting: int = 0
    skipped_missing_retrieval_a: int = 0
    skipped_missing_retrieval_b: int = 0
    skipped_missing_answer_a: int = 0
    skipped_missing_answer_b: int = 0
    skipped_restrict_filter: int = 0


def manifest_excerpt(e2e: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "router_id",
        "router_architecture_id",
        "routing_policy",
        "reranker",
        "benchmark_id",
        "benchmark_id_legacy",
    )
    return {k: e2e.get(k) for k in keys if k in e2e}


def extract_interesting_rows(
    *,
    bench_by_qid: Mapping[str, Mapping[str, Any]],
    retr_a: Mapping[str, dict],
    retr_b: Mapping[str, dict],
    ans_a: Mapping[str, dict],
    ans_b: Mapping[str, dict],
    restrict_qids: Optional[set[str]],
    rule: SelectionRule,
    top_k_chunks: int,
    chunk_preview_chars: int,
) -> Tuple[List[Dict[str, Any]], ComparisonCounts]:
    counts = ComparisonCounts()
    interesting: List[Dict[str, Any]] = []

    for qid in sorted(bench_by_qid.keys(), key=lambda x: str(x)):
        sample = bench_by_qid[qid]
        qs = str(qid).strip()
        if restrict_qids is not None and qs not in restrict_qids:
            counts.skipped_restrict_filter += 1
            continue
        counts.joined_question_ids += 1

        ra_row = retr_a.get(qs)
        rb_row = retr_b.get(qs)
        aa_row = ans_a.get(qs)
        ab_row = ans_b.get(qs)

        if ra_row is None:
            counts.skipped_missing_retrieval_a += 1
        if rb_row is None:
            counts.skipped_missing_retrieval_b += 1
        if aa_row is None:
            counts.skipped_missing_answer_a += 1
        if ab_row is None:
            counts.skipped_missing_answer_b += 1

        nd_a, _, _, f1_a, em_a, pred_a, st_a, res_a = score_question_for_metrics(
            sample, ra_row, aa_row, question_id=qs
        )
        nd_b, _, _, f1_b, em_b, pred_b, st_b, res_b = score_question_for_metrics(
            sample, rb_row, ab_row, question_id=qs
        )

        d_nd = nd_b - nd_a
        d_f1 = f1_b - f1_a
        if d_nd >= rule.epsilon_ndcg and f1_b <= f1_a + rule.delta_f1:
            jid = jaccard_chunk_ids(
                chunk_ids_top_k(res_a, top_k_chunks),
                chunk_ids_top_k(res_b, top_k_chunks),
            )
            gold_answers_list = list(sample.get("gold_answers") or [])
            row_out: Dict[str, Any] = {
                "question_id": qs,
                "dataset_source": sample.get("dataset_source"),
                "question": sample.get("question", ""),
                "gold_answers": gold_answers_list,
                "run_a": {
                    "ndcg_at_10": nd_a,
                    "f1": f1_a,
                    "em": em_a,
                    "prediction": pred_a,
                    "retrieval_status": st_a,
                    "top_chunks": top_chunks_preview(
                        res_a, top_k=top_k_chunks, preview_chars=chunk_preview_chars
                    ),
                },
                "run_b": {
                    "ndcg_at_10": nd_b,
                    "f1": f1_b,
                    "em": em_b,
                    "prediction": pred_b,
                    "retrieval_status": st_b,
                    "top_chunks": top_chunks_preview(
                        res_b, top_k=top_k_chunks, preview_chars=chunk_preview_chars
                    ),
                },
                "delta_ndcg_at_10": d_nd,
                "delta_f1": d_f1,
                "extras": {"top10_chunk_id_jaccard": jid},
            }
            interesting.append(row_out)

    interesting.sort(
        key=lambda r: (
            -float(r["delta_ndcg_at_10"]),
            float(r["delta_f1"]),
            r["question_id"],
        )
    )
    counts.interesting = len(interesting)
    return interesting, counts


def load_split_test_question_ids(path: Path) -> set[str]:
    """Load router ``split_question_ids.json`` ``test`` list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("test") or []
    return {str(qid).strip() for qid in rows if str(qid).strip()}


def assert_reranker_none_or_allow_ce(
    run_root_a: Path,
    run_root_b: Path,
    *,
    allow_cross_encoder: bool,
) -> None:
    """Unless ``allow_cross_encoder``, require ``e2e.reranker`` in {none,''} for both runs."""
    if allow_cross_encoder:
        return
    for tag, root in (("run_a", run_root_a), ("run_b", run_root_b)):
        blk = read_e2e_manifest_block(root)
        if blk and not reranker_is_none_or_disabled(blk):
            rk = blk.get("reranker")
            raise ValueError(
                f"{tag}: manifest e2e.reranker is {rk!r}; "
                f"omit cross-encoder runs or pass allow_cross_encoder=True."
            )


def benchmark_id_conflict_message(
    excerpt_a: Mapping[str, Any], excerpt_b: Mapping[str, Any]
) -> str | None:
    id_a = excerpt_a.get("benchmark_id")
    id_b = excerpt_b.get("benchmark_id")
    if id_a is None or id_b is None:
        return None
    if str(id_a) != str(id_b):
        return (
            "e2e.benchmark_id differs between runs "
            f"({id_a!r} vs {id_b!r}); ensure benchmark JSONL aligns."
        )
    return None


def _question_preview(text: Any, *, max_chars: int) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def write_discrepancy_bundle(
    out_dir: Path,
    *,
    test_id: str,
    benchmark_path: Path,
    run_root_a: Path,
    run_root_b: Path,
    e2e_a: Mapping[str, Any],
    e2e_b: Mapping[str, Any],
    rule: SelectionRule,
    interesting_rows: Sequence[MutableMapping[str, Any]],
    counts: ComparisonCounts,
    markdown_max_rows: int,
    comparison_resolution: Mapping[str, Any] | None = None,
) -> Tuple[Path, Path, Path]:
    """Writes manifest.json, interesting.jsonl, interesting.md under ``out_dir``."""
    out_dir = as_resolved_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bm_res = benchmark_path.expanduser().resolve()
    root_a = as_resolved_path(Path(run_root_a))
    root_b = as_resolved_path(Path(run_root_b))

    excerpt_a = manifest_excerpt(e2e_a)
    excerpt_b = manifest_excerpt(e2e_b)

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "test_id": str(test_id),
        "selection_rule": {
            "retrieval_metric": "ndcg@10",
            "baseline_run": rule.baseline_label,
            "comparison": "run_b_relative_to_run_a",
            "epsilon_ndcg": rule.epsilon_ndcg,
            "delta_f1": rule.delta_f1,
        },
        "inputs": {
            "benchmark_path": str(bm_res),
            "run_a": {
                "run_root": str(root_a),
                "manifest_excerpt": excerpt_a,
            },
            "run_b": {
                "run_root": str(root_b),
                "manifest_excerpt": excerpt_b,
            },
        },
        "counts": {
            "joined_question_ids": counts.joined_question_ids,
            "interesting": counts.interesting,
            "skipped_missing_retrieval_run_a": counts.skipped_missing_retrieval_a,
            "skipped_missing_retrieval_run_b": counts.skipped_missing_retrieval_b,
            "skipped_missing_answer_run_a": counts.skipped_missing_answer_a,
            "skipped_missing_answer_run_b": counts.skipped_missing_answer_b,
            "skipped_restrict_filter": counts.skipped_restrict_filter,
        },
    }
    if comparison_resolution:
        manifest["inputs"]["comparison_resolution"] = dict(comparison_resolution)
    v = surf_rag_version_string()
    if v:
        manifest["software"] = {"surf_rag": v}
    warn = benchmark_id_conflict_message(excerpt_a, excerpt_b)
    if warn:
        manifest["warnings"] = [warn]
        logger.warning("%s", warn)

    mf_path = out_dir / "manifest.json"
    mf_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    jsonl_path = out_dir / "interesting.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in interesting_rows:
            jf.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    md_path = out_dir / "interesting.md"
    lines: List[str] = [
        f"# Interesting rows (discrepancy debug test `{test_id}`)",
        "",
        f"- Baseline **run_a**: `{root_a}`",
        f"- Candidate **run_b**: `{root_b}`",
        f"- Rule: ΔnDCG@10 ≥ `{rule.epsilon_ndcg}`, F1_b ≤ F1_a + `{rule.delta_f1}`",
        "",
        "| question_id | ΔnDCG@10 | ΔF1 | question (preview) |",
        "|---|---:|---:|---|",
    ]
    preview_q = 80
    for row in interesting_rows[: max(0, markdown_max_rows)]:
        qid = str(row.get("question_id", ""))
        dnd = row.get("delta_ndcg_at_10", "")
        df1 = row.get("delta_f1", "")
        pq = _question_preview(row.get("question", ""), max_chars=preview_q)
        esc = pq.replace("|", "\\|")
        lines.append(f"| `{qid}` | {dnd} | {df1} | {esc} |")
    if len(interesting_rows) > markdown_max_rows:
        lines.append("")
        lines.append(
            f"_(truncated markdown table to {markdown_max_rows} rows; full data in interesting.jsonl)_"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return mf_path, jsonl_path, md_path
