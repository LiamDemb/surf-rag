"""Compare RRF vs learned-soft retrieval; export questions where RRF wins on nDCG@k."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from surf_rag.config.loader import e2e_run_root
from surf_rag.config.schema import PipelineConfig
from surf_rag.evaluation.discrepancy_debug import (
    chunk_ids_top_k,
    jaccard_chunk_ids,
    load_benchmark_index,
    load_retrieval_by_qid,
    read_e2e_manifest_block,
    surf_rag_version_string,
    top_chunks_preview,
)
from surf_rag.evaluation.manifest import utc_now_iso
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.evaluation.run_artifacts import as_resolved_path
from surf_rag.results.loaders import load_split_qids

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "surf-rag/policy-retrieval-compare/v1"
POLICY_RRF = "rrf"
POLICY_LEARNED_SOFT = "learned-soft"

RETRIEVAL_ARTIFACT_PRETRUNC = "pretrunc"
RETRIEVAL_ARTIFACT_FINAL = "final"
RETRIEVAL_ARTIFACT_FILES = {
    RETRIEVAL_ARTIFACT_PRETRUNC: "retrieval_results_pretrunc.jsonl",
    RETRIEVAL_ARTIFACT_FINAL: "retrieval_results.jsonl",
}


@dataclass(frozen=True)
class PolicyRetrievalCompareConfig:
    """Resolved inputs for one RRF vs learned-soft comparison run."""

    metric_k: int = 10
    epsilon_ndcg: float = 1e-9
    top_k_chunks: int = 10
    chunk_preview_chars: int = 50
    retrieval_artifact: str = RETRIEVAL_ARTIFACT_PRETRUNC
    restrict_split: str | None = None
    output_root: Path = Path("temp/policy-retrieval-compare")
    output_id: str = "default"


@dataclass
class PolicyCompareCounts:
    evaluated_question_ids: int = 0
    rrf_wins: int = 0
    learned_soft_wins: int = 0
    ties: int = 0
    skipped_restrict_filter: int = 0
    skipped_missing_rrf_retrieval: int = 0
    skipped_missing_learned_soft_retrieval: int = 0


def resolve_policy_run_ids(
    cfg: PipelineConfig,
    *,
    run_id: str | None = None,
    run_id_rrf: str | None = None,
    run_id_learned_soft: str | None = None,
) -> Tuple[str, str]:
    """Resolve E2E run ids for RRF and learned-soft (CLI overrides > results > e2e)."""
    rrf_cli = str(run_id_rrf or "").strip()
    ls_cli = str(run_id_learned_soft or "").strip()
    if rrf_cli and ls_cli:
        return rrf_cli, ls_cli

    shared = str(run_id or "").strip()
    if shared:
        return shared, shared

    policies = cfg.results.policies
    rrf_from_results = (
        str(policies[POLICY_RRF].run_id).strip()
        if POLICY_RRF in policies and policies[POLICY_RRF].run_id
        else ""
    )
    ls_from_results = (
        str(policies[POLICY_LEARNED_SOFT].run_id).strip()
        if POLICY_LEARNED_SOFT in policies and policies[POLICY_LEARNED_SOFT].run_id
        else ""
    )
    if rrf_from_results and ls_from_results and not rrf_cli and not ls_cli:
        return rrf_from_results, ls_from_results

    e2e_rid = str(cfg.e2e.run_id or "").strip()
    rid_rrf = rrf_cli or rrf_from_results or e2e_rid
    rid_ls = ls_cli or ls_from_results or e2e_rid
    if not rid_rrf or not rid_ls:
        raise ValueError(
            "Missing run id: set e2e.run_id, results.policies.rrf/learned-soft.run_id, "
            "or pass --run-id / --run-id-rrf and --run-id-learned-soft."
        )
    return rid_rrf, rid_ls


def retrieval_ndcg_at_k(
    sample: Mapping[str, Any],
    retrieval_row: dict | None,
    *,
    question_id: str,
    k: int,
) -> Tuple[float, str, RetrievalResult]:
    """nDCG@k, retrieval status, and parsed result for one question."""
    from surf_rag.evaluation.discrepancy_debug import _retrieval_from_row, _suite_at_k
    from surf_rag.evaluation.e2e_aggregation import aggregate_per_question

    gold_sents = list(sample.get("gold_support_sentences") or [])
    ds = sample.get("dataset_source")
    rr = _retrieval_from_row(sample, retrieval_row)
    pe = aggregate_per_question(
        question_id,
        result=rr,
        gold_support_sentences=gold_sents,
        dataset_source=str(ds) if ds else None,
        gold_answers=[],
        prediction="",
        ks=(int(k),),
    )
    suite = _suite_at_k(pe.retrieval_suites, int(k))
    return float(suite.ndcg), str(rr.status or ""), rr


def resolve_restrict_split(
    cfg: PipelineConfig,
    *,
    restrict_split: str | None,
    compare_yaml: Mapping[str, Any] | None,
) -> str:
    for candidate in (
        restrict_split,
        (compare_yaml or {}).get("restrict_split"),
    ):
        if candidate is None:
            continue
        s = str(candidate).strip().lower()
        if s:
            return s
    if cfg.results.policies and str(cfg.results.split or "").strip():
        return str(cfg.results.split).strip().lower()
    if str(cfg.e2e.split or "").strip():
        return str(cfg.e2e.split).strip().lower()
    return "all"


def retrieval_jsonl_path(run_root: Path, artifact: str) -> Path:
    key = str(artifact or RETRIEVAL_ARTIFACT_PRETRUNC).strip().lower()
    fname = RETRIEVAL_ARTIFACT_FILES.get(key)
    if not fname:
        choices = ", ".join(sorted(RETRIEVAL_ARTIFACT_FILES))
        raise ValueError(
            f"Unknown retrieval_artifact {artifact!r}; expected one of: {choices}"
        )
    return Path(run_root) / "retrieval" / fname


def extract_rrf_win_rows(
    *,
    bench_by_qid: Mapping[str, Mapping[str, Any]],
    retr_rrf: Mapping[str, dict],
    retr_ls: Mapping[str, dict],
    restrict_qids: Optional[set[str]],
    metric_k: int,
    epsilon_ndcg: float,
    top_k_chunks: int,
    chunk_preview_chars: int,
) -> Tuple[List[Dict[str, Any]], PolicyCompareCounts]:
    """Questions where RRF nDCG@k strictly exceeds learned-soft."""
    counts = PolicyCompareCounts()
    wins: List[Dict[str, Any]] = []

    for qid in sorted(bench_by_qid.keys(), key=lambda x: str(x)):
        sample = bench_by_qid[qid]
        qs = str(qid).strip()
        if restrict_qids is not None and qs not in restrict_qids:
            counts.skipped_restrict_filter += 1
            continue

        row_rrf = retr_rrf.get(qs)
        row_ls = retr_ls.get(qs)
        if row_rrf is None:
            counts.skipped_missing_rrf_retrieval += 1
        if row_ls is None:
            counts.skipped_missing_learned_soft_retrieval += 1
        if row_rrf is None or row_ls is None:
            continue

        counts.evaluated_question_ids += 1
        nd_rrf, st_rrf, res_rrf = retrieval_ndcg_at_k(
            sample, row_rrf, question_id=qs, k=int(metric_k)
        )
        nd_ls, st_ls, res_ls = retrieval_ndcg_at_k(
            sample, row_ls, question_id=qs, k=int(metric_k)
        )

        delta = float(nd_rrf) - float(nd_ls)
        if delta > float(epsilon_ndcg):
            counts.rrf_wins += 1
            jid = jaccard_chunk_ids(
                chunk_ids_top_k(res_rrf, top_k_chunks),
                chunk_ids_top_k(res_ls, top_k_chunks),
            )
            wins.append(
                {
                    "question_id": qs,
                    "dataset_source": sample.get("dataset_source"),
                    "question": sample.get("question", ""),
                    "gold_support_sentences": list(
                        sample.get("gold_support_sentences") or []
                    ),
                    "ndcg_rrf": float(nd_rrf),
                    "ndcg_learned_soft": float(nd_ls),
                    "delta_ndcg": delta,
                    "rrf": {
                        "ndcg_at_k": float(nd_rrf),
                        "retrieval_status": st_rrf,
                        "top_chunks": top_chunks_preview(
                            res_rrf,
                            top_k=top_k_chunks,
                            preview_chars=chunk_preview_chars,
                        ),
                    },
                    "learned_soft": {
                        "ndcg_at_k": float(nd_ls),
                        "retrieval_status": st_ls,
                        "top_chunks": top_chunks_preview(
                            res_ls,
                            top_k=top_k_chunks,
                            preview_chars=chunk_preview_chars,
                        ),
                    },
                    "extras": {"top_k_chunk_id_jaccard": jid},
                }
            )
        elif float(nd_ls) - float(nd_rrf) > float(epsilon_ndcg):
            counts.learned_soft_wins += 1
        else:
            counts.ties += 1

    wins.sort(
        key=lambda r: (
            -float(r["delta_ndcg"]),
            r["question_id"],
        )
    )
    return wins, counts


def _question_preview(text: Any, *, max_chars: int) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def write_rrf_wins_bundle(
    out_dir: Path,
    *,
    output_id: str,
    benchmark_path: Path,
    run_root_rrf: Path,
    run_root_ls: Path,
    run_id_rrf: str,
    run_id_ls: str,
    compare: PolicyRetrievalCompareConfig,
    win_rows: Sequence[MutableMapping[str, Any]],
    counts: PolicyCompareCounts,
    restrict_split: str,
    e2e_rrf: Mapping[str, Any],
    e2e_ls: Mapping[str, Any],
    markdown_max_rows: int = 200,
) -> Tuple[Path, Path, Path]:
    out_dir = as_resolved_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bm_res = benchmark_path.expanduser().resolve()
    root_rrf = as_resolved_path(Path(run_root_rrf))
    root_ls = as_resolved_path(Path(run_root_ls))

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "output_id": str(output_id),
        "comparison": {
            "baseline_policy": POLICY_RRF,
            "comparison_policy": POLICY_LEARNED_SOFT,
            "winner": POLICY_RRF,
            "retrieval_metric": f"ndcg@{compare.metric_k}",
            "epsilon_ndcg": compare.epsilon_ndcg,
            "retrieval_artifact": compare.retrieval_artifact,
        },
        "inputs": {
            "benchmark_path": str(bm_res),
            "restrict_split": restrict_split,
            "run_rrf": {
                "policy": POLICY_RRF,
                "run_id": run_id_rrf,
                "run_root": str(root_rrf),
                "manifest_excerpt": {
                    k: e2e_rrf.get(k)
                    for k in (
                        "router_id",
                        "router_architecture_id",
                        "routing_policy",
                        "reranker",
                        "benchmark_id",
                    )
                    if k in e2e_rrf
                },
            },
            "run_learned_soft": {
                "policy": POLICY_LEARNED_SOFT,
                "run_id": run_id_ls,
                "run_root": str(root_ls),
                "manifest_excerpt": {
                    k: e2e_ls.get(k)
                    for k in (
                        "router_id",
                        "router_architecture_id",
                        "routing_policy",
                        "reranker",
                        "benchmark_id",
                    )
                    if k in e2e_ls
                },
            },
        },
        "counts": {
            "evaluated_question_ids": counts.evaluated_question_ids,
            "rrf_wins": counts.rrf_wins,
            "learned_soft_wins": counts.learned_soft_wins,
            "ties": counts.ties,
            "skipped_restrict_filter": counts.skipped_restrict_filter,
            "skipped_missing_rrf_retrieval": counts.skipped_missing_rrf_retrieval,
            "skipped_missing_learned_soft_retrieval": (
                counts.skipped_missing_learned_soft_retrieval
            ),
        },
    }
    v = surf_rag_version_string()
    if v:
        manifest["software"] = {"surf_rag": v}

    mf_path = out_dir / "manifest.json"
    mf_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    jsonl_path = out_dir / "rrf_wins.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in win_rows:
            jf.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    md_path = out_dir / "rrf_wins.md"
    lines: List[str] = [
        f"# RRF wins over learned-soft (`{output_id}`)",
        "",
        f"- **RRF run**: `{root_rrf}`",
        f"- **learned-soft run**: `{root_ls}`",
        f"- **Rule**: nDCG@{compare.metric_k}(rrf) > nDCG@{compare.metric_k}(learned-soft) + `{compare.epsilon_ndcg}`",
        f"- **Split filter**: `{restrict_split}`",
        f"- **Rows**: {len(win_rows)} wins / {counts.evaluated_question_ids} evaluated",
        "",
        "| question_id | ΔnDCG | nDCG rrf | nDCG ls | Jaccard@k | question (preview) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    preview_q = 80
    for row in win_rows[: max(0, markdown_max_rows)]:
        qid = str(row.get("question_id", ""))
        dnd = row.get("delta_ndcg", "")
        nd_r = (row.get("rrf") or {}).get("ndcg_at_k", "")
        nd_l = (row.get("learned_soft") or {}).get("ndcg_at_k", "")
        jac = (row.get("extras") or {}).get("top_k_chunk_id_jaccard", "")
        pq = _question_preview(row.get("question", ""), max_chars=preview_q)
        esc = pq.replace("|", "\\|")
        lines.append(f"| `{qid}` | {dnd} | {nd_r} | {nd_l} | {jac} | {esc} |")
    if len(win_rows) > markdown_max_rows:
        lines.append("")
        lines.append(
            f"_(markdown table truncated to {markdown_max_rows} rows; "
            "see rrf_wins.jsonl for chunk previews)_"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mf_path, jsonl_path, md_path


def run_policy_retrieval_compare(
    cfg: PipelineConfig,
    *,
    benchmark_path: Path,
    split_question_ids_path: Path,
    run_id: str | None = None,
    run_id_rrf: str | None = None,
    run_id_learned_soft: str | None = None,
    compare: PolicyRetrievalCompareConfig,
    compare_yaml: Mapping[str, Any] | None = None,
    markdown_max_rows: int = 200,
) -> Tuple[Path, PolicyCompareCounts, int]:
    """Execute comparison; return output directory, counts, and number of win rows."""
    rid_rrf, rid_ls = resolve_policy_run_ids(
        cfg,
        run_id=run_id,
        run_id_rrf=run_id_rrf,
        run_id_learned_soft=run_id_learned_soft,
    )
    restrict_split = resolve_restrict_split(
        cfg,
        restrict_split=compare.restrict_split,
        compare_yaml=compare_yaml,
    )
    restrict_qids: set[str] | None
    if restrict_split == "all":
        restrict_qids = None
    else:
        restrict_qids = load_split_qids(split_question_ids_path, restrict_split)

    root_rrf = e2e_run_root(cfg, policy_value=POLICY_RRF, run_id=rid_rrf)
    root_ls = e2e_run_root(cfg, policy_value=POLICY_LEARNED_SOFT, run_id=rid_ls)

    retr_path_rrf = retrieval_jsonl_path(root_rrf, compare.retrieval_artifact)
    retr_path_ls = retrieval_jsonl_path(root_ls, compare.retrieval_artifact)
    if not retr_path_rrf.is_file():
        raise FileNotFoundError(f"RRF retrieval not found: {retr_path_rrf}")
    if not retr_path_ls.is_file():
        raise FileNotFoundError(f"learned-soft retrieval not found: {retr_path_ls}")

    bench = load_benchmark_index(benchmark_path)
    retr_rrf = load_retrieval_by_qid(retr_path_rrf)
    retr_ls = load_retrieval_by_qid(retr_path_ls)

    win_rows, counts = extract_rrf_win_rows(
        bench_by_qid=bench,
        retr_rrf=retr_rrf,
        retr_ls=retr_ls,
        restrict_qids=restrict_qids,
        metric_k=compare.metric_k,
        epsilon_ndcg=compare.epsilon_ndcg,
        top_k_chunks=compare.top_k_chunks,
        chunk_preview_chars=compare.chunk_preview_chars,
    )

    out_dir = (
        compare.output_root.expanduser().resolve() / str(compare.output_id).strip()
    )
    write_rrf_wins_bundle(
        out_dir,
        output_id=compare.output_id,
        benchmark_path=benchmark_path,
        run_root_rrf=root_rrf,
        run_root_ls=root_ls,
        run_id_rrf=rid_rrf,
        run_id_ls=rid_ls,
        compare=compare,
        win_rows=win_rows,
        counts=counts,
        restrict_split=restrict_split,
        e2e_rrf=read_e2e_manifest_block(root_rrf),
        e2e_ls=read_e2e_manifest_block(root_ls),
        markdown_max_rows=int(markdown_max_rows),
    )
    return out_dir, counts, len(win_rows)
