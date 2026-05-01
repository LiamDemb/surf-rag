"""End-to-end prepare: routed retrieval, optional rerank, batch JSONL + manifest."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from surf_rag.config.schema import PipelineConfig

from surf_rag.core.prompts import get_generator_prompt
from surf_rag.evaluation.artifact_paths import (
    default_benchmark_base,
    default_router_base,
    e2e_policy_run_dir,
)
from surf_rag.evaluation.e2e_aggregation import (
    aggregate_e2e_report,
    aggregate_per_question,
    load_benchmark_index,
)
from surf_rag.evaluation.latency_metrics import (
    LATENCY_PROTOCOL_VERSION,
    canonicalize_latency_ms,
)
from surf_rag.evaluation.e2e_policies import (
    e2e_pipeline_manifest_name,
    parse_routing_policy,
)
from surf_rag.evaluation.manifest import update_manifest_artifacts, write_manifest
from surf_rag.evaluation.retrieval_jsonl import (
    dict_to_retrieval_result,
    write_retrieval_line,
)
from surf_rag.evaluation.router_overlap import RouterSplitSets
from surf_rag.evaluation.run_artifacts import (
    RunArtifactPaths,
    make_generation_custom_id,
)
from surf_rag.generation.batch import build_completion_body
from surf_rag.generation.batch_compiler import BatchRequestRecord
from surf_rag.generation.batch_orchestrator import (
    _load_completed_question_ids_from_answers,
    finalize_batch_submission,
    finalize_sync_submission,
)
from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.reranking.reranker import build_reranker
from surf_rag.retrieval.routed import RoutedFusionPipeline
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.router.inference_inputs import (
    compute_query_tensors_for_router,
    load_router_inference_context,
)
from surf_rag.router.policies import RoutingPolicyName
from surf_rag.strategies.factory import build_dense_retriever, build_graph_retriever

logger = logging.getLogger(__name__)
_BASE_METRIC_KS: tuple[int, ...] = (5, 10, 20)


class _UnusedRetriever:
    """Guard retriever to catch unexpected branch execution."""

    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, query: str, **kwargs: object):  # pragma: no cover - defensive
        raise RuntimeError(f"{self.name} retriever should not be used for this policy")


def _load_benchmark(path: Path, limit: int | None) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def _load_retrieval_rows(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
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


def _load_answers_rows(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
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


def make_e2e_run_paths(
    *,
    benchmark_base: Path,
    benchmark_name: str,
    benchmark_id: str,
    policy: RoutingPolicyName,
    run_id: str,
) -> RunArtifactPaths:
    run_root = e2e_policy_run_dir(
        benchmark_base, benchmark_name, benchmark_id, policy.value, run_id
    )
    return RunArtifactPaths(run_root=run_root)


def evaluate_e2e_run(
    *,
    run_paths: RunArtifactPaths,
    benchmark_path: Path,
    split_question_ids_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Load retrieval + answers under ``run_paths``; return overlap report + per-row list."""
    from surf_rag.evaluation.retrieval_metrics import DEFAULT_NDCG_KS

    bench = load_benchmark_index(benchmark_path)
    split_sets: RouterSplitSets | None = RouterSplitSets.from_path_or_default(
        split_question_ids_path
    )

    retrieval_path = run_paths.retrieval_results_jsonl()
    retrieval_pretrunc_path = run_paths.retrieval_results_pretrunc_jsonl()
    answers_path = run_paths.generation_answers_jsonl()
    manifest: dict[str, Any] = {}
    e2e_meta: dict[str, Any] = {}
    if run_paths.manifest.is_file():
        try:
            manifest = json.loads(run_paths.manifest.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                e2e_meta = dict(manifest.get("e2e") or {})
        except Exception:
            manifest = {}
            e2e_meta = {}

    reranker_kind = str(e2e_meta.get("reranker", "") or "").strip().lower()
    try:
        rerank_top_k = int(e2e_meta.get("rerank_top_k", 0) or 0)
    except Exception:
        rerank_top_k = 0

    retrieval_by_qid = _load_retrieval_rows(retrieval_path)
    pretrunc_by_qid = _load_retrieval_rows(retrieval_pretrunc_path)
    answers_by_qid = _load_answers_rows(answers_path)
    all_qids = sorted(set(bench) | set(retrieval_by_qid) | set(answers_by_qid))

    rows_out: list[dict[str, Any]] = []
    eval_rows_generation = []
    eval_rows_pure = []
    eval_rows_post_ce = []
    overlap_keys = ("all", "train", "dev", "test", "unseen")
    pure_available = retrieval_pretrunc_path.is_file()
    ce_enabled = reranker_kind in ("cross_encoder", "cross-encoder", "ce")
    ce_ks = [int(k) for k in DEFAULT_NDCG_KS if int(k) <= max(0, rerank_top_k)]

    for qid in all_qids:
        sample = bench.get(qid, {})
        gold_sents = list(sample.get("gold_support_sentences") or [])
        ds = sample.get("dataset_source")
        gold_ans = list(sample.get("gold_answers") or [])
        pred = str((answers_by_qid.get(qid) or {}).get("answer", "") or "")

        rj = retrieval_by_qid.get(qid)
        if rj:
            rr_gen = dict_to_retrieval_result(rj)
        else:
            rr_gen = RetrievalResult(
                query=str(sample.get("question", "") or ""),
                retriever_name="missing",
                status="EMPTY",
                chunks=[],
                latency_ms={},
                error="no_retrieval_row",
            )
        if "retrieval_stage_total_ms" not in rr_gen.latency_ms:
            rr_gen.latency_ms = canonicalize_latency_ms(
                retriever_name=rr_gen.retriever_name,
                latency_ms=rr_gen.latency_ms,
                routing_input_ms=float(
                    rr_gen.latency_ms.get("routing_input_ms", 0.0) or 0.0
                ),
            )
        pe_gen = aggregate_per_question(
            qid,
            result=rr_gen,
            gold_support_sentences=gold_sents,
            dataset_source=str(ds) if ds else None,
            gold_answers=gold_ans,
            prediction=pred,
            ks=DEFAULT_NDCG_KS,
        )
        eval_rows_generation.append(pe_gen)

        q_row: dict[str, Any] = {
            "question_id": qid,
            "qa": {"em": pe_gen.em, "f1": pe_gen.f1, "prediction": pred},
            "latency_ms": dict(rr_gen.latency_ms),
        }

        if pure_available:
            rj_pre = pretrunc_by_qid.get(qid)
            if rj_pre:
                rr_pre = dict_to_retrieval_result(rj_pre)
            else:
                rr_pre = RetrievalResult(
                    query=str(sample.get("question", "") or ""),
                    retriever_name="missing",
                    status="EMPTY",
                    chunks=[],
                    latency_ms={},
                    error="no_pretrunc_retrieval_row",
                )
            pe_pre = aggregate_per_question(
                qid,
                result=rr_pre,
                gold_support_sentences=gold_sents,
                dataset_source=str(ds) if ds else None,
                gold_answers=gold_ans,
                prediction=pred,
                ks=_BASE_METRIC_KS,
            )
            eval_rows_pure.append(pe_pre)
            q_row["retrieval_before_ce"] = {
                "status": "ok",
                "reported_ks": list(_BASE_METRIC_KS),
                "retrieval": {str(s.k): s.to_json() for s in pe_pre.retrieval_suites},
            }
        else:
            q_row["retrieval_before_ce"] = {
                "status": "unavailable",
                "reason": "missing retrieval/retrieval_results_pretrunc.jsonl",
                "reported_ks": list(_BASE_METRIC_KS),
            }

        if ce_enabled:
            if ce_ks:
                pe_ce = aggregate_per_question(
                    qid,
                    result=rr_gen,
                    gold_support_sentences=gold_sents,
                    dataset_source=str(ds) if ds else None,
                    gold_answers=gold_ans,
                    prediction=pred,
                    ks=ce_ks,
                )
                eval_rows_post_ce.append(pe_ce)
                q_row["retrieval_after_ce"] = {
                    "status": "ok",
                    "reported_ks": ce_ks,
                    "retrieval": {
                        str(s.k): s.to_json() for s in pe_ce.retrieval_suites
                    },
                }
            else:
                q_row["retrieval_after_ce"] = {
                    "status": "unavailable",
                    "reason": (
                        f"rerank_top_k={rerank_top_k} is below smallest metric k="
                        f"{min(_BASE_METRIC_KS)}"
                    ),
                    "reported_ks": [],
                }
        else:
            q_row["retrieval_after_ce"] = {
                "status": "not_applicable",
                "reason": f"reranker={reranker_kind or 'none'}",
                "reported_ks": [],
            }

        rows_out.append(q_row)

    report_generation = aggregate_e2e_report(
        eval_rows_generation, split_sets=split_sets, ks=DEFAULT_NDCG_KS
    )
    report_pure = (
        aggregate_e2e_report(eval_rows_pure, split_sets=split_sets, ks=_BASE_METRIC_KS)
        if pure_available
        else None
    )
    report_ce = (
        aggregate_e2e_report(eval_rows_post_ce, split_sets=split_sets, ks=ce_ks)
        if (ce_enabled and ce_ks)
        else None
    )
    overlap_breakdown: dict[str, Any] = {}
    for key in overlap_keys:
        gen_block = report_generation.get(key, {})
        out_block: dict[str, Any] = {
            "count": int(gen_block.get("count", 0)),
            "latency_ms": dict(gen_block.get("latency_ms") or {}),
            "qa": dict(gen_block.get("qa") or {}),
        }
        if report_pure is not None:
            pure_block = dict((report_pure.get(key) or {}).get("retrieval_at_k") or {})
            out_block["retrieval_before_ce"] = {
                "status": "ok",
                "reported_ks": list(_BASE_METRIC_KS),
                "retrieval_at_k": pure_block,
            }
        else:
            out_block["retrieval_before_ce"] = {
                "status": "unavailable",
                "reason": "missing retrieval/retrieval_results_pretrunc.jsonl",
                "reported_ks": list(_BASE_METRIC_KS),
            }
        if report_ce is not None:
            ce_block = dict((report_ce.get(key) or {}).get("retrieval_at_k") or {})
            out_block["retrieval_after_ce"] = {
                "status": "ok",
                "reported_ks": list(ce_ks),
                "retrieval_at_k": ce_block,
            }
        elif ce_enabled and not ce_ks:
            out_block["retrieval_after_ce"] = {
                "status": "unavailable",
                "reason": (
                    f"rerank_top_k={rerank_top_k} is below smallest metric k="
                    f"{min(_BASE_METRIC_KS)}"
                ),
                "reported_ks": [],
            }
        else:
            out_block["retrieval_after_ce"] = {
                "status": "not_applicable",
                "reason": f"reranker={reranker_kind or 'none'}",
                "reported_ks": [],
            }
        overlap_breakdown[str(key)] = out_block

    startup_latency: dict[str, Any] = {}
    if isinstance(e2e_meta, dict):
        startup_latency = dict(e2e_meta.get("startup_latency_ms") or {})
    return {
        "run_root": str(run_paths.run_root.resolve()),
        "split_question_ids": (
            str(split_question_ids_path) if split_question_ids_path else None
        ),
        "latency_protocol_version": LATENCY_PROTOCOL_VERSION,
        "startup_latency_ms": startup_latency,
        "overlap_breakdown": overlap_breakdown,
        "per_question": rows_out,
    }


def e2e_prepare_and_submit(
    benchmark_path: Path,
    *,
    benchmark_base: Path,
    benchmark_name: str,
    benchmark_id: str,
    split: str,
    run_id: str,
    routing_policy: str | RoutingPolicyName,
    retrieval_asset_dir: Path,
    router_id: Optional[str] = None,
    router_base: Optional[Path] = None,
    fusion_keep_k: int = 25,
    reranker_kind: str = "none",
    rerank_top_k: int = 10,
    cross_encoder_model: Optional[str] = None,
    limit: Optional[int] = None,
    only_question_ids: Optional[Set[str]] = None,
    completion_window: str = "24h",
    include_graph_provenance: bool = False,
    dry_run: bool = False,
    router_device: str = "cpu",
    router_input_mode: str = "both",
    router_inference_batch_size: int = 32,
    latency_warmup_questions: int = 0,
    dev_sync: bool = False,
    pipeline_config_for_artifact: Optional["PipelineConfig"] = None,
    run_paths_override: Optional[RunArtifactPaths] = None,
) -> int:
    """Routed fusion retrieval + optional rerank + OpenAI batch submission.

    When ``run_paths_override`` is set, artifacts are written there instead of under
    ``evaluations/<policy>/<run_id>/`` (used by graph retrieval grid search).
    """
    policy = (
        routing_policy
        if isinstance(routing_policy, RoutingPolicyName)
        else parse_routing_policy(routing_policy)
    )
    pipeline_name = e2e_pipeline_manifest_name(policy)

    if policy in (RoutingPolicyName.LEARNED_SOFT, RoutingPolicyName.LEARNED_HARD):
        if not router_id or not str(router_id).strip():
            raise ValueError("router_id is required for learned routing policies")

    paths = (
        run_paths_override
        if run_paths_override is not None
        else make_e2e_run_paths(
            benchmark_base=benchmark_base,
            benchmark_name=benchmark_name,
            benchmark_id=benchmark_id,
            policy=policy,
            run_id=run_id,
        )
    )
    paths.ensure_dirs()
    run_root = paths.run_root.resolve()
    asset_dir = retrieval_asset_dir.resolve()

    if not dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set (required unless dry_run).")
        return 1

    effective_limit = None if only_question_ids else limit
    samples = _load_benchmark(benchmark_path, limit=effective_limit)
    if only_question_ids:
        before = len(samples)
        samples = [
            s
            for s in samples
            if str(s.get("question_id", "")).strip() in only_question_ids
        ]
        logger.info(
            "Filtered benchmark %d → %d rows for only_question_ids.",
            before,
            len(samples),
        )
    if not samples:
        logger.error("No benchmark samples to process.")
        return 1

    answers_path = paths.generation_answers_jsonl()
    completed_qids = _load_completed_question_ids_from_answers(answers_path)

    base_prompt = get_generator_prompt()
    model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("GENERATOR_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("GENERATOR_MAX_TOKENS", "512"))
    renderer = PromptRenderer(
        base_prompt=base_prompt,
        include_graph_provenance=include_graph_provenance,
    )

    startup_t0 = time.perf_counter()
    startup_components: dict[str, float] = {}

    need_dense = policy in (
        RoutingPolicyName.DENSE_ONLY,
        RoutingPolicyName.EQUAL_50_50,
        RoutingPolicyName.LEARNED_SOFT,
        RoutingPolicyName.LEARNED_HARD,
    )
    need_graph = policy in (
        RoutingPolicyName.GRAPH_ONLY,
        RoutingPolicyName.EQUAL_50_50,
        RoutingPolicyName.LEARNED_SOFT,
        RoutingPolicyName.LEARNED_HARD,
    )

    if need_dense:
        t_dense = time.perf_counter()
        dense_retriever = build_dense_retriever(str(asset_dir))
        startup_components["dense_init_ms"] = (time.perf_counter() - t_dense) * 1000.0
    else:
        dense_retriever = _UnusedRetriever("Dense")

    if need_graph:
        t_graph = time.perf_counter()
        graph_retriever = build_graph_retriever(
            str(asset_dir),
            pipeline_config=pipeline_config_for_artifact,
        )
        startup_components["graph_init_ms"] = (time.perf_counter() - t_graph) * 1000.0
    else:
        graph_retriever = _UnusedRetriever("Graph")

    loaded_router = None
    router_ctx = None
    rb = router_base if router_base is not None else default_router_base()
    if policy in (RoutingPolicyName.LEARNED_SOFT, RoutingPolicyName.LEARNED_HARD):
        t_router = time.perf_counter()
        router_ctx = load_router_inference_context(
            str(router_id),
            input_mode=router_input_mode,
            router_base=rb,
            retrieval_asset_dir=asset_dir,
            device=router_device,
        )
        startup_components["router_init_ms"] = (time.perf_counter() - t_router) * 1000.0
        loaded_router = router_ctx.router

    startup_total_ms = (time.perf_counter() - startup_t0) * 1000.0

    pipeline = RoutedFusionPipeline(
        dense_retriever,
        graph_retriever,
        fusion_keep_k=fusion_keep_k,
        router=loaded_router,
    )
    reranker = build_reranker(reranker_kind, cross_encoder_model=cross_encoder_model)

    write_manifest(
        paths,
        run_id=run_id,
        benchmark=benchmark_name,
        split=split,
        pipeline_name=pipeline_name,
        retrieval_asset_dir=str(asset_dir),
        generator_model=model_id,
        include_graph_provenance=include_graph_provenance,
        completion_window=completion_window,
        artifact_paths={
            "retrieval_results": str(
                paths.retrieval_results_jsonl().relative_to(run_root)
            ),
            "retrieval_results_pretrunc": str(
                paths.retrieval_results_pretrunc_jsonl().relative_to(run_root)
            ),
            "batch_input": str(paths.batch_input_jsonl().relative_to(run_root)),
            "batch_state": str(paths.batch_state_json().relative_to(run_root)),
            "generation_answers": str(
                paths.generation_answers_jsonl().relative_to(run_root)
            ),
        },
        extra={
            "e2e": {
                "schema": "surf-rag/e2e/v1",
                "benchmark_id": benchmark_id,
                "routing_policy": policy.value,
                "fusion_keep_k": fusion_keep_k,
                "reranker": reranker_kind,
                "rerank_top_k": rerank_top_k,
                "retrieval_metric_base_ks": list(_BASE_METRIC_KS),
                "router_id": router_id,
                "router_input_mode": router_input_mode,
                "router_inference_batch_size": router_inference_batch_size,
                "latency_protocol": {
                    "version": LATENCY_PROTOCOL_VERSION,
                    "included_components": [
                        "routing_input",
                        "router_predict",
                        "branch_retrieval",
                        "fusion",
                    ],
                    "excluded_components": [
                        "startup",
                        "warmup",
                        "reranker",
                        "generation",
                    ],
                    "warmup_questions": int(max(0, latency_warmup_questions)),
                },
                "startup_latency_ms": {
                    "startup_total_ms": float(startup_total_ms),
                    "startup_components": startup_components,
                },
            },
        },
    )

    if pipeline_config_for_artifact is not None:
        from surf_rag.config.loader import resolve_paths
        from surf_rag.config.resolved import write_resolved_config_yaml

        rp = resolve_paths(pipeline_config_for_artifact)
        write_resolved_config_yaml(
            run_root / "resolved_config.yaml",
            pipeline_config_for_artifact,
            rp,
        )
        from surf_rag.evaluation.manifest import update_manifest_artifacts

        update_manifest_artifacts(paths, {"resolved_config": "resolved_config.yaml"})

    records: List[BatchRequestRecord] = []
    pending = [
        s
        for s in samples
        if s.get("question", "").strip()
        and str(s.get("question_id", "")).strip() not in completed_qids
    ]
    progress = (
        tqdm(total=len(pending), desc="E2E retrieval + batch", unit="q")
        if pending
        else None
    )

    warmup_n = int(max(0, latency_warmup_questions))
    if warmup_n > 0 and pending:
        warmup_samples = pending[: min(len(pending), warmup_n)]
        for sample in warmup_samples:
            question = str(sample.get("question", "") or "").strip()
            if not question:
                continue
            q_emb = feat = None
            if router_ctx is not None:
                q_emb, feat = compute_query_tensors_for_router(question, router_ctx)
            pipeline.run(
                question,
                policy,
                query_embedding=q_emb,
                feature_vector=feat,
            )

    retrieval_fp = paths.retrieval_results_jsonl().open("a", encoding="utf-8")
    retrieval_pretrunc_fp = paths.retrieval_results_pretrunc_jsonl().open(
        "a", encoding="utf-8"
    )
    skipped = 0
    try:
        for sample in samples:
            question = sample.get("question", "").strip()
            qid = str(sample.get("question_id", "")).strip()
            if not question:
                continue
            if qid in completed_qids:
                skipped += 1
                continue

            routing_input_ms = 0.0
            q_emb = feat = None
            if router_ctx is not None:
                rt0 = time.perf_counter()
                q_emb, feat = compute_query_tensors_for_router(question, router_ctx)
                routing_input_ms = (time.perf_counter() - rt0) * 1000.0

            routed = pipeline.run_with_pretrunc(
                question,
                policy,
                query_embedding=q_emb,
                feature_vector=feat,
            )
            rr_pre = routed.pretrunc_result
            rr_pre.latency_ms = canonicalize_latency_ms(
                retriever_name=rr_pre.retriever_name,
                latency_ms=rr_pre.latency_ms,
                routing_input_ms=routing_input_ms,
            )
            rr = routed.generation_result
            rr.latency_ms = canonicalize_latency_ms(
                retriever_name=rr.retriever_name,
                latency_ms=rr.latency_ms,
                routing_input_ms=routing_input_ms,
            )
            rr = reranker.rerank(question, rr, top_k=rerank_top_k)

            write_retrieval_line(retrieval_pretrunc_fp, rr_pre, qid)
            write_retrieval_line(retrieval_fp, rr, qid)

            custom_id = make_generation_custom_id(
                run_id, benchmark_name, split, pipeline_name, qid
            )
            messages = renderer.to_messages(question, rr)
            body = build_completion_body(
                messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            records.append(BatchRequestRecord(custom_id=custom_id, body=body))
            if progress is not None:
                progress.update(1)
    finally:
        retrieval_fp.close()
        retrieval_pretrunc_fp.close()
        if progress is not None:
            progress.close()

    if skipped:
        logger.info("Skipped %d questions with existing answers.", skipped)

    if dev_sync:
        if dry_run:
            logger.warning("--dev-sync with --dry-run skips sync API calls.")
        code = (
            finalize_batch_submission(
                records,
                samples,
                benchmark_path=benchmark_path,
                paths=paths,
                benchmark=benchmark_name,
                split=split,
                pipeline_name=pipeline_name,
                run_id=run_id,
                retrieval_asset_dir=asset_dir,
                completion_window=completion_window,
                dry_run=True,
            )
            if dry_run
            else finalize_sync_submission(
                records,
                samples,
                benchmark_path=benchmark_path,
                paths=paths,
                benchmark=benchmark_name,
                split=split,
                pipeline_name=pipeline_name,
                run_id=run_id,
                retrieval_asset_dir=asset_dir,
            )
        )
    else:
        code = finalize_batch_submission(
            records,
            samples,
            benchmark_path=benchmark_path,
            paths=paths,
            benchmark=benchmark_name,
            split=split,
            pipeline_name=pipeline_name,
            run_id=run_id,
            retrieval_asset_dir=asset_dir,
            completion_window=completion_window,
            dry_run=dry_run,
        )
    return code
