"""End-to-end prepare: routed retrieval, optional rerank, batch JSONL + manifest."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm.auto import tqdm

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
)
from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.reranking.reranker import build_reranker
from surf_rag.retrieval.routed import RoutedFusionPipeline
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.router.inference_inputs import (
    compute_query_tensors_for_router_batch,
    load_router_inference_context,
)
from surf_rag.router.policies import RoutingPolicyName
from surf_rag.strategies.factory import build_dense_retriever, build_graph_retriever

logger = logging.getLogger(__name__)


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
    answers_path = run_paths.generation_answers_jsonl()
    rows_out: list[dict[str, Any]] = []
    eval_rows = []

    retrieval_by_qid: dict[str, dict] = {}
    if retrieval_path.is_file():
        with retrieval_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                rj = json.loads(line)
                qid = str(rj.get("question_id", "") or "").strip()
                if qid:
                    retrieval_by_qid[qid] = rj

    answers_by_qid: dict[str, dict] = {}
    if answers_path.is_file():
        with answers_path.open("r", encoding="utf-8") as af:
            for line in af:
                line = line.strip()
                if not line:
                    continue
                aj = json.loads(line)
                qid = str(aj.get("question_id", "") or "").strip()
                if qid:
                    answers_by_qid[qid] = aj

    all_qids = sorted(set(bench) | set(retrieval_by_qid) | set(answers_by_qid))
    for qid in all_qids:
        sample = bench.get(qid, {})
        gold_sents = list(sample.get("gold_support_sentences") or [])
        ds = sample.get("dataset_source")
        gold_ans = list(sample.get("gold_answers") or [])
        pred = str((answers_by_qid.get(qid) or {}).get("answer", "") or "")

        rj = retrieval_by_qid.get(qid)
        if rj:
            rr = dict_to_retrieval_result(rj)
        else:
            rr = RetrievalResult(
                query=str(sample.get("question", "") or ""),
                retriever_name="missing",
                status="EMPTY",
                chunks=[],
                latency_ms={},
                error="no_retrieval_row",
            )

        pe = aggregate_per_question(
            qid,
            result=rr,
            gold_support_sentences=gold_sents,
            dataset_source=str(ds) if ds else None,
            gold_answers=gold_ans,
            prediction=pred,
            ks=DEFAULT_NDCG_KS,
        )
        eval_rows.append(pe)
        rows_out.append(
            {
                "question_id": qid,
                "retrieval": {str(s.k): s.to_json() for s in pe.retrieval_suites},
                "qa": {"em": pe.em, "f1": pe.f1, "prediction": pred},
            }
        )

    report = aggregate_e2e_report(eval_rows, split_sets=split_sets, ks=DEFAULT_NDCG_KS)
    return {
        "run_root": str(run_paths.run_root.resolve()),
        "split_question_ids": (
            str(split_question_ids_path) if split_question_ids_path else None
        ),
        "overlap": report,
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
) -> int:
    """Routed fusion retrieval + optional rerank + OpenAI batch submission."""
    policy = (
        routing_policy
        if isinstance(routing_policy, RoutingPolicyName)
        else parse_routing_policy(routing_policy)
    )
    pipeline_name = e2e_pipeline_manifest_name(policy)

    if policy in (RoutingPolicyName.LEARNED_SOFT, RoutingPolicyName.LEARNED_HARD):
        if not router_id or not str(router_id).strip():
            raise ValueError("router_id is required for learned routing policies")

    paths = make_e2e_run_paths(
        benchmark_base=benchmark_base,
        benchmark_name=benchmark_name,
        benchmark_id=benchmark_id,
        policy=policy,
        run_id=run_id,
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

    logger.info("Building dense + graph retrievers from %s", asset_dir)
    dense_retriever = build_dense_retriever(str(asset_dir))
    graph_retriever = build_graph_retriever(str(asset_dir))

    loaded_router = None
    router_ctx = None
    rb = router_base if router_base is not None else default_router_base()
    if policy in (RoutingPolicyName.LEARNED_SOFT, RoutingPolicyName.LEARNED_HARD):
        router_ctx = load_router_inference_context(
            str(router_id),
            input_mode=router_input_mode,
            router_base=rb,
            retrieval_asset_dir=asset_dir,
            device=router_device,
        )
        loaded_router = router_ctx.router

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
                "router_id": router_id,
                "router_input_mode": router_input_mode,
                "router_inference_batch_size": router_inference_batch_size,
            },
        },
    )

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

    tensor_by_qid: dict[str, tuple] = {}
    if router_ctx is not None and pending:
        bs = max(1, int(router_inference_batch_size))
        for i in range(0, len(pending), bs):
            chunk = pending[i : i + bs]
            qs = [str(s.get("question", "") or "").strip() for s in chunk]
            qe, qf = compute_query_tensors_for_router_batch(
                qs, router_ctx, st_batch_size=bs
            )
            for j, s in enumerate(chunk):
                qid = str(s.get("question_id", "") or "").strip()
                tensor_by_qid[qid] = (qe[j : j + 1], qf[j : j + 1])

    retrieval_fp = paths.retrieval_results_jsonl().open("a", encoding="utf-8")
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

            q_emb = feat = None
            if router_ctx is not None:
                q_emb, feat = tensor_by_qid[qid]

            rr = pipeline.run(
                question,
                policy,
                query_embedding=q_emb,
                feature_vector=feat,
            )
            rr = reranker.rerank(question, rr, top_k=rerank_top_k)

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
        if progress is not None:
            progress.close()

    if skipped:
        logger.info("Skipped %d questions with existing answers.", skipped)

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
