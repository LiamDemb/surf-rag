"""Apply :class:`PipelineConfig` to argparse namespaces when flags were not passed."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from surf_rag.config.argv import argv_provides
from surf_rag.config.loader import (
    PipelineConfig,
    load_pipeline_config,
    resolve_paths,
)
from surf_rag.config.env import apply_pipeline_env_from_config


def load_config_and_apply_env(path: Path | str | None) -> PipelineConfig | None:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    cfg = load_pipeline_config(p)
    apply_pipeline_env_from_config(cfg)
    return cfg


def merge_ingest_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    rs = cfg.raw_sources
    # YAML is authoritative: null/"" → None (clears path; no argparse/env fallback).
    if not argv_provides(argv, "--nq"):
        args.nq = rs.nq_path
    if not argv_provides(argv, "--2wiki"):
        args.wiki2 = rs.wiki2_path
    if not argv_provides(argv, "--output-dir"):
        args.output_dir = str(rp.benchmark_dir)
    if not argv_provides(argv, "--nq-version") and rs.nq_version:
        args.nq_version = rs.nq_version
    if not argv_provides(argv, "--2wiki-version") and rs.wiki2_version:
        args.wiki2_version = rs.wiki2_version


def merge_fetch_wikipedia_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    rs = cfg.raw_sources
    if not argv_provides(argv, "--benchmark"):
        args.benchmark = str(rp.benchmark_path)
    if not argv_provides(argv, "--nq"):
        args.nq = rs.nq_path
    if not argv_provides(argv, "--2wiki"):
        args.wiki2 = rs.wiki2_path
    if not argv_provides(argv, "--output-dir"):
        args.output_dir = str(rp.corpus_dir)
    if not argv_provides(argv, "--docstore"):
        args.docstore = str(rp.docstore_path)


def merge_align_2wiki_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    c = cfg.corpus
    a = cfg.alignment
    m = cfg.model_setup
    if not argv_provides(argv, "--benchmark"):
        args.benchmark = str(rp.benchmark_path)
    if not argv_provides(argv, "--docstore"):
        args.docstore = str(rp.docstore_path)
    if not argv_provides(argv, "--corpus"):
        args.corpus = str(rp.corpus_path) if rp.corpus_path else None
    if not argv_provides(argv, "--model-name"):
        args.model_name = m.embedding_model
    if a.tau_sem is not None and not argv_provides(argv, "--tau-sem"):
        args.tau_sem = a.tau_sem
    if a.tau_lex is not None and not argv_provides(argv, "--tau-lex"):
        args.tau_lex = a.tau_lex
    if a.full_report and not argv_provides(argv, "--full-report"):
        args.full_report = True
    if a.keep_unresolved and not argv_provides(argv, "--keep-unresolved"):
        args.keep_unresolved = True
    if not argv_provides(argv, "--chunk-min-tokens"):
        args.chunk_min_tokens = c.chunk_min_tokens
    if not argv_provides(argv, "--chunk-max-tokens"):
        args.chunk_max_tokens = c.chunk_max_tokens
    if not argv_provides(argv, "--chunk-overlap-tokens"):
        args.chunk_overlap_tokens = c.chunk_overlap_tokens


def merge_filter_benchmark_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    if not argv_provides(argv, "--benchmark"):
        args.benchmark = str(rp.benchmark_path)
    if not argv_provides(argv, "--corpus"):
        args.corpus = str(rp.corpus_path)


def merge_build_corpus_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    c = cfg.corpus
    m = cfg.model_setup
    if not argv_provides(argv, "--benchmark"):
        args.benchmark = str(rp.benchmark_path)
    if not argv_provides(argv, "--nq"):
        args.nq = cfg.raw_sources.nq_path
    if not argv_provides(argv, "--2wiki"):
        args.wiki2 = cfg.raw_sources.wiki2_path
    if not argv_provides(argv, "--output-dir"):
        args.output_dir = str(rp.corpus_dir)
    if not argv_provides(argv, "--docstore"):
        args.docstore = str(rp.docstore_path)
    if not argv_provides(argv, "--model-name"):
        args.model_name = m.embedding_model
    if not argv_provides(argv, "--max-pages"):
        args.max_pages = c.max_pages
    if not argv_provides(argv, "--max-hops"):
        args.max_hops = c.max_hops
    if not argv_provides(argv, "--max-list-pages"):
        args.max_list_pages = c.max_list_pages
    if not argv_provides(argv, "--max-country-pages"):
        args.max_country_pages = c.max_country_pages
    if not argv_provides(argv, "--chunk-min-tokens"):
        args.chunk_min_tokens = c.chunk_min_tokens
    if not argv_provides(argv, "--chunk-max-tokens"):
        args.chunk_max_tokens = c.chunk_max_tokens
    if not argv_provides(argv, "--chunk-overlap-tokens"):
        args.chunk_overlap_tokens = c.chunk_overlap_tokens


def merge_oracle_prepare_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    o = cfg.oracle
    p = cfg.paths
    if not argv_provides(argv, "--router-id"):
        args.router_id = p.router_id
    if not argv_provides(argv, "--benchmark-name"):
        args.benchmark_name = p.benchmark_name
    if not argv_provides(argv, "--benchmark-id"):
        args.benchmark_id = p.benchmark_id
    if not argv_provides(argv, "--benchmark-path"):
        args.benchmark_path = rp.benchmark_path
    if not argv_provides(argv, "--retrieval-asset-dir"):
        args.retrieval_asset_dir = rp.corpus_dir
    if not argv_provides(argv, "--branch-top-k"):
        args.branch_top_k = o.branch_top_k
    if not argv_provides(argv, "--fusion-keep-k"):
        args.fusion_keep_k = o.fusion_keep_k
    if not argv_provides(argv, "--router-base"):
        args.router_base = rp.router_base


def merge_sweep_beta_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    o = cfg.oracle
    if not argv_provides(argv, "--router-id"):
        args.router_id = cfg.paths.router_id
    if not argv_provides(argv, "--min-entropy-nats"):
        args.min_entropy_nats = o.min_entropy_nats
    if not argv_provides(argv, "--betas"):
        args.betas = o.betas
    if not argv_provides(argv, "--router-base"):
        args.router_base = rp.router_base


def merge_create_soft_labels_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    o = cfg.oracle
    if not argv_provides(argv, "--router-id"):
        args.router_id = cfg.paths.router_id
    if not args.beta and o.betas:
        args.beta = list(o.betas)
    if not argv_provides(argv, "--selected-beta") and o.selected_beta is not None:
        args.selected_beta = o.selected_beta
    if not argv_provides(argv, "--router-base"):
        args.router_base = rp.router_base


def merge_router_build_dataset_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    rd = cfg.router.dataset
    p = cfg.paths
    if not argv_provides(argv, "--router-id"):
        args.router_id = p.router_id
    if not argv_provides(argv, "--benchmark-name"):
        args.benchmark_name = p.benchmark_name
    if not argv_provides(argv, "--benchmark-id"):
        args.benchmark_id = p.benchmark_id
    if not argv_provides(argv, "--benchmark-path"):
        args.benchmark_path = rp.benchmark_path
    if not argv_provides(argv, "--retrieval-asset-dir"):
        args.retrieval_asset_dir = rp.corpus_dir
    if not argv_provides(argv, "--router-base"):
        args.router_base = rp.router_base
    if not argv_provides(argv, "--embedding-model"):
        args.embedding_model = rd.embedding_model
    if not argv_provides(argv, "--split-seed"):
        args.split_seed = cfg.seed
    if not argv_provides(argv, "--train-ratio"):
        args.train_ratio = rd.train_ratio
    if not argv_provides(argv, "--dev-ratio"):
        args.dev_ratio = rd.dev_ratio
    if not argv_provides(argv, "--test-ratio"):
        args.test_ratio = rd.test_ratio


def merge_router_train_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    rt = cfg.router.train
    p = cfg.paths
    if not argv_provides(argv, "--router-id"):
        args.router_id = p.router_id
    if not argv_provides(argv, "--router-base"):
        args.router_base = rp.router_base
    if not argv_provides(argv, "--epochs"):
        args.epochs = rt.epochs
    if not argv_provides(argv, "--batch-size"):
        args.batch_size = rt.batch_size
    if not argv_provides(argv, "--learning-rate"):
        args.learning_rate = rt.learning_rate
    if not argv_provides(argv, "--device"):
        args.device = rt.device
    if not argv_provides(argv, "--input-mode"):
        args.input_mode = rt.input_mode


def merge_router_evaluate_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    merge_router_train_args(args, cfg, argv=argv)


def merge_e2e_common_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    """Fill E2E ``print-config``/``prepare``/``collect``/``evaluate`` common fields."""
    argv = argv if argv is not None else sys.argv
    rp = resolve_paths(cfg)
    e = cfg.e2e
    if not argv_provides(argv, "--benchmark-base"):
        args.benchmark_base = rp.benchmark_base
    if not argv_provides(argv, "--benchmark-name"):
        args.benchmark_name = rp.benchmark_name
    if not argv_provides(argv, "--benchmark-id"):
        args.benchmark_id = rp.benchmark_id
    if not argv_provides(argv, "--benchmark-path"):
        args.benchmark_path = rp.benchmark_path
    if not argv_provides(argv, "--split"):
        args.split = e.split
    if not argv_provides(argv, "--run-id") and e.run_id:
        args.run_id = e.run_id
    if not argv_provides(argv, "--policy"):
        args.policy = e.policy
    if not argv_provides(argv, "--retrieval-asset-dir"):
        args.retrieval_asset_dir = rp.corpus_dir


def merge_e2e_prepare_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    merge_e2e_common_args(args, cfg, argv=argv)
    e = cfg.e2e
    p = cfg.paths
    if not argv_provides(argv, "--router-id") and p.router_id:
        args.router_id = p.router_id
    rb = resolve_paths(cfg).router_base
    if not argv_provides(argv, "--router-base"):
        args.router_base = rb
    if not argv_provides(argv, "--fusion-keep-k"):
        args.fusion_keep_k = e.fusion_keep_k
    if not argv_provides(argv, "--branch-top-k"):
        args.branch_top_k = e.branch_top_k
    if not argv_provides(argv, "--reranker"):
        args.reranker = e.reranker
    if not argv_provides(argv, "--rerank-top-k"):
        args.rerank_top_k = e.rerank_top_k
    if not argv_provides(argv, "--cross-encoder-model") and e.cross_encoder_model:
        args.cross_encoder_model = e.cross_encoder_model
    if e.limit is not None and not argv_provides(argv, "--limit"):
        args.limit = e.limit
    if not argv_provides(argv, "--completion-window") and e.completion_window:
        args.completion_window = e.completion_window
    elif not hasattr(args, "completion_window") or args.completion_window is None:
        args.completion_window = cfg.generation.completion_window
    if e.include_graph_provenance and not argv_provides(
        argv, "--include-graph-provenance"
    ):
        args.include_graph_provenance = True
    if e.dry_run and not argv_provides(argv, "--dry-run"):
        args.dry_run = True
    if not argv_provides(argv, "--router-device"):
        args.router_device = e.router_device
    if not argv_provides(argv, "--router-input-mode"):
        args.router_input_mode = e.router_input_mode
    if not argv_provides(argv, "--router-inference-batch-size"):
        args.router_inference_batch_size = e.router_inference_batch_size
    if e.only_question_ids and not argv_provides(argv, "--only-question-id"):
        args.only_question_id = list(e.only_question_ids)


def merge_e2e_evaluate_args(
    args: Namespace, cfg: PipelineConfig, argv: list[str] | None = None
) -> None:
    argv = argv if argv is not None else sys.argv
    merge_e2e_common_args(args, cfg, argv=argv)
    p = cfg.paths
    rb = resolve_paths(cfg).router_base
    if not argv_provides(argv, "--router-id") and p.router_id:
        args.router_id = p.router_id
    if not argv_provides(argv, "--router-base"):
        args.router_base = rb
