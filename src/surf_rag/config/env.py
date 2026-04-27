"""Single entry point for loading dotenv in CLI tools."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_app_env(*, override: bool = False) -> None:
    """Load ``.env`` from the current working directory if present.

    ``override=False`` matches python-dotenv defaults: existing process env wins.
    """
    load_dotenv(override=override)


def _env_s(value: object) -> str:
    """Coerce a config value to ``str`` for ``os.environ`` (YAML may use int/float/bool)."""
    if value is None:
        return ""
    return str(value)


def set_env_if_unset(key: str, value: str | None) -> None:
    """Set ``os.environ[key]`` only if missing or empty."""
    if value is None or str(value).strip() == "":
        return
    if not os.environ.get(key) or not str(os.environ.get(key)).strip():
        os.environ[key] = str(value)


def apply_pipeline_env_from_config(config: object) -> None:
    """Push resolved :class:`PipelineConfig` values into the process environment.

    Used when running with ``--config`` so libraries that still read
    ``os.environ`` see consistent settings. Call after ``load_app_env()``.
    """
    from surf_rag.config.schema import PipelineConfig  # local import

    if not isinstance(config, PipelineConfig):
        return
    cfg: PipelineConfig = config
    g = cfg.generation
    r = cfg.retrieval
    m = cfg.model_setup
    os.environ["OPENAI_MODEL"] = _env_s(g.model)
    os.environ["GENERATOR_MAX_TOKENS"] = _env_s(g.max_tokens)
    os.environ["GENERATOR_TEMPERATURE"] = _env_s(g.temperature)
    os.environ["GENERATOR_BASE_PROMPT_FILE"] = _env_s(g.prompt_file)
    os.environ["OPENAI_BATCH_MAX_REQUESTS"] = _env_s(g.batch_max_requests)
    os.environ["EMBEDDING_MODEL"] = _env_s(m.embedding_model)
    os.environ["MODEL_NAME"] = _env_s(m.embedding_model)
    os.environ["CROSS_ENCODER_MODEL"] = _env_s(m.cross_encoder_model)
    os.environ["SPACY_MODEL"] = _env_s(m.spacy_model)
    os.environ["DATA_BASE"] = _env_s(cfg.paths.data_base)
    if cfg.paths.benchmark_base:
        os.environ["BENCHMARK_BASE"] = _env_s(cfg.paths.benchmark_base)
    if cfg.paths.router_base:
        os.environ["ROUTER_BASE"] = _env_s(cfg.paths.router_base)
    os.environ["BENCHMARK_NAME"] = _env_s(cfg.paths.benchmark_name)
    os.environ["BENCHMARK_ID"] = _env_s(cfg.paths.benchmark_id)
    os.environ["ROUTER_ID"] = _env_s(cfg.paths.router_id)
    if cfg.e2e.include_graph_provenance:
        os.environ["INCLUDE_GRAPH_PATHS_IN_PROMPT"] = "true"
    # Graph + scoring (downstream `strategies/graph.py` and `scoring_config.py`)
    os.environ["GRAPH_MAX_HOPS"] = str(r.graph_max_hops)
    os.environ["GRAPH_BIDIRECTIONAL"] = "true" if r.graph_bidirectional else "false"
    os.environ["GRAPH_ENTITY_VECTOR_TOP_K"] = str(r.graph_entity_vector_top_k)
    os.environ["GRAPH_ENTITY_VECTOR_THRESHOLD"] = str(r.graph_entity_vector_threshold)
    os.environ["GRAPH_LOCAL_PRED_WEIGHT"] = str(r.graph_local_pred_weight)
    os.environ["GRAPH_BUNDLE_PRED_WEIGHT"] = str(r.graph_bundle_pred_weight)
    os.environ["GRAPH_LENGTH_PENALTY"] = str(r.graph_length_penalty)
    c = cfg.corpus
    os.environ["MAX_PAGES"] = str(c.max_pages)
    os.environ["MAX_HOPS"] = str(c.max_hops)
    os.environ["MAX_LIST_PAGES"] = str(c.max_list_pages)
    os.environ["MAX_COUNTRY_PAGES"] = str(c.max_country_pages)
    os.environ["CHUNK_MIN_TOKENS"] = str(c.chunk_min_tokens)
    os.environ["CHUNK_MAX_TOKENS"] = str(c.chunk_max_tokens)
    os.environ["CHUNK_OVERLAP_TOKENS"] = str(c.chunk_overlap_tokens)
    o = cfg.oracle
    os.environ["ORACLE_BRANCH_TOP_K"] = str(o.branch_top_k)
    os.environ["ORACLE_FUSION_KEEP_K"] = str(o.fusion_keep_k)
    os.environ["ORACLE_MIN_ENTROPY_NATS"] = str(o.min_entropy_nats)
    rd = cfg.router.dataset
    os.environ["SEED"] = str(cfg.seed)
    os.environ["TRAIN_RATIO"] = str(rd.train_ratio)
    os.environ["DEV_RATIO"] = str(rd.dev_ratio)
    os.environ["TEST_RATIO"] = str(rd.test_ratio)
    rt = cfg.router.train
    os.environ["ROUTER_EPOCHS"] = str(rt.epochs)
    os.environ["ROUTER_BATCH_SIZE"] = str(rt.batch_size)
    os.environ["ROUTER_LEARNING_RATE"] = str(rt.learning_rate)
    os.environ["ROUTER_TRAIN_DEVICE"] = str(rt.device)
    os.environ["ROUTER_INPUT_MODE"] = str(rt.input_mode)
    os.environ["EMBEDDING_MODEL_FOR_ROUTER"] = _env_s(rd.embedding_model)
    em = cfg.entity_matching
    os.environ["ENTITY_MATCH_MAX_DF"] = str(em.max_df)
    os.environ["ENTITY_MATCH_MAX_PER_QUERY"] = str(em.max_per_query)
    os.environ["ENTITY_MATCH_MIN_KEY_LEN"] = str(em.min_key_len)
