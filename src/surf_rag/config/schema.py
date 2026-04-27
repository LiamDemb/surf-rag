"""Dataclass schema for ``surf-rag/pipeline/v1`` YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _pop(d: dict[str, Any], key: str, default: Any) -> Any:
    if key not in d:
        return default
    return d.pop(key)


@dataclass
class PathsSection:
    data_base: str = "data"
    benchmark_base: str | None = None
    router_base: str | None = None
    benchmark_name: str = "benchmark-name"
    benchmark_id: str = "v01"
    router_id: str = "v01"
    hf_home: str | None = None
    transformers_cache: str | None = None


@dataclass
class RawSourcesSection:
    # None or "" in YAML = omit this dataset (no fallback to process env for --config runs).
    nq_path: str | None = "data/raw/nq_100.jsonl"
    wiki2_path: str | None = "data/raw/2wikimultihop_100.jsonl"
    nq_version: str | None = None
    wiki2_version: str | None = None


@dataclass
class ModelSetupSection:
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CorpusSection:
    max_pages: int = 12
    max_hops: int = 2
    max_list_pages: int = 2
    max_country_pages: int = 1
    chunk_min_tokens: int = 500
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100
    fetch_missing: bool = False


@dataclass
class AlignmentSection:
    full_report: bool = False
    keep_unresolved: bool = False
    tau_sem: float | None = None
    tau_lex: float | None = None


@dataclass
class EntityMatchingSection:
    build_artifacts: bool = True
    force: bool = False
    max_df: int = 8
    max_per_query: int = 12
    min_key_len: int = 3


@dataclass
class OracleSection:
    branch_top_k: int = 20
    fusion_keep_k: int = 20
    betas: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    )
    min_entropy_nats: float = 0.1
    selected_beta: float | None = None


@dataclass
class RouterDatasetSection:
    embedding_model: str = "all-MiniLM-L6-v2"
    train_ratio: float = 0.6
    dev_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class RouterTrainSection:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cpu"
    input_mode: str = "both"
    input_modes: list[str] = field(
        default_factory=lambda: ["both", "query-features", "embedding"]
    )


@dataclass
class RouterSection:
    dataset: RouterDatasetSection = field(default_factory=RouterDatasetSection)
    train: RouterTrainSection = field(default_factory=RouterTrainSection)


@dataclass
class RetrievalSection:
    dense_model: str = "all-MiniLM-L6-v2"
    graph_max_hops: int = 2
    graph_bidirectional: bool = True
    graph_entity_vector_top_k: int = 3
    graph_entity_vector_threshold: float = 0.5
    graph_local_pred_weight: float = 0.55
    graph_bundle_pred_weight: float = 0.55
    graph_length_penalty: float = 0.04


@dataclass
class GenerationSection:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_file: str = "prompts/generator.txt"
    include_graph_provenance: bool = False
    completion_window: str = "24h"
    batch_max_requests: int = 50000


@dataclass
class E2ESection:
    run_id: str | None = None
    split: str = "test"
    policy: str = "learned-soft"
    policies: list[str] = field(
        default_factory=lambda: [
            "dense-only",
            "graph-only",
            "50-50",
            "learned-soft",
            "learned-hard",
        ]
    )
    branch_top_k: int = 20
    fusion_keep_k: int = 20
    reranker: str = "none"
    rerank_top_k: int = 5
    cross_encoder_model: str | None = None
    sentence_rerank_enabled: bool = False
    sentence_rerank_top_k: int = 20
    sentence_rerank_max_sentences: int = 48
    sentence_rerank_max_words: int = 1280
    sentence_rerank_include_title: bool = True
    sentence_rerank_prompt_style: str = "structured"
    router_device: str = "cpu"
    router_input_mode: str = "both"
    router_inference_batch_size: int = 32
    limit: int | None = None
    only_question_ids: list[str] = field(default_factory=list)
    dry_run: bool = False
    include_graph_provenance: bool = False
    completion_window: str | None = None


@dataclass
class SecretsSection:
    openai_api_key_env: str = "OPENAI_API_KEY"
    wikimedia_token_env: str = "WIKIMEDIA_OAUTH2_ACCESS_TOKEN"


@dataclass
class PipelineConfig:
    schema_version: str = "surf-rag/pipeline/v1"
    experiment_id: str | None = None
    seed: int = 42

    paths: PathsSection = field(default_factory=PathsSection)
    raw_sources: RawSourcesSection = field(default_factory=RawSourcesSection)
    model_setup: ModelSetupSection = field(default_factory=ModelSetupSection)
    corpus: CorpusSection = field(default_factory=CorpusSection)
    alignment: AlignmentSection = field(default_factory=AlignmentSection)
    entity_matching: EntityMatchingSection = field(
        default_factory=EntityMatchingSection
    )
    oracle: OracleSection = field(default_factory=OracleSection)
    router: RouterSection = field(default_factory=RouterSection)
    retrieval: RetrievalSection = field(default_factory=RetrievalSection)
    generation: GenerationSection = field(default_factory=GenerationSection)
    e2e: E2ESection = field(default_factory=E2ESection)
    secrets: SecretsSection = field(default_factory=SecretsSection)
