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
    # Root for rendered figures when figures.output_dir is unset. When None, defaults
    # to ~/figures (see resolve_paths); set explicitly to place figures under data/.
    figures_base: str | None = None
    benchmark_name: str = "benchmark-name"
    benchmark_id: str = "v01"
    router_id: str = "v01"
    router_architecture_id: str | None = None
    hf_home: str | None = None
    transformers_cache: str | None = None


@dataclass
class RawSourcesSection:
    # None or "" in YAML = omit this dataset (no fallback to process env for --config runs).
    nq_path: str | None = "data/raw/nq_100.jsonl"
    wiki2_path: str | None = "data/raw/2wikimultihop_100.jsonl"
    hotpotqa_path: str | None = None
    nq_version: str | None = None
    wiki2_version: str | None = None
    hotpotqa_version: str | None = None


@dataclass
class ModelSetupSection:
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str | None = None


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
    architecture: str = "mlp-v1"
    architecture_kwargs: dict[str, Any] = field(default_factory=dict)
    excluded_features: list[str] = field(default_factory=list)
    input_mode: str = "both"
    midpoint_balance_masking: bool = False
    midpoint_balance_epsilon: float = 1e-6
    loss: str = "regret"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
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
    graph_hop_support_threshold: float = 0.5
    # Matches ``surf_rag.core.scoring_config.ScoringConfig`` / ``get_default_scoring_config``.
    graph_ppr_alpha: float = 0.85
    graph_ppr_max_iter: int = 64
    graph_ppr_tol: float = 1e-6
    graph_transition_mode: str = "support"
    graph_max_entities: int = 256
    graph_max_paths: int = 500
    graph_max_frontier_pops: int = 50_000
    graph_seed_softmax_temperature: float = 0.1
    graph_entity_chunk_edge_weight: float = 0.5


@dataclass
class GenerationSection:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    reasoning_effort: str | None = None
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
            "learned-hybrid",
            "oracle-upper-bound",
        ]
    )
    branch_top_k: int = 20
    fusion_keep_k: int = 20
    reranker: str = "none"
    rerank_top_k: int = 5
    cross_encoder_model: str | None = None
    router_device: str = "cpu"
    router_input_mode: str = "both"
    router_inference_batch_size: int = 32
    latency_warmup_questions: int = 0
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
class FiguresThemeSection:
    """Matplotlib style preset for ``surf_rag.viz.theme.apply_theme``."""

    name: str = "default"
    dpi: int = 200
    font_size: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class FiguresSection:
    """Optional figure generation (opt-in via ``enabled``).

    When ``output_dir`` is unset, the output directory depends on each plot's
    ``kind``: router plots use
    :func:`surf_rag.viz.paths_layout.canonical_router_figure_dir`, and
    ``benchmark_oracle_ndcg_heatmap`` and ``oracle_argmax_weight_histogram`` use
    :func:`surf_rag.viz.paths_layout.canonical_benchmark_figure_dir`
    (``{figures_base}/benchmarks/{name}/{id}/``). Set ``output_dir`` to force a
    single directory for all plots in the run.
    """

    enabled: bool = False
    output_dir: str | None = None
    theme: FiguresThemeSection = field(default_factory=FiguresThemeSection)
    image_format: str = "png"
    plots: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GraphRetrievalSweepSection:
    """Optional grid-search settings for ``scripts/dev/graph_retrieval_grid_search.py``.

    Keys under ``grid`` are :class:`RetrievalSection` field names; each value is a list
    (or a single scalar) of candidates. The sweep takes the Cartesian product.
    """

    grid: dict[str, Any] = field(default_factory=dict)
    objective: str = "overlap_breakdown.all.retrieval_before_ce.retrieval_at_k.10.ndcg"
    sweep_id: str | None = None
    use_router_overlap_splits: bool = False


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
    graph_retrieval_sweep: GraphRetrievalSweepSection = field(
        default_factory=GraphRetrievalSweepSection
    )
    figures: FiguresSection = field(default_factory=FiguresSection)
