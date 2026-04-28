from __future__ import annotations

import os
from dataclasses import dataclass

from surf_rag.config.schema import RetrievalSection


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class ScoringConfig:
    """Embedding bundle scorer weights (``score_bundle``) + canonical graph PPR settings."""

    local_pred_weight: float = 0.7
    bundle_pred_weight: float = 0.6
    length_penalty: float = 0.05

    ppr_alpha: float = 0.85
    ppr_max_iter: int = 64
    ppr_tol: float = 1e-6

    graph_transition_mode: str = "support"
    """``support``: row-normalize by corpus relation support counts. ``uniform``: uniform over rel neighbors."""

    graph_max_entities: int = 256
    graph_max_paths: int = 500
    graph_max_frontier_pops: int = 50_000

    graph_seed_softmax_temperature: float = 0.1
    graph_entity_chunk_edge_weight: float = 0.5


def get_default_scoring_config() -> ScoringConfig:
    return ScoringConfig(
        local_pred_weight=float(os.getenv("GRAPH_LOCAL_PRED_WEIGHT", "0.7")),
        bundle_pred_weight=float(os.getenv("GRAPH_BUNDLE_PRED_WEIGHT", "0.6")),
        length_penalty=float(os.getenv("GRAPH_LENGTH_PENALTY", "0.05")),
        ppr_alpha=float(os.getenv("GRAPH_PPR_ALPHA", "0.85")),
        ppr_max_iter=int(os.getenv("GRAPH_PPR_MAX_ITER", "64")),
        ppr_tol=float(os.getenv("GRAPH_PPR_TOL", "1e-6")),
        graph_transition_mode=(
            m
            if (m := os.getenv("GRAPH_TRANSITION_MODE", "support").strip().lower())
            in ("support", "uniform")
            else "support"
        ),
        graph_max_entities=_int_env("GRAPH_MAX_ENTITIES", 256),
        graph_max_paths=_int_env("GRAPH_MAX_PATHS", 500),
        graph_max_frontier_pops=_int_env("GRAPH_MAX_FRONTIER_POPS", 50_000),
        graph_seed_softmax_temperature=float(
            os.getenv("GRAPH_SEED_SOFTMAX_TEMPERATURE", "0.1")
        ),
        graph_entity_chunk_edge_weight=float(
            os.getenv("GRAPH_ENTITY_CHUNK_EDGE_WEIGHT", "0.5")
        ),
    )


DEFAULT_SCORING_CONFIG = get_default_scoring_config()


def scoring_config_from_retrieval_section(r: RetrievalSection) -> ScoringConfig:
    """Build :class:`ScoringConfig` from pipeline ``retrieval`` (no env indirection)."""
    tm = r.graph_transition_mode.strip().lower()
    if tm not in ("support", "uniform"):
        tm = "support"
    return ScoringConfig(
        local_pred_weight=float(r.graph_local_pred_weight),
        bundle_pred_weight=float(r.graph_bundle_pred_weight),
        length_penalty=float(r.graph_length_penalty),
        ppr_alpha=float(r.graph_ppr_alpha),
        ppr_max_iter=int(r.graph_ppr_max_iter),
        ppr_tol=float(r.graph_ppr_tol),
        graph_transition_mode=tm,
        graph_max_entities=int(r.graph_max_entities),
        graph_max_paths=int(r.graph_max_paths),
        graph_max_frontier_pops=int(r.graph_max_frontier_pops),
        graph_seed_softmax_temperature=float(r.graph_seed_softmax_temperature),
        graph_entity_chunk_edge_weight=float(r.graph_entity_chunk_edge_weight),
    )
