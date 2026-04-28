from surf_rag.graph.graph_beam_paths import enumerate_beam_candidate_paths
from surf_rag.graph.graph_grounding import ground_path, ground_path_report
from surf_rag.graph.graph_paths import (
    GraphPathEnumerationDiagnostics,
    enumerate_candidate_paths,
)
from surf_rag.graph.graph_scoring import score_bundle
from surf_rag.graph.graph_types import EvidenceBundle, GraphHop, GraphPath, GroundedHop

__all__ = [
    "EvidenceBundle",
    "GraphHop",
    "GraphPath",
    "GraphPathEnumerationDiagnostics",
    "GroundedHop",
    "LLMQueryEntityExtractor",
    "NetworkXGraphStore",
    "enumerate_beam_candidate_paths",
    "enumerate_candidate_paths",
    "ground_path",
    "ground_path_report",
    "score_bundle",
]


def __getattr__(name: str):
    """Lazy-import heavy optional modules (pandas-backed store, LLM client)."""
    if name == "NetworkXGraphStore":
        from surf_rag.graph.graph_store import NetworkXGraphStore

        return NetworkXGraphStore
    if name == "LLMQueryEntityExtractor":
        from surf_rag.graph.query_entity_extractor import LLMQueryEntityExtractor

        return LLMQueryEntityExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
