from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.fusion import (
    FUSED_RETRIEVER_NAME,
    FusedCandidate,
    FusionPipeline,
    build_fused_retrieval_result,
    fuse_branch_results,
    fuse_cached_results,
    fused_candidates_to_chunks,
    min_max_normalize,
)
from surf_rag.retrieval.pipeline import SingleBranchPipeline
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult

__all__ = [
    "BranchRetriever",
    "RetrievedChunk",
    "RetrievalResult",
    "SingleBranchPipeline",
    "FusionPipeline",
    "FusedCandidate",
    "FUSED_RETRIEVER_NAME",
    "build_fused_retrieval_result",
    "fuse_branch_results",
    "fuse_cached_results",
    "fused_candidates_to_chunks",
    "min_max_normalize",
    "RoutedFusionPipeline",
    "trim_retrieval_top_k",
]


def __getattr__(name: str):
    """Lazy-import routed fusion (pulls router stack including torch)."""
    if name == "RoutedFusionPipeline":
        from surf_rag.retrieval.routed import RoutedFusionPipeline

        return RoutedFusionPipeline
    if name == "trim_retrieval_top_k":
        from surf_rag.retrieval.routed import trim_retrieval_top_k

        return trim_retrieval_top_k
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
