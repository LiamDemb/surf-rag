"""Factory for building branch retrievers."""

from __future__ import annotations

import os

from surf_rag.core.embedder import SentenceTransformersEmbedder
from surf_rag.core.scoring_config import DEFAULT_SCORING_CONFIG


def build_dense_retriever(output_dir: str, top_k: int = 10):
    """Build DenseRetriever with corpus, index, embedder."""
    from surf_rag.core.index_store import FaissIndexStore
    from surf_rag.core.mapping import JsonCorpusLoader, VectorMetaMapper
    from surf_rag.strategies.dense import DenseRetriever

    corpus_path = f"{output_dir}/corpus.jsonl"
    index_path = f"{output_dir}/vector_index.faiss"
    meta_path = f"{output_dir}/vector_meta.parquet"

    return DenseRetriever(
        index_store=FaissIndexStore(index_path=index_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        top_k=top_k,
    )


def build_graph_retriever(output_dir: str, top_k: int = 10):
    """Build GraphRetriever with corpus, graph, entity resolution."""
    from surf_rag.core.entity_index_store import EntityIndexStore
    from surf_rag.core.mapping import JsonCorpusLoader
    from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline
    from surf_rag.graph.graph_store import NetworkXGraphStore
    from surf_rag.strategies.graph import (
        GraphRetriever,
        # Revert: add `from surf_rag.strategies.graph import _default_query_entity_extractor`
    )

    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    alias_map_path = f"{output_dir}/alias_map.json"

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus with Phase 1 pipeline."
        )

    # When reverting to the LLM query entity extractor, restore:
    #   alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)
    #   and pass entity_extractor=_default_query_entity_extractor(alias_resolver).

    entity_index_store = None
    if os.path.exists(f"{output_dir}/entity_index.faiss") and os.path.exists(
        f"{output_dir}/entity_index_meta.parquet"
    ):
        entity_index_store = EntityIndexStore(
            f"{output_dir}/entity_index.faiss",
            f"{output_dir}/entity_index_meta.parquet",
        )

    # Lexicon + alias exact phrase matching for query seeds (no per-query LLM call).
    # To revert to LLM-based seeds, replace the line below with:
    #   entity_extractor=_default_query_entity_extractor(alias_resolver),
    entity_extractor = LexiconAliasEntityPipeline.from_artifacts(str(output_dir))

    return GraphRetriever(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        entity_extractor=entity_extractor,
        top_k=top_k,
        max_hops=int(os.getenv("GRAPH_MAX_HOPS", "1")),
        entity_index_store=entity_index_store,
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
        scoring_config=DEFAULT_SCORING_CONFIG,
    )
