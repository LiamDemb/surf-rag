"""Tests for GraphRetriever."""

from __future__ import annotations

from surf_rag.core.build_graph import build_graph
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.strategies.graph import GraphRetriever

from tests._graph_test_utils import (
    CorpusStub,
    GraphStoreStub,
    StaticExtractor,
    strategy_embedder,
    toy_chunks,
)


def test_graph_retriever_returns_ok_with_chunks():
    graph = build_graph(toy_chunks())
    retriever = GraphRetriever(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
    )

    result = retriever.retrieve("How is A related to B?")

    assert isinstance(result, RetrievalResult)
    assert result.status == "OK"
    assert len(result.chunks) > 0
    assert "retrieval" in result.latency_ms
    assert "total" in result.latency_ms
    assert result.debug_info is not None
    gd = result.debug_info.get("graph_diagnostics")
    assert gd is not None
    assert gd.get("schema_version") == "surf-rag/graph_diag/v1"
    assert "enumeration" in gd and "grounding" in gd


def test_graph_retriever_returns_no_context_for_unmatched_entities():
    graph = build_graph(toy_chunks())
    retriever = GraphRetriever(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        entity_extractor=StaticExtractor(["does-not-exist"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
    )

    result = retriever.retrieve("Unknown entity query")

    assert result.status == "NO_CONTEXT"
    assert result.chunks == []
    assert result.debug_info is not None
    gd = result.debug_info.get("graph_diagnostics")
    assert gd is not None
    assert gd.get("no_context_reason") == "no_start_nodes"


def test_graph_retriever_is_deterministic_for_same_query():
    graph = build_graph(toy_chunks())
    retriever = GraphRetriever(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
    )

    result1 = retriever.retrieve("How is A related to C?")
    result2 = retriever.retrieve("How is A related to C?")

    assert result1.status == result2.status
    assert [c.chunk_id for c in result1.chunks] == [c.chunk_id for c in result2.chunks]


def test_graph_retriever_debug_trace_contains_path_and_bundle_fields():
    graph = build_graph(toy_chunks())
    retriever = GraphRetriever(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
        bidirectional=True,
    )

    result = retriever.retrieve("How is A related to C?", debug=True)

    assert result.status == "OK"
    assert result.debug_info is not None
    assert "start_nodes" in result.debug_info
    assert "bundle_trace" in result.debug_info
