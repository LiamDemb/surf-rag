from __future__ import annotations

from surf_rag.core.build_graph import build_graph
from surf_rag.strategies.graph import GraphRetriever

from tests._graph_test_utils import (
    CorpusStub,
    GraphStoreStub,
    StaticExtractor,
    comparison_chunks,
    comparison_corpus,
    strategy_embedder,
)


def test_graph_comparison_regression_retrieves_both_reasoning_chains():
    graph = build_graph(comparison_chunks())
    retriever = GraphRetriever(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub(comparison_corpus()),
        entity_extractor=StaticExtractor(
            ["valentin the good", "a daughter of two worlds"]
        ),
        embedder=strategy_embedder(),
        top_k=10,
        max_hops=2,
        bidirectional=True,
    )

    result = retriever.retrieve(
        "Which film has the director who died later, Valentin the Good or A Daughter of Two Worlds?",
        debug=True,
    )

    assert result.status == "OK"
    texts = [c.text for c in result.chunks]
    joined = "\n\n".join(texts)

    assert "A Daughter of Two Worlds was directed by James Young." in joined
    assert "James Young died on 1948-06-09." in joined
    assert "Valentin the Good was directed by Martin Fric." in joined
    assert "Martin Fric died on 1968-08-26." in joined
