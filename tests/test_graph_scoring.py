from surf_rag.core.build_graph import build_graph
from surf_rag.graph.graph_scoring import score_bundle
from surf_rag.graph.graph_types import EvidenceBundle, GraphHop, GraphPath, GroundedHop
from surf_rag.core.scoring_config import ScoringConfig

class TinyEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_query(self, text: str):
        if text not in self.vectors:
            raise KeyError(f"Missing vector for: {text}")
        return self.vectors[text]


class CorpusStub:
    def __init__(self, texts):
        self.texts = texts

    def get_text(self, chunk_id):
        return self.texts.get(chunk_id)


def _bundle(*hops: GraphHop, chunk_ids=None):
    if chunk_ids is None:
        chunk_ids = ["c1"]

    return EvidenceBundle(
        path=GraphPath(start_node=hops[0].source, hops=tuple(hops)),
        grounded_hops=[
            GroundedHop(
                hop=hop,
                chunk_id=chunk_ids[min(i, len(chunk_ids) - 1)],
                support_score=1.0,
            )
            for i, hop in enumerate(hops)
        ],
        supporting_chunk_ids=list(chunk_ids),
    )


def test_score_bundle_prefers_relation_semantically_closer_to_query():
    graph = build_graph([])
    corpus = CorpusStub(
        {
            "c1": "The film was directed by John Smith.",
            "c2": "The country of origin is France.",
        }
    )
    embedder = TinyEmbedder(
        {
            "query::director death question": [1.0, 0.0, 0.0],
            "directed by": [1.0, 0.0, 0.0],
            "country": [0.0, 1.0, 0.0],
            "film directed by person": [1.0, 0.0, 0.0],
            "film country usa": [0.0, 1.0, 0.0],
        }
    )

    better = _bundle(
        GraphHop(source="E:film", relation="directed_by", target="E:person"),
        chunk_ids=["c1"],
    )
    worse = _bundle(
        GraphHop(source="E:film", relation="country", target="E:usa"),
        chunk_ids=["c2"],
    )

    config = ScoringConfig(length_penalty=0.75)

    better_score, _ = score_bundle(
        query="director death question",
        bundle=better,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        config=config,
        cache={},
    )
    worse_score, _ = score_bundle(
        query="director death question",
        bundle=worse,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        config=config,
        cache={},
    )

    assert better_score > worse_score


def test_score_bundle_penalizes_longer_paths_when_relation_relevance_is_equal():
    graph = build_graph([])
    corpus = CorpusStub(
        {
            "c1": "The film was directed by John Smith.",
            "c2": "John Smith died in 1970.",
        }
    )
    embedder = TinyEmbedder(
        {
            "query::same query": [1.0, 0.0, 0.0],
            "r": [1.0, 0.0, 0.0],
            "a r b": [1.0, 0.0, 0.0],
            "b r c": [1.0, 0.0, 0.0],
        }
    )

    short_bundle = _bundle(
        GraphHop(source="E:a", relation="r", target="E:b"),
        chunk_ids=["c1"],
    )
    long_bundle = _bundle(
        GraphHop(source="E:a", relation="r", target="E:b"),
        GraphHop(source="E:b", relation="r", target="E:c"),
        chunk_ids=["c1", "c2"],
    )

    config = ScoringConfig(length_penalty=-0.75)

    short_score, _ = score_bundle(
        query="same query",
        bundle=short_bundle,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        config=config,
        cache={},
    )
    long_score, _ = score_bundle(
        query="same query",
        bundle=long_bundle,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        config=config,
        cache={},
    )

    assert short_score > long_score


def test_score_bundle_returns_breakdown_with_expected_keys():
    graph = build_graph([])
    corpus = CorpusStub({"c1": "The film was directed by John Smith."})
    embedder = TinyEmbedder(
        {
            "query::query": [1.0, 0.0, 0.0],
            "directed by": [1.0, 0.0, 0.0],
            "film directed by person": [1.0, 0.0, 0.0],
        }
    )
    bundle = _bundle(
        GraphHop(source="E:film", relation="directed_by", target="E:person"),
        chunk_ids=["c1"],
    )

    config = ScoringConfig(length_penalty=0.75)

    score, breakdown = score_bundle(
        query="query",
        bundle=bundle,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        config=config,
        cache={},
    )

    assert isinstance(score, float)
    assert "s_pred" in breakdown
    assert "s_len" in breakdown
    assert "s_triple" in breakdown
    assert "hop_scores" in breakdown

