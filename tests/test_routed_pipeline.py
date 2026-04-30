"""Routed fusion pipeline: branch execution counts."""

from __future__ import annotations

import numpy as np
import pytest

from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.routed import RoutedFusionPipeline, trim_retrieval_top_k
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult
from surf_rag.router.inference import LoadedRouter
from surf_rag.router.model import RouterMLP, RouterMLPConfig
from surf_rag.router.policies import RoutingPolicyName

pytest.importorskip("torch")


def _chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, text="t", score=score, rank=0, metadata={})


def _ok(name: str, cids: list[str]) -> RetrievalResult:
    ch = [_chunk(cid, 1.0 - 0.01 * i) for i, cid in enumerate(cids)]
    return RetrievalResult(
        query="q",
        retriever_name=name,
        status="OK",
        chunks=ch,
        latency_ms={"retrieval": 1.0},
    )


class _CountingRetriever(BranchRetriever):
    def __init__(self, name: str, r: RetrievalResult) -> None:
        self.name = name
        self._r = r
        self.calls = 0

    def retrieve(self, query: str, **kwargs: object) -> RetrievalResult:
        self.calls += 1
        return self._r


def test_trim_top_k() -> None:
    r = _ok("Dense", [f"x{i}" for i in range(5)])
    t = trim_retrieval_top_k(r, 2)
    assert len(t.chunks) == 2


def test_dense_only_one_branch() -> None:
    d = _CountingRetriever("Dense", _ok("Dense", ["a", "b", "c"]))
    g = _CountingRetriever("Graph", _ok("Graph", ["d"]))
    pl = RoutedFusionPipeline(d, g, fusion_keep_k=2, router=None)
    out = pl.run("q", RoutingPolicyName.DENSE_ONLY)
    assert d.calls == 1 and g.calls == 0
    assert out.status == "OK"
    assert len(out.chunks) == 2


def test_50_50_both_branches() -> None:
    d = _CountingRetriever("Dense", _ok("Dense", ["a"]))
    g = _CountingRetriever("Graph", _ok("Graph", ["b"]))
    pl = RoutedFusionPipeline(d, g, fusion_keep_k=5, router=None)
    pl.run("q", RoutingPolicyName.EQUAL_50_50)
    assert d.calls == 1 and g.calls == 1


def test_learned_soft_with_router() -> None:
    cfg = RouterMLPConfig(
        embedding_dim=4,
        feature_dim=2,
        embed_proj_dim=2,
        feat_proj_dim=2,
        hidden_dim=4,
        dropout=0.0,
    )
    m = RouterMLP(cfg)
    router = LoadedRouter(
        model=m,
        config=cfg,
        weight_grid=np.asarray([float(i) / 10.0 for i in range(11)]),
        device="cpu",
        manifest={},
    )
    d = _CountingRetriever("Dense", _ok("Dense", ["a"]))
    g = _CountingRetriever("Graph", _ok("Graph", ["b"]))
    pl = RoutedFusionPipeline(d, g, fusion_keep_k=3, router=router)
    emb = np.ones(4, dtype=np.float32)
    feat = np.zeros(2, dtype=np.float32)
    pl.run(
        "q",
        RoutingPolicyName.LEARNED_SOFT,
        query_embedding=emb,
        feature_vector=feat,
    )
    assert d.calls == 1 and g.calls == 1
