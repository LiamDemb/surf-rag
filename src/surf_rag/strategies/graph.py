from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Set, Tuple

import networkx as nx

from surf_rag.core.embedder import Embedder
from surf_rag.core.mapping import ChunkIdToText
from surf_rag.core.scoring_config import ScoringConfig, get_default_scoring_config
from surf_rag.entity_matching.types import PhraseSource, SeedCandidate
from surf_rag.graph.graph_beam_paths import enumerate_global_frontier_paths
from surf_rag.graph.graph_grounding import ground_path_report
from surf_rag.graph.graph_scoring import canonical_ppr_rank_chunks, score_bundle
from surf_rag.graph.graph_seeds import compute_restart_distribution_canonical
from surf_rag.graph.graph_specificity import node_specificity_score
from surf_rag.graph.graph_types import EvidenceBundle, GraphPath
from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult

if TYPE_CHECKING:
    from surf_rag.core.entity_alias_resolver import EntityAliasResolver
    from surf_rag.core.entity_index_store import EntityIndexStore
    from surf_rag.graph.graph_store import NetworkXGraphStore


class QueryEntityExtractor(Protocol):
    def extract(self, query: str) -> List[str]: ...


def _default_query_entity_extractor(
    alias_resolver: EntityAliasResolver,
) -> QueryEntityExtractor:
    """Return LLM-based query entity extractor."""
    from surf_rag.graph.query_entity_extractor import LLMQueryEntityExtractor

    return LLMQueryEntityExtractor(alias_resolver=alias_resolver)


def _seed_candidate_dict(c: SeedCandidate) -> Dict[str, Any]:
    return {
        "canonical_norm": c.canonical_norm,
        "matched_text": c.matched_text,
        "start": c.start,
        "end": c.end,
        "df": c.df,
        "source": getattr(c.source, "value", str(c.source)),
        "match_key": c.match_key,
        "node_id": c.node_id,
        "graph_present": c.graph_present,
        "vector_score": c.vector_score,
    }


@dataclass
class GraphRetriever(BranchRetriever):
    """
    Canonical graph retrieval: semantic softmax × IDF restart masses → frontier paths →
    heterogeneous entity+chunk Personalized PageRank → chunk-node stationary masses.
    """

    name = "Graph"

    graph_store: NetworkXGraphStore = None  # type: ignore[assignment]
    corpus: ChunkIdToText = None  # type: ignore[assignment]
    embedder: Embedder = None  # type: ignore[assignment]

    entity_extractor: Optional[QueryEntityExtractor] = None
    alias_resolver: Optional[EntityAliasResolver] = None
    entity_index_store: Optional[EntityIndexStore] = None

    top_k: int = 10
    max_hops: int = int(os.getenv("GRAPH_MAX_HOPS", "2"))

    entity_vector_top_k: int = int(os.getenv("GRAPH_ENTITY_VECTOR_TOP_K", "3"))
    entity_vector_threshold: float = float(
        os.getenv("GRAPH_ENTITY_VECTOR_THRESHOLD", "0.5")
    )

    bidirectional: bool = field(
        default_factory=lambda: os.getenv("GRAPH_BIDIRECTIONAL", "true").lower()
        in ("1", "true", "yes")
    )
    hop_support_threshold: float = float(
        os.getenv("GRAPH_HOP_SUPPORT_THRESHOLD", "0.5")
    )

    scoring_config: ScoringConfig = field(default_factory=get_default_scoring_config)

    _graph: Optional[nx.DiGraph] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.graph_store is None:
            raise ValueError("GraphRetriever requires graph_store")
        if self.corpus is None:
            raise ValueError("GraphRetriever requires corpus")
        if self.embedder is None:
            raise ValueError("GraphRetriever requires embedder")

        if self.entity_extractor is None:
            if self.alias_resolver is None:
                raise ValueError(
                    "GraphRetriever requires alias_resolver when entity_extractor is not provided"
                )
            from surf_rag.graph.query_entity_extractor import LLMQueryEntityExtractor

            self.entity_extractor = LLMQueryEntityExtractor(
                alias_resolver=self.alias_resolver
            )

    def _ensure_loaded(self) -> None:
        if self._graph is None:
            self._graph = self.graph_store.load()

    @staticmethod
    def _entity_display(node_id: str) -> str:
        return node_id[2:] if node_id.startswith("E:") else node_id

    def _fallback_seed_candidates(self, norms: List[str]) -> List[SeedCandidate]:
        """Build synthetic :class:`SeedCandidate` records when only norms are available."""
        out: List[SeedCandidate] = []
        pos = 0
        for n in norms:
            ln = max(1, len(n))
            out.append(
                SeedCandidate(
                    canonical_norm=n,
                    matched_text=n,
                    start=pos,
                    end=pos + ln,
                    span_token_count=max(1, len(n.split())),
                    df=0,
                    source=PhraseSource.CANONICAL,
                    match_key=n.casefold().replace(" ", "_"),
                )
            )
            pos += ln + 1
        return out

    def _extract_seed_candidates(self, query: str) -> List[SeedCandidate]:
        ex = self.entity_extractor
        fn = getattr(ex, "extract_candidates", None)
        if callable(fn):
            return fn(query, soft_df=True)
        norms = sorted(set(ex.extract(query)))
        return self._fallback_seed_candidates(norms)

    def _enrich_seed_candidates(
        self, candidates: List[SeedCandidate]
    ) -> Tuple[List[SeedCandidate], Set[str], List[Dict[str, Any]]]:
        """Resolve ``E:{norm}`` and optional vector search; annotate graph presence."""
        start_nodes: Set[str] = set()
        vector_diag: List[Dict[str, Any]] = []
        enriched: List[SeedCandidate] = []

        pending: List[SeedCandidate] = []
        for c in candidates:
            node = f"E:{c.canonical_norm}"
            if self._graph.has_node(node):
                enriched.append(
                    replace(c, node_id=node, graph_present=True),  # type: ignore[misc]
                )
                start_nodes.add(node)
            else:
                pending.append(c)

        if pending and self.entity_index_store:
            for c in pending:
                matches = self.entity_index_store.search(
                    c.canonical_norm,
                    top_k=self.entity_vector_top_k,
                    threshold=self.entity_vector_threshold,
                )
                matched = False
                for match_norm, score in matches:
                    node = f"E:{match_norm}"
                    if not self._graph.has_node(node):
                        continue
                    enriched.append(
                        replace(
                            c,
                            canonical_norm=match_norm,
                            node_id=node,
                            vector_score=float(score),
                            graph_present=True,
                        ),  # type: ignore[misc]
                    )
                    start_nodes.add(node)
                    vector_diag.append(
                        {
                            "query_norm": c.canonical_norm,
                            "matched_norm": match_norm,
                            "score": float(score),
                        }
                    )
                    matched = True
                    break
                if not matched:
                    enriched.append(replace(c, graph_present=False))  # type: ignore[misc]
        else:
            for c in pending:
                enriched.append(replace(c, graph_present=False))  # type: ignore[misc]

        return enriched, start_nodes, vector_diag

    def _format_path(self, path: GraphPath) -> str:
        if not path.hops:
            return self._entity_display(path.start_node)

        parts = [self._entity_display(path.hops[0].source)]
        for hop in path.hops:
            relation = (
                f"inv:{hop.relation}"
                if getattr(hop, "is_reverse", False)
                else hop.relation
            )
            parts.append(f"-[{relation}]-> {self._entity_display(hop.target)}")
        return " ".join(parts)

    def _path_to_debug(self, path: GraphPath) -> Dict[str, Any]:
        return {
            "start_node": path.start_node,
            "hops": [
                {
                    "source": hop.source,
                    "relation": hop.relation,
                    "is_reverse": bool(getattr(hop, "is_reverse", False)),
                    "target": hop.target,
                }
                for hop in path.hops
            ],
        }

    def _bundle_to_debug(
        self,
        bundle: EvidenceBundle,
        score: float,
        score_breakdown: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "path": self._path_to_debug(bundle.path),
            "score": float(score),
            "score_breakdown": {
                k: [float(x) for x in v] if isinstance(v, list) else float(v)
                for k, v in score_breakdown.items()
            },
            "supporting_chunk_ids": list(bundle.supporting_chunk_ids),
            "grounded_hops": [
                {
                    "source": gh.hop.source,
                    "relation": gh.hop.relation,
                    "is_reverse": bool(getattr(gh.hop, "is_reverse", False)),
                    "target": gh.hop.target,
                    "chunk_id": gh.chunk_id,
                    "support_score": float(gh.support_score),
                }
                for gh in bundle.grounded_hops
            ],
        }

    def _bundle_signature(self, bundle: EvidenceBundle) -> Tuple[Any, ...]:
        hop_sig = tuple(
            (
                hop.source,
                hop.relation,
                bool(getattr(hop, "is_reverse", False)),
                hop.target,
            )
            for hop in bundle.path.hops
        )
        chunk_sig = tuple(sorted(set(bundle.supporting_chunk_ids)))
        return hop_sig, chunk_sig

    def _bundle_ppr_rank_score(
        self, bundle: EvidenceBundle, pi_dict: Dict[str, float]
    ) -> float:
        """Explain bundle ranking using max entity PPR mass × mean hop specificity."""
        masses: List[float] = []
        specs: List[float] = []
        for gh in bundle.grounded_hops:
            hop = gh.hop
            for nid in (hop.source, hop.target):
                masses.append(float(pi_dict.get(nid, 0.0)))
                specs.append(float(node_specificity_score(self._graph, nid)))
        mass_max = max(masses) if masses else 0.0
        smean = float(sum(specs) / len(specs)) if specs else 0.0
        return mass_max * (0.5 + 0.5 * smean)

    def _local_subgraph_edge_counts(self, entity_nodes: List[str]) -> Dict[str, int]:
        g = self._graph
        rel_edges = 0
        chunk_edges = 0
        ent_set = set(entity_nodes)
        for u in entity_nodes:
            if u not in g:
                continue
            for _, v, data in g.out_edges(u, data=True):
                kind = data.get("kind")
                if kind == "rel" and v in ent_set:
                    rel_edges += 1
                elif kind == "appears_in":
                    chunk_edges += 1
        return {
            "relation_edges_within_entity_scope": rel_edges,
            "appears_in_edges_from_scope_entities": chunk_edges,
        }

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        return self._retrieve_canonical_ppr(query, **kwargs)

    def _retrieve_canonical_ppr(self, query: str, **kwargs) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        debug = bool(kwargs.get("debug", False))
        debug_info: Dict[str, Any] = {}

        try:
            self._ensure_loaded()
            r0 = time.perf_counter()

            raw_candidates = self._extract_seed_candidates(query)
            extracted_norms = sorted({c.canonical_norm for c in raw_candidates})
            enriched, start_nodes, vector_diag = self._enrich_seed_candidates(
                raw_candidates
            )

            unmatched_entities = sorted(
                n for n in extracted_norms if f"E:{n}" not in start_nodes
            )

            graph_diag: Dict[str, Any] = {
                "schema_version": "surf-rag/graph_diag/canonical_v1",
                "retriever_config": {
                    "graph_retrieval_mode": "canonical_ppr",
                    "max_hops": self.max_hops,
                    "bidirectional": self.bidirectional,
                    "hop_support_threshold": self.hop_support_threshold,
                    "top_k": self.top_k,
                    "ppr_alpha": self.scoring_config.ppr_alpha,
                    "ppr_max_iter": self.scoring_config.ppr_max_iter,
                    "ppr_tol": self.scoring_config.ppr_tol,
                    "graph_transition_mode": self.scoring_config.graph_transition_mode,
                    "graph_entity_chunk_edge_weight": self.scoring_config.graph_entity_chunk_edge_weight,
                    "graph_seed_softmax_temperature": self.scoring_config.graph_seed_softmax_temperature,
                    "graph_max_entities": self.scoring_config.graph_max_entities,
                    "graph_max_paths": self.scoring_config.graph_max_paths,
                    "graph_max_frontier_pops": self.scoring_config.graph_max_frontier_pops,
                },
                "seed": {
                    "extracted_entity_count": len(raw_candidates),
                    "start_node_count": len(start_nodes),
                    "has_entity_index": self.entity_index_store is not None,
                    "extracted_candidates": [
                        _seed_candidate_dict(c) for c in raw_candidates
                    ],
                    "resolved_candidates": [_seed_candidate_dict(c) for c in enriched],
                },
            }

            if debug:
                debug_info.update(
                    {
                        "extracted_entities": extracted_norms,
                        "start_nodes": sorted(start_nodes),
                        "unmatched_entities": unmatched_entities,
                        "entity_index_matches": vector_diag,
                    }
                )
            else:
                graph_diag["seed"].update(
                    {
                        "unmatched_entity_count": len(unmatched_entities),
                        "entity_index_match_count": len(vector_diag),
                    }
                )

            embed_cache: Dict[str, Any] = {}

            if not start_nodes:
                graph_diag["no_context_reason"] = "no_start_nodes"
                debug_info["graph_diagnostics"] = graph_diag
                return self._no_context_result(t0, r0, timings, query, debug_info)

            posterior_masses, restart_diag = compute_restart_distribution_canonical(
                self._graph,
                query,
                enriched,
                extracted_norms,
                self.embedder,
                softmax_temperature=self.scoring_config.graph_seed_softmax_temperature,
                cache=embed_cache,
            )
            graph_diag["seed"]["restart_mass"] = {
                str(k): float(v) for k, v in posterior_masses.items()
            }
            graph_diag["seed"]["idf_components"] = restart_diag.get("per_node", {})
            graph_diag["seed"]["df_reference"] = restart_diag.get("df_reference")
            graph_diag["seed"]["semantic_restart_diag"] = restart_diag

            if not posterior_masses:
                nn = max(len(start_nodes), 1)
                posterior_masses = {n: 1.0 / float(nn) for n in start_nodes}

            candidate_paths, enum_diag = enumerate_global_frontier_paths(
                graph=self._graph,
                seed_weights=posterior_masses,
                max_hops=self.max_hops,
                bidirectional=self.bidirectional,
                global_max_paths=int(self.scoring_config.graph_max_paths),
                global_max_pops=int(self.scoring_config.graph_max_frontier_pops),
            )
            graph_diag["enumeration"] = enum_diag.to_json()

            grounded_bundles: List[EvidenceBundle] = []
            grounding_failed: Dict[str, int] = {}
            weak_support_scores: List[float] = []

            for path in candidate_paths:
                report = ground_path_report(
                    self._graph,
                    path,
                    support_threshold=self.hop_support_threshold,
                )
                if report.bundle is not None and report.bundle.grounded_hops:
                    grounded_bundles.append(report.bundle)
                    continue
                if report.failure_kind:
                    grounding_failed[report.failure_kind] = (
                        grounding_failed.get(report.failure_kind, 0) + 1
                    )
                    if (
                        report.failure_kind == "weak_hop_support"
                        and report.best_support_at_failure is not None
                        and len(weak_support_scores) < 5
                    ):
                        weak_support_scores.append(
                            float(report.best_support_at_failure)
                        )

            graph_diag["grounding"] = {
                "candidate_paths": len(candidate_paths),
                "grounded_paths_ok": len(grounded_bundles),
                "grounding_failures_by_kind": grounding_failed,
                "weak_support_sample_scores": weak_support_scores,
            }

            if debug:
                debug_info["grounded_bundle_count"] = len(grounded_bundles)
                debug_info["candidate_paths"] = [
                    self._path_to_debug(p) for p in candidate_paths[:800]
                ]

            if not candidate_paths:
                graph_diag["no_context_reason"] = "zero_candidate_paths"

            if not grounded_bundles:
                graph_diag["grounding"][
                    "note"
                ] = "no_grounded_paths; ranking uses heterogeneous chunk PPR only"

            chunk_scores, pi_dict, ppr_extra = canonical_ppr_rank_chunks(
                self._graph,
                candidate_paths,
                start_nodes,
                posterior_masses,
                config=self.scoring_config,
            )

            entity_nodes = sorted(pi_dict.keys())
            locality = self._local_subgraph_edge_counts(entity_nodes)
            graph_diag["local_subgraph"] = {
                "entity_count": len(entity_nodes),
                "relation_edge_count": locality["relation_edges_within_entity_scope"],
                "appears_in_edge_count": locality[
                    "appears_in_edges_from_scope_entities"
                ],
            }
            graph_diag["ppr"] = ppr_extra.get("ppr", {})
            chunk_scoring = ppr_extra.get("chunk_scoring", {})
            top_preview = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))[
                : min(15, len(chunk_scores))
            ]
            graph_diag["chunk_projection"] = {
                "top_chunks": [
                    {"chunk_id": cid, "score": float(sc)} for cid, sc in top_preview
                ],
                "chunk_scoring": chunk_scoring,
            }

            paths_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for bundle in grounded_bundles:
                pd = self._path_to_debug(bundle.path)
                for cid in set(bundle.supporting_chunk_ids):
                    if len(paths_by_chunk[cid]) < 8:
                        paths_by_chunk[cid].append(pd)
            graph_diag["explanations"] = {"paths_by_chunk": dict(paths_by_chunk)}

            bundle_by_chunk: Dict[str, List[EvidenceBundle]] = defaultdict(list)
            for bundle in grounded_bundles:
                for cid in set(bundle.supporting_chunk_ids):
                    bundle_by_chunk[cid].append(bundle)

            ranked_chunk_ids = sorted(
                chunk_scores.keys(),
                key=lambda cid: (-float(chunk_scores.get(cid, 0.0)), cid),
            )

            chunks: List[RetrievedChunk] = []
            for cid in ranked_chunk_ids:
                if len(chunks) >= self.top_k:
                    break
                text = self.corpus.get_text(cid)
                if not text:
                    continue
                bundles_here = bundle_by_chunk.get(cid, [])
                bundles_here_sorted = sorted(
                    bundles_here,
                    key=lambda bb: (
                        -self._bundle_ppr_rank_score(bb, pi_dict),
                        self._bundle_signature(bb),
                    ),
                )
                path_lines = [
                    f"Path: {self._format_path(b.path)}"
                    for b in bundles_here_sorted[:3]
                ]
                chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=text,
                        score=float(chunk_scores.get(cid, 0.0)),
                        rank=0,
                        metadata={
                            "branch": "graph",
                            "graph_retrieval_mode": "canonical_ppr",
                            "graph_path_lines": path_lines,
                        },
                    )
                )

            graph_diag["ranking"] = {
                "mode": "canonical_ppr",
                "chunk_scores_nonzero": sum(
                    1 for v in chunk_scores.values() if v > 0.0
                ),
                "unique_chunk_ids_considered": len(chunk_scores),
                "chunks_with_text": len(chunks),
            }

            graph_diag["ranking"]["top_chunk_scores"] = [
                {"chunk_id": cid, "score": float(sc)} for cid, sc in top_preview
            ]

            if debug:
                score_cache: Dict[Any, Any] = {}
                debug_info["bundle_trace"] = []
                for b in grounded_bundles[:80]:
                    emb_score, bd = score_bundle(
                        query=query,
                        bundle=b,
                        graph=self._graph,
                        embedder=self.embedder,
                        corpus=self.corpus,
                        config=self.scoring_config,
                        cache=score_cache,
                        debug=False,
                    )
                    debug_info["bundle_trace"].append(
                        self._bundle_to_debug(b, float(emb_score), bd)
                    )

            timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
            timings["total"] = (time.perf_counter() - t0) * 1000.0

            debug_info["graph_diagnostics"] = graph_diag

            if not chunks:
                graph_diag["no_context_reason"] = "no_chunk_text_in_corpus"
                return self._no_context_result(t0, r0, timings, query, debug_info)

            return RetrievalResult(
                query=query,
                retriever_name=self.name,
                status="OK",
                chunks=chunks,
                latency_ms=timings,
                debug_info=debug_info,
            )

        except Exception as exc:
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query,
                retriever_name=self.name,
                status="ERROR",
                chunks=[],
                latency_ms=timings,
                error=f"GraphRetriever failed during retrieval: {type(exc).__name__}: {exc}",
                debug_info=debug_info,
            )

    def _no_context_result(
        self,
        t0: float,
        r0: float,
        timings: Dict[str, float],
        query: str,
        debug_info: Optional[Dict[str, Any]],
    ) -> RetrievalResult:
        timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
        timings["total"] = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            query=query,
            retriever_name=self.name,
            status="NO_CONTEXT",
            chunks=[],
            latency_ms=timings,
            debug_info=debug_info,
        )
