from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import networkx as nx

from surf_rag.core.embedder import Embedder
from surf_rag.core.entity_alias_resolver import EntityAliasResolver
from surf_rag.core.entity_index_store import EntityIndexStore
from surf_rag.core.mapping import ChunkIdToText
from surf_rag.core.scoring_config import DEFAULT_SCORING_CONFIG, ScoringConfig
from surf_rag.graph.graph_grounding import ground_path
from surf_rag.graph.graph_paths import enumerate_candidate_paths
from surf_rag.graph.graph_scoring import score_bundle
from surf_rag.graph.graph_store import NetworkXGraphStore
from surf_rag.graph.graph_types import EvidenceBundle, GraphPath
from surf_rag.graph.query_entity_extractor import LLMQueryEntityExtractor
from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


class QueryEntityExtractor(Protocol):
    def extract(self, query: str) -> List[str]: ...


def _default_query_entity_extractor(
    alias_resolver: EntityAliasResolver,
) -> QueryEntityExtractor:
    """Return LLM-based query entity extractor."""
    return LLMQueryEntityExtractor(alias_resolver=alias_resolver)


@dataclass
class GraphRetriever(BranchRetriever):
    """
    Graph-based retrieval: query -> entities -> paths -> grounded bundles -> ranked chunks.

    Path provenance is stored in RetrievedChunk.metadata for prompt rendering.
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
    max_paths_per_start: int = int(os.getenv("GRAPH_MAX_PATHS_PER_START", "50"))

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

    scoring_config: ScoringConfig = field(
        default_factory=lambda: DEFAULT_SCORING_CONFIG
    )

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
            self.entity_extractor = LLMQueryEntityExtractor(
                alias_resolver=self.alias_resolver
            )

    def _ensure_loaded(self) -> None:
        if self._graph is None:
            self._graph = self.graph_store.load()

    @staticmethod
    def _entity_display(node_id: str) -> str:
        return node_id[2:] if node_id.startswith("E:") else node_id

    def _resolve_start_nodes(
        self, extracted_norms: List[str]
    ) -> Tuple[Set[str], List[str], List[Dict[str, Any]]]:
        candidate_nodes = {f"E:{norm}" for norm in extracted_norms}
        start_nodes = {node for node in candidate_nodes if self._graph.has_node(node)}

        vector_matches: List[Dict[str, Any]] = []
        unmatched_norms = sorted(
            norm for norm in extracted_norms if f"E:{norm}" not in start_nodes
        )

        if unmatched_norms and self.entity_index_store:
            for norm in unmatched_norms:
                matches = self.entity_index_store.search(
                    norm,
                    top_k=self.entity_vector_top_k,
                    threshold=self.entity_vector_threshold,
                )
                for match_norm, score in matches:
                    node = f"E:{match_norm}"
                    if not self._graph.has_node(node):
                        continue
                    start_nodes.add(node)
                    vector_matches.append(
                        {
                            "query_norm": norm,
                            "matched_norm": match_norm,
                            "score": float(score),
                        }
                    )

        unresolved = sorted(
            norm for norm in extracted_norms if f"E:{norm}" not in start_nodes
        )
        return start_nodes, unresolved, vector_matches

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

    def _bundle_start_entities(self, bundle: EvidenceBundle) -> Set[str]:
        entities: Set[str] = set()
        for hop in bundle.path.hops:
            if hop.source.startswith("E:"):
                entities.add(hop.source)
            if hop.target.startswith("E:"):
                entities.add(hop.target)
        if bundle.path.start_node.startswith("E:"):
            entities.add(bundle.path.start_node)
        return entities

    def _select_ranked_bundles(
        self,
        ranked_bundles: List[Tuple[EvidenceBundle, float, Dict[str, float]]],
        start_nodes: Set[str],
        top_k: int,
    ) -> List[Tuple[EvidenceBundle, float, Dict[str, float]]]:
        best_per_chunk: Dict[
            Tuple[str, ...], Tuple[EvidenceBundle, float, Dict[str, float]]
        ] = {}

        for item in ranked_bundles:
            bundle, score, _breakdown = item
            chunk_key = tuple(sorted(set(bundle.supporting_chunk_ids)))

            existing = best_per_chunk.get(chunk_key)
            if existing is None:
                best_per_chunk[chunk_key] = item
            else:
                if score > existing[1]:
                    best_per_chunk[chunk_key] = item

        deduped = sorted(
            best_per_chunk.values(),
            key=lambda item: (-item[1], self._bundle_signature(item[0])),
        )

        selected: List[Tuple[EvidenceBundle, float, Dict[str, float]]] = []
        selected_signatures: Set[Tuple[Any, ...]] = set()
        covered_start_nodes: Set[str] = set()

        for start_node in sorted(start_nodes):
            best_item = None
            for item in deduped:
                bundle, score, breakdown = item
                sig = self._bundle_signature(bundle)
                if sig in selected_signatures:
                    continue
                bundle_entities = self._bundle_start_entities(bundle)
                if start_node in bundle_entities:
                    best_item = item
                    break

            if best_item is not None:
                bundle, score, breakdown = best_item
                sig = self._bundle_signature(bundle)
                selected.append(best_item)
                selected_signatures.add(sig)
                covered_start_nodes.add(start_node)

                if len(selected) >= top_k:
                    return selected

        for item in deduped:
            bundle, score, breakdown = item
            sig = self._bundle_signature(bundle)
            if sig in selected_signatures:
                continue
            selected.append(item)
            selected_signatures.add(sig)
            if len(selected) >= top_k:
                break

        return selected

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        debug = bool(kwargs.get("debug", False))
        debug_info: Optional[Dict[str, Any]] = {} if debug else None

        try:
            self._ensure_loaded()
            r0 = time.perf_counter()

            extracted_norms = sorted(set(self.entity_extractor.extract(query)))
            start_nodes, unmatched_entities, vector_matches = self._resolve_start_nodes(
                extracted_norms
            )

            if debug_info is not None:
                debug_info.update(
                    {
                        "extracted_entities": extracted_norms,
                        "start_nodes": sorted(start_nodes),
                        "unmatched_entities": unmatched_entities,
                        "entity_index_matches": vector_matches,
                    }
                )

            if not start_nodes:
                return self._no_context_result(t0, r0, timings, query, debug_info)

            candidate_paths = enumerate_candidate_paths(
                graph=self._graph,
                start_nodes=start_nodes,
                max_hops=self.max_hops,
                bidirectional=self.bidirectional,
                max_paths_per_start=self.max_paths_per_start,
            )

            grounded_bundles: List[EvidenceBundle] = []
            for path in candidate_paths:
                bundle = ground_path(path=path, graph=self._graph)
                if bundle is not None and bundle.grounded_hops:
                    grounded_bundles.append(bundle)

            if not grounded_bundles:
                return self._no_context_result(t0, r0, timings, query, debug_info)

            score_cache: Dict[Any, Any] = {}
            scored_bundles: List[Tuple[EvidenceBundle, float, Dict[str, float]]] = []
            for bundle in grounded_bundles:
                score, breakdown = score_bundle(
                    query=query,
                    bundle=bundle,
                    graph=self._graph,
                    embedder=self.embedder,
                    corpus=self.corpus,
                    config=self.scoring_config,
                    cache=score_cache,
                    debug=debug,
                )
                scored_bundles.append((bundle, float(score), breakdown))

            ranked_bundles = sorted(
                scored_bundles,
                key=lambda item: (-item[1], self._bundle_signature(item[0])),
            )
            selected_bundles = self._select_ranked_bundles(
                ranked_bundles, start_nodes, self.top_k
            )

            chunk_id_to_bundles: Dict[str, List[Tuple[EvidenceBundle, float]]] = {}
            for bundle, score, _ in selected_bundles:
                for cid in set(bundle.supporting_chunk_ids):
                    if cid not in chunk_id_to_bundles:
                        chunk_id_to_bundles[cid] = []
                    chunk_id_to_bundles[cid].append((bundle, score))

            chunks: List[RetrievedChunk] = []
            for cid, bundles in chunk_id_to_bundles.items():
                bundles.sort(key=lambda x: x[1], reverse=True)
                text = self.corpus.get_text(cid)
                if not text:
                    continue
                path_lines = [
                    f"Path: {self._format_path(b.path)}" for b, _ in bundles[:3]
                ]
                score = bundles[0][1]
                meta: Dict[str, Any] = {
                    "branch": "graph",
                    "graph_path_lines": path_lines,
                }
                gt = getattr(self.corpus, "get_title", None)
                if callable(gt):
                    t = gt(cid)
                    if t:
                        meta["title"] = str(t)
                gs = getattr(self.corpus, "get_source", None)
                if callable(gs):
                    s = gs(cid)
                    if s:
                        meta["source"] = str(s)
                chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=text,
                        score=float(score),
                        rank=0,
                        metadata=meta,
                    )
                )

            if debug_info is not None:
                debug_info["bundle_trace"] = [
                    self._bundle_to_debug(b, s, d) for b, s, d in selected_bundles
                ]

            timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
            timings["total"] = (time.perf_counter() - t0) * 1000.0

            if not chunks:
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
