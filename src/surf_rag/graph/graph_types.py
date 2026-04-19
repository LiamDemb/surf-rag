from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GraphHop:
    source: str
    relation: str
    target: str
    is_reverse: bool = False


@dataclass(frozen=True)
class GraphPath:
    start_node: str
    hops: Tuple[GraphHop, ...]


@dataclass(frozen=True)
class GroundedHop:
    hop: GraphHop
    chunk_id: str
    support_score: float


@dataclass
class EvidenceBundle:
    path: GraphPath
    grounded_hops: List[GroundedHop]
    supporting_chunk_ids: List[str]
    score: float | None = None
    score_breakdown: Optional[Dict[str, float]] = field(default_factory=dict)
