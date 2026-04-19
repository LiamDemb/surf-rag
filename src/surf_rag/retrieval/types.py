"""Structured retrieval outputs: canonical handoff to prompt rendering and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class RetrievedChunk:
    """One piece of evidence with stable identity and optional branch-specific metadata."""

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


RetrievalStatus = Literal["OK", "NO_CONTEXT", "ERROR"]


@dataclass
class RetrievalResult:
    """Result of a branch retriever or single-branch pipeline."""

    query: str
    retriever_name: str
    status: RetrievalStatus
    chunks: List[RetrievedChunk] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.chunks.sort(key=lambda c: (-c.score, c.chunk_id))
        for i, ch in enumerate(self.chunks):
            ch.rank = i

        if self.status == "OK":
            if not self.chunks:
                raise ValueError("Status OK requires at least one chunk")
            if self.error is not None:
                raise ValueError("Status OK must not set error")
        elif self.status == "NO_CONTEXT":
            if self.chunks:
                raise ValueError("NO_CONTEXT must have empty chunks")
            if self.error is not None:
                raise ValueError("NO_CONTEXT must not set error")
        elif self.status == "ERROR":
            if not self.error:
                raise ValueError("ERROR requires error message")
