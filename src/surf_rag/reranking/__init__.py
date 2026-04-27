"""Second-stage reranking over a retrieval shortlist."""

from surf_rag.reranking.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    Reranker,
    build_reranker,
)
from surf_rag.reranking.sentence_reranker import (
    PROMPT_EVIDENCE_KEY,
    PROMPT_EVIDENCE_SENTENCE_SHORTLIST,
    apply_sentence_rerank,
)

__all__ = [
    "Reranker",
    "NoOpReranker",
    "CrossEncoderReranker",
    "build_reranker",
    "PROMPT_EVIDENCE_KEY",
    "PROMPT_EVIDENCE_SENTENCE_SHORTLIST",
    "apply_sentence_rerank",
]
