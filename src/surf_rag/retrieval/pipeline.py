"""Single-branch pipeline"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.types import RetrievalResult


@dataclass
class SingleBranchPipeline:
    retriever: BranchRetriever

    def run(self, query: str, **kwargs: Any) -> RetrievalResult:
        return self.retriever.retrieve(query, **kwargs)
