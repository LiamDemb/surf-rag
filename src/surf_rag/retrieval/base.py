"""Abstract branch retriever."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from surf_rag.retrieval.types import RetrievalResult


class BranchRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        # Return ranked evidence for the query
        raise NotImplementedError
