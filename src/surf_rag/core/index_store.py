from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import faiss

class FaissIndex(Protocol):
    def search(self, x, k: int): 
        ...

@dataclass
class FaissIndexStore:
    index_path: str

    def load(self) -> FaissIndex:
        return faiss.read_index(self.index_path)