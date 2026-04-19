from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import pandas as pd


@dataclass
class NetworkXGraphStore:
    """Lazy loader for a pickled NetworkX graph artifact."""

    graph_path: str
    _graph: Optional[nx.DiGraph] = None

    def load(self) -> nx.DiGraph:
        if self._graph is None:
            graph = pd.read_pickle(self.graph_path)
            if not isinstance(graph, nx.DiGraph):
                raise TypeError(
                    f"Expected nx.DiGraph at '{self.graph_path}', got {type(graph).__name__}"
                )
            self._graph = graph
        return self._graph
