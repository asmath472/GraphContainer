from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseGraphContainer
from .index import BaseIndexer
from .types import EdgeRecord, NodeRecord


class SimpleGraphContainer(BaseGraphContainer):
    def __init__(self):
        self.nodes: Dict[str, NodeRecord] = {}
        self.edges: List[EdgeRecord] = []
        self._adj: Dict[str, List[int]] = {}

    def add_node(self, node: NodeRecord) -> None:
        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])

    def add_edge(self, edge: EdgeRecord) -> None:
        # Ensure endpoint nodes exist.
        if edge.source not in self.nodes:
            self.add_node(NodeRecord(id=edge.source))
        if edge.target not in self.nodes:
            self.add_node(NodeRecord(id=edge.target))

        edge_idx = len(self.edges)
        self.edges.append(edge)
        self._adj.setdefault(edge.source, []).append(edge_idx)

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[EdgeRecord]:
        indices = self._adj.get(node_id, [])
        return [self.edges[i] for i in indices]

    def _normalize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        return value

    def _normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._normalize_value(value) for key, value in row.items()}

    def save(self, path: str) -> None:
        node_data = [n.model_dump() for n in self.nodes.values()]
        edge_data = [e.model_dump() for e in self.edges]
        pd.DataFrame(node_data).to_parquet(f"{path}_nodes.parquet")
        pd.DataFrame(edge_data).to_parquet(f"{path}_edges.parquet")

    def load(self, path: str) -> None:
        node_path = Path(f"{path}_nodes.parquet")
        edge_path = Path(f"{path}_edges.parquet")

        if not node_path.exists():
            raise FileNotFoundError(f"Node file not found: {node_path}")
        if not edge_path.exists():
            raise FileNotFoundError(f"Edge file not found: {edge_path}")

        node_df = pd.read_parquet(node_path)
        edge_df = pd.read_parquet(edge_path)

        self.nodes.clear()
        self.edges.clear()
        self._adj.clear()

        for _, row in node_df.iterrows():
            self.add_node(NodeRecord(**self._normalize_row(row.to_dict())))
        for _, row in edge_df.iterrows():
            self.add_edge(EdgeRecord(**self._normalize_row(row.to_dict())))


class SearchableGraphContainer(SimpleGraphContainer):
    def __init__(self):
        super().__init__()
        self._indexes: Dict[str, Optional[BaseIndexer]] = {
            "node_vector": None,
            "edge_vector": None,
        }

    def attach_index(self, name: str, indexer: BaseIndexer) -> None:
        self._indexes[name] = indexer

    def get_index(self, name: str) -> Optional[BaseIndexer]:
        return self._indexes.get(name)

    def list_indexes(self) -> List[str]:
        return [name for name, indexer in self._indexes.items() if indexer is not None]

    def search(self, index_name: str, query: Any, k: int = 5, **kwargs: Any):
        """
        Backward-compatible search wrapper:
        - accepts `k`
        - forwards to indexer with `top_k` first, then fallback to `k`
        """
        indexer = self._indexes.get(index_name)
        if indexer is None:
            raise ValueError(f"Index '{index_name}' is not attached.")

        top_k = kwargs.pop("top_k", k)
        try:
            return indexer.search(query, top_k=top_k, **kwargs)
        except TypeError:
            return indexer.search(query, k=top_k, **kwargs)
