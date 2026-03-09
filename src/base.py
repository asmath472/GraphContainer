from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .types import EdgeRecord, NodeRecord


class BaseGraphContainer(ABC):
    """Base graph container interface."""

    @abstractmethod
    def add_node(self, node: NodeRecord) -> None:
        """Add or replace a node."""

    @abstractmethod
    def add_edge(self, edge: EdgeRecord) -> None:
        """Add an edge."""

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        """Return node by id."""

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[EdgeRecord]:
        """Return outgoing edges from the given node id."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist graph data to storage."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load graph data from storage."""
