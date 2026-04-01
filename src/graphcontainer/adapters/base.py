# src/GraphContainer/adapters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core import SimpleGraphContainer


class GraphAdapterError(RuntimeError):
    """Base error for adapter-level failures."""


class UnsupportedSourceError(GraphAdapterError):
    """Raised when an adapter cannot import the given source."""


class GraphAdapter(ABC):
    """Base interface for importing / exporting graphs."""

    def __init__(self, *, name: str, version: str = "0.1.0"):
        self.name = name
        self.version = version

    @abstractmethod
    def can_import(self, source: Any) -> bool:
        """Return True if this adapter can import the given source."""

    @abstractmethod
    def import_graph(
        self,
        source: Any,
        container: Optional[SimpleGraphContainer] = None,
        *,
        keep_source_reference: bool = False,
    ) -> SimpleGraphContainer:
        """Load a graph from source into a container.

        keep_source_reference is adapter-defined optional metadata.
        """

    @abstractmethod
    def export_graph(
        self,
        container: SimpleGraphContainer,
        destination: Any,
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Serialize container into destination format and return result metadata."""
