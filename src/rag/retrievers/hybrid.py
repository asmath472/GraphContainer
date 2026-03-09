from __future__ import annotations

from typing import Any, Optional, Sequence

from ...core import SearchableGraphContainer
from ..contracts import RetrievalResult
from .base import BaseRetriever


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def retrieve(
        self,
        graph: SearchableGraphContainer,
        query: str,
        query_vector: Sequence[float],
        *,
        index_name: str,
        top_k: int,
        session_id: Optional[str] = None,
        visualizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError("HybridRetriever is a placeholder. Implement `retrieve` in this class.")
