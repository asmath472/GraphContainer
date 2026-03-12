from __future__ import annotations

from typing import Any, Optional

from ...core import SearchableGraphContainer
from ..contracts import RetrievalResult
from .base import BaseRetriever


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def retrieve(
        self,
        graph: SearchableGraphContainer,
        query: str,
        *,
        index_name: str,
        top_k: int,
        embedding_service: Optional[Any] = None,
        session_id: Optional[str] = None,
        visualizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError("HybridRetriever is a placeholder. Implement `retrieve` in this class.")
