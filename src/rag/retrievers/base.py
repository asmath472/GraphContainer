from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from ...core import SearchableGraphContainer
from ..contracts import RetrievalResult

if TYPE_CHECKING:
    from ...visualizer.live_visualizer import LiveGraphVisualizer


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(
        self,
        graph: SearchableGraphContainer,
        query: str,
        *,
        index_name: str,
        top_k: int,
        embedding_service: Optional[Any] = None,
        session_id: Optional[str] = None,
        visualizer: Optional["LiveGraphVisualizer"] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError
