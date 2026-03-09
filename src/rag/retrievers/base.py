from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, TYPE_CHECKING

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
        query_vector: Sequence[float],
        *,
        index_name: str,
        top_k: int,
        session_id: Optional[str] = None,
        visualizer: Optional["LiveGraphVisualizer"] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError
