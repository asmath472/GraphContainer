from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseIndexer(ABC):
    @abstractmethod
    def describe_store(self) -> Dict[str, Any]:
        """Describe backing store metadata (for manifest/export)."""
    
    @abstractmethod
    def add(self, id: str, content: Any) -> None:
        """Insert or upsert one record into index."""
    
    @abstractmethod
    def search(self, query: Any, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Search index with query and return normalized rows."""
    
