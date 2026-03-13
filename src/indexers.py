# src/GraphContainer/indexers.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .index import BaseIndexer


def to_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    return None


class InMemoryVectorIndexer(BaseIndexer):
    """Simple in-memory vector indexer for fallback/testing."""

    def __init__(self):
        self._rows: Dict[str, Dict[str, Any]] = {}

    def describe_store(self) -> Dict[str, Any]:
        return {"library": "in_memory", "size": len(self._rows)}

    def add(self, id: str, content: Any) -> None:
        if not isinstance(content, dict):
            raise ValueError("content must be dict with optional embedding/document/metadata.")
        emb = to_float_list(content.get("embedding"))
        if emb is None:
            raise ValueError("embedding is required.")
        self._rows[id] = {
            "embedding": emb,
            "document": content.get("document"),
            "metadata": content.get("metadata"),
        }

    def search(self, query: Any, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        k = kwargs.pop("k", None)
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if k is not None:
            top_k = int(k)

        q = to_float_list(query)
        if q is None or top_k <= 0:
            return []
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for item_id, row in self._rows.items():
            v = row["embedding"]
            if len(v) != len(q):
                continue
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            sim = sum(a * b for a, b in zip(q, v)) / (qn * vn)
            scored.append((sim, item_id, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": item_id,
                "document": row.get("document"),
                "metadata": row.get("metadata"),
                "distance": 1.0 - sim,
                "embedding": row.get("embedding"),
            }
            for sim, item_id, row in scored[:top_k]
        ]


class ChromaCollectionIndexer(BaseIndexer):
    """Wrapper for Chroma collection to fit BaseIndexer."""

    def __init__(
        self,
        collection: Any,
        persist_path: Optional[str] = None,
        distance_metric: str = "cosine",
    ):
        self.collection = collection
        self.persist_path = persist_path
        self.distance_metric = distance_metric

    def describe_store(self) -> Dict[str, Any]:
        collection_name = getattr(self.collection, "name", None)
        return {
            "library": "chromadb",
            "path": self.persist_path,
            "collection_name": collection_name,
            "distance_metric": self.distance_metric,
        }

    def add(self, id: str, content: Any) -> None:
        if not isinstance(content, dict):
            raise ValueError("content must be dict with optional embedding/document/metadata.")

        payload: Dict[str, Any] = {"ids": [id]}
        embedding = to_float_list(content.get("embedding"))
        if embedding is not None:
            payload["embeddings"] = [embedding]
        if "document" in content:
            payload["documents"] = [content["document"]]
        if "metadata" in content:
            payload["metadatas"] = [content["metadata"]]

        if len(payload) == 1:
            raise ValueError("nothing to upsert.")
        self.collection.upsert(**payload)

    def search(
        self,
        query_vector: Any,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        k = kwargs.pop("k", None)
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if k is not None:
            top_k = int(k)

        query_vec = to_float_list(query_vector)
        if query_vec is None:
            raise ValueError("query_vector must be a numeric vector.")
        if top_k <= 0:
            return []

        payload: Dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": top_k,
            "include": ["metadatas", "documents", "distances", "embeddings"],
        }
        if filter:
            payload["where"] = filter

        result = self.collection.query(**payload)

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        embs = (result.get("embeddings") or [[]])[0]

        rows: List[Dict[str, Any]] = []
        for i, item_id in enumerate(ids):
            rows.append(
                {
                    "id": item_id,
                    "document": docs[i] if i < len(docs) else None,
                    "metadata": metas[i] if i < len(metas) else None,
                    "distance": dists[i] if i < len(dists) else None,
                    "embedding": embs[i] if i < len(embs) else None,
                }
            )
        return rows


# TODO: PGVector Indexer
class PGVectorIndexer(BaseIndexer):
    def __init__(self, connection: Any, table_name: str):
        self.connection = connection
        self.table_name = table_name

    def describe_store(self) -> Dict[str, Any]:
        return {"library": "pgvector", "table_name": self.table_name}

    def add(self, id: str, content: Any) -> None:
        raise NotImplementedError("PGVectorIndexer is not implemented yet.")

    def search(self, query: Any, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("PGVectorIndexer is not implemented yet.")
