from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core import SearchableGraphContainer
from .contracts import ChatRequest, ChatResponse
from .retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

# Retrievers that do NOT need an embedding vector to operate
_EMBEDDING_FREE_RETRIEVERS = {"one-hop", "graph-hop", "graph-2-hop", "graph"}


class GraphRAGPipeline:
    def __init__(
        self,
        *,
        embedder: Any,
        generator: Any,
        retrievers: Dict[str, BaseRetriever],
        default_retrieval: str = "one-hop",
        retrieval_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.retrievers = dict(retrievers)
        self.default_retrieval = default_retrieval
        self.retrieval_aliases = dict(retrieval_aliases or {})

    def _resolve_retrieval(self, raw_name: str) -> str:
        name = str(raw_name or "").strip().lower()
        if not name:
            name = self.default_retrieval
        name = self.retrieval_aliases.get(name, name)
        if name not in self.retrievers:
            supported = ", ".join(sorted(self.retrievers.keys()))
            raise ValueError(f"Unsupported retrieval method: {raw_name}. Supported: {supported}")
        return name

    def _needs_embedding(self, retrieval_name: str) -> bool:
        """Return True if the retriever requires a query embedding vector."""
        return retrieval_name not in _EMBEDDING_FREE_RETRIEVERS

    def run(
        self,
        *,
        graph: SearchableGraphContainer,
        request: ChatRequest,
        visualizer: Optional[Any] = None,
    ) -> ChatResponse:
        retrieval_name = self._resolve_retrieval(request.retrieval)
        retriever = self.retrievers[retrieval_name]

        query_vector: List[float] = []
        if self._needs_embedding(retrieval_name):
            try:
                query_vector = self.embedder.embed(
                    request.message,
                    provider=request.embedding_provider,
                    model=request.embedding_model,
                )
            except Exception as exc:
                logger.warning(
                    "Embedding failed (%s: %s). Falling back to one-hop retrieval without vector.",
                    type(exc).__name__,
                    exc,
                )
                # Try to fall back to one-hop (no vector needed)
                if "one-hop" in self.retrievers:
                    retrieval_name = "one-hop"
                    retriever = self.retrievers["one-hop"]
                    query_vector = []
                else:
                    raise

        retrieval_result = retriever.retrieve(
            graph=graph,
            query=request.message,
            query_vector=query_vector,
            index_name=request.index_name,
            top_k=request.top_k,
            session_id=request.session_id,
            visualizer=visualizer,
        )

        if visualizer is not None and request.session_id:
            visualizer.update_session(
                request.session_id,
                progress={"current": 100, "total": 100, "message": "Generating answer"},
            )

        answer = self.generator.generate(
            question=request.message,
            history=request.history,
            context_chunks=retrieval_result.context_chunks,
            model=request.model,
        )

        return ChatResponse(
            answer=answer,
            retrieval=retrieval_result.method,
            session_id=request.session_id,
            context_chunks=retrieval_result.context_chunks,
            nodes=[node.to_dict() for node in retrieval_result.nodes],
            edges=list(retrieval_result.edges),
            metadata=dict(retrieval_result.metadata),
        )
