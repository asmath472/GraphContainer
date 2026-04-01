from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..core import SearchableGraphContainer
from .contracts import ChatRequest, ChatResponse
from .retrievers.base import BaseRetriever

class GraphRAGPipeline:
    def __init__(
        self,
        *,
        embedding_service: Any,
        generator: Any,
        retrievers: Dict[str, BaseRetriever],
        default_retrieval: str = "one-hop",
        retrieval_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.embedding_service = embedding_service
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

    def run(
        self,
        *,
        graph: SearchableGraphContainer,
        request: ChatRequest,
        visualizer: Optional[Any] = None,
    ) -> ChatResponse:
        total_start = time.perf_counter()
        retrieval_name = self._resolve_retrieval(request.retrieval)
        retriever = self.retrievers[retrieval_name]

        retrieval_start = time.perf_counter()
        retrieval_result = retriever.retrieve(
            graph=graph,
            query=request.message,
            index_name=request.index_name,
            top_k=request.top_k,
            embedding_service=self.embedding_service,
            session_id=request.session_id,
            visualizer=visualizer,
            embedding_provider=request.embedding_provider,
            embedding_model=request.embedding_model,
            embedding_error_policy=request.embedding_error_policy,
        )
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0

        if visualizer is not None and request.session_id:
            visualizer.update_session(
                request.session_id,
                metadata={"retrieval_elapsed_ms": round(retrieval_elapsed_ms, 2)},
                progress={"message": "Generating answer"},
            )

        generation_start = time.perf_counter()
        answer = self.generator.generate(
            question=request.message,
            history=request.history,
            context_chunks=retrieval_result.context_chunks,
            model=request.model,
        )
        generation_elapsed_ms = (time.perf_counter() - generation_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        response_metadata = dict(retrieval_result.metadata)
        response_metadata.update(
            {
                "retrieval_elapsed_ms": round(retrieval_elapsed_ms, 2),
                "generation_elapsed_ms": round(generation_elapsed_ms, 2),
                "total_elapsed_ms": round(total_elapsed_ms, 2),
            }
        )

        return ChatResponse(
            answer=answer,
            retrieval=retrieval_result.method,
            session_id=request.session_id,
            context_chunks=retrieval_result.context_chunks,
            nodes=[node.to_dict() for node in retrieval_result.nodes],
            edges=list(retrieval_result.edges),
            metadata=response_metadata,
        )
