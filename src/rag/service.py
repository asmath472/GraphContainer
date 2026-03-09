from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, Optional

from ..core import SearchableGraphContainer
from .contracts import ChatRequest
from .embeddings import EmbedderRouter
from .generator import OpenAIChatGenerator
from .pipeline import GraphRAGPipeline
from .retrievers import HybridRetriever, OneHopRetriever, VectorRetriever


class GraphRAGService:
    def __init__(
        self,
        graph: SearchableGraphContainer,
        *,
        visualizer: Optional[Any] = None,
        pipeline: Optional[GraphRAGPipeline] = None,
    ) -> None:
        self.graph = graph
        self.visualizer = visualizer

        if pipeline is None:
            embedder = EmbedderRouter(
                default_provider=os.getenv("GRAPH_RAG_EMBEDDER", "bge"),
                default_bge_model=os.getenv("GRAPH_RAG_BGE_MODEL", "BAAI/bge-m3"),
                default_openai_model=os.getenv("GRAPH_RAG_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            )
            generator = OpenAIChatGenerator(
                default_model=os.getenv("GRAPH_RAG_CHAT_MODEL", "gpt-4o-mini"),
            )
            pipeline = GraphRAGPipeline(
                embedder=embedder,
                generator=generator,
                retrievers={
                    "one-hop": OneHopRetriever(),
                    "vector": VectorRetriever(),
                    "hybrid": HybridRetriever(),
                },
                default_retrieval="one-hop",
                retrieval_aliases={
                    "graph-hop": "one-hop",
                    "graph-2-hop": "one-hop",
                    "graph": "one-hop",
                },
            )
        self.pipeline = pipeline

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.graph, "search"):
            raise RuntimeError(
                "This graph does not support retrieval search. "
                "Load a SearchableGraphContainer with an attached index."
            )

        request = ChatRequest.from_payload(payload)

        session_id = request.session_id
        if self.visualizer is not None:
            if session_id is None:
                session_id = self.visualizer.create_session(
                        metadata={
                            "graph": request.graph,
                            "model": request.model,
                            "retrieval": request.retrieval,
                            "embedding_provider": request.embedding_provider,
                            "embedding_model": request.embedding_model,
                        }
                    )
            else:
                if not self.visualizer.has_session(session_id):
                    session_id = self.visualizer.create_session(
                        metadata={
                            "graph": request.graph,
                            "model": request.model,
                            "retrieval": request.retrieval,
                            "embedding_provider": request.embedding_provider,
                            "embedding_model": request.embedding_model,
                        }
                    )
                else:
                    self.visualizer.clear_session(session_id)

            self.visualizer.update_session(
                session_id,
                metadata={
                    "graph": request.graph,
                    "model": request.model,
                    "retrieval": request.retrieval,
                    "embedding_provider": request.embedding_provider,
                    "embedding_model": request.embedding_model,
                },
                progress={"current": 0, "total": 100, "message": "Starting graph retrieval"},
            )

        request = replace(request, session_id=session_id)
        response = self.pipeline.run(
            graph=self.graph,
            request=request,
            visualizer=self.visualizer,
        )
        return response.to_dict()
