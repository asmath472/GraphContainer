from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional

from ..core import SearchableGraphContainer
from .contracts import ChatRequest
from .embeddings import EmbeddingService
from .generator import OpenAIChatGenerator
from .pipeline import GraphRAGPipeline
from .retrievers import FastInsightRetriever, HybridRetriever, OneHopRetriever, VectorRetriever


class GraphRAGService:
    def __init__(
        self,
        graph: SearchableGraphContainer,
        *,
        visualizer: Optional[Any] = None,
        pipeline: Optional[GraphRAGPipeline] = None,
        default_chat_model: str = "gpt-5-nano",
        default_embedding_provider: str = "hf",
        default_hf_embedding_model: str = "BAAI/bge-m3",
        default_openai_embedding_model: str = "text-embedding-3-small",
        hf_embedding_models: Optional[Iterable[str]] = None,
        openai_embedding_models: Optional[Iterable[str]] = None,
        embedding_model_catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.visualizer = visualizer

        if pipeline is None:
            embedding_service = EmbeddingService(
                default_provider=default_embedding_provider,
                default_hf_model=default_hf_embedding_model,
                default_openai_model=default_openai_embedding_model,
                hf_models=hf_embedding_models,
                openai_models=openai_embedding_models,
                model_catalog=embedding_model_catalog,
            )
            generator = OpenAIChatGenerator(
                default_model=default_chat_model,
            )
            pipeline = GraphRAGPipeline(
                embedding_service=embedding_service,
                generator=generator,
                retrievers={
                    "one-hop": OneHopRetriever(),
                    "vector": VectorRetriever(),
                    "hybrid": HybridRetriever(),
                    "fastinsight": FastInsightRetriever(),
                },
                default_retrieval="one-hop",
                retrieval_aliases={
                    "graph-hop": "one-hop",
                    "graph-2-hop": "one-hop",
                    "graph": "one-hop",
                    "fi": "fastinsight",
                },
            )
        self.pipeline = pipeline

    def list_embedding_options(self) -> Dict[str, Any]:
        embedding_service = getattr(self.pipeline, "embedding_service", None)
        if embedding_service is not None and hasattr(embedding_service, "list_options"):
            payload = embedding_service.list_options()
            if isinstance(payload, dict):
                return payload
        return {
            "default_provider": "hf",
            "default_model": "BAAI/bge-m3",
            "default_value": "hf:BAAI/bge-m3",
            "providers": [],
            "options": [],
        }

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
                            "embedding_error_policy": request.embedding_error_policy,
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
                            "embedding_error_policy": request.embedding_error_policy,
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
                    "embedding_error_policy": request.embedding_error_policy,
                },
                progress={"message": "Starting graph retrieval"},
            )

        request = replace(request, session_id=session_id)
        response = self.pipeline.run(
            graph=self.graph,
            request=request,
            visualizer=self.visualizer,
        )
        if self.visualizer is not None and session_id:
            self.visualizer.update_session(
                session_id,
                metadata={
                    "graph": request.graph,
                    "model": request.model,
                    "retrieval": request.retrieval,
                    "embedding_provider": request.embedding_provider,
                    "embedding_model": request.embedding_model,
                    "embedding_error_policy": request.embedding_error_policy,
                    "llm_answer": response.answer,
                    "retrieval_elapsed_ms": response.metadata.get("retrieval_elapsed_ms"),
                    "generation_elapsed_ms": response.metadata.get("generation_elapsed_ms"),
                    "total_elapsed_ms": response.metadata.get("total_elapsed_ms"),
                },
                progress={"message": "Answer generated"},
            )
        return response.to_dict()
