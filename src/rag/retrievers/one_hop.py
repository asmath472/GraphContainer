from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core import SearchableGraphContainer
from ..contracts import RetrievedNode, RetrievalResult
from .base import BaseRetriever
from .utils import dedup_preserve_order, embed_query, keyword_fallback_seed_scores, vector_seed_scores

import time

class OneHopRetriever(BaseRetriever):
    name = "one-hop"

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
        if top_k <= 0:
            top_k = 1

        if visualizer is not None and session_id:
            visualizer.update_session(
                session_id,
                progress={"current": 10, "total": 100, "message": "Running one-hop retrieval"},
            )

        seed_ids: List[str] = []
        seed_score_by_id: Dict[str, Optional[float]] = {}
        embedding_provider = str(kwargs.pop("embedding_provider", "hf")).strip().lower()
        embedding_model = str(kwargs.pop("embedding_model", "BAAI/bge-m3")).strip()
        embedding_error_policy = kwargs.pop("embedding_error_policy", "raise")
        query_vector = embed_query(
            query=query,
            embedding_service=embedding_service,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_error_policy=str(embedding_error_policy),
            visualizer=visualizer,
            session_id=session_id,
            retriever_name=self.name,
        )

        vector_search_error: Optional[str] = None
        if query_vector:
            # ── Vector search path (normal) ──────────────────────────────
            vector_seed_ids, vector_seed_scores_by_id, vector_search_error = vector_seed_scores(
                graph,
                index_name=index_name,
                query_vector=list(query_vector),
                top_k=top_k,
                search_kwargs=kwargs,
            )
            seed_ids.extend(vector_seed_ids)
            seed_score_by_id.update(vector_seed_scores_by_id)

        if not query_vector or vector_search_error is not None:
            # ── Keyword fallback (no embedding available) ─────────────────
            fallback_ids, fallback_scores = keyword_fallback_seed_scores(graph, query=query, top_k=top_k)
            if not seed_ids:
                seed_ids.extend(fallback_ids)
                seed_score_by_id.update(fallback_scores)

        if visualizer is not None and session_id and vector_search_error is not None:
            visualizer.update_session(
                session_id,
                metadata={"vector_search_fallback_reason": vector_search_error},
            )

        if visualizer is not None and session_id:
            visualizer.update_session(
                session_id,
                nodes=[
                    {
                        "id": sid,
                        "style": {
                            "color": {"background": "#ffeb3b", "border": "#f44336"},
                            "borderWidth": 5,
                        },
                    }
                    for sid in seed_ids
                ],
                progress={
                    "current": 35,
                    "total": 100,
                    "message": f"Seed nodes found: {len(seed_ids)}",
                },
            )

        one_hop_ids: List[str] = []
        one_hop_edges: List[Dict[str, Any]] = []

        for seed_id in seed_ids:
            for edge in graph.get_neighbors(seed_id):
                target_id = str(edge.target)
                one_hop_ids.append(target_id)
                one_hop_edges.append(
                    {
                        "source": str(edge.source),
                        "target": target_id,
                        "relation": str(edge.relation),
                        "weight": float(edge.weight),
                    }
                )

        one_hop_ids = dedup_preserve_order(one_hop_ids)
        ordered_node_ids = dedup_preserve_order(seed_ids + one_hop_ids)

        context_chunks: List[str] = []
        retrieved_nodes: List[RetrievedNode] = []
        for node_id in ordered_node_ids:
            node = graph.get_node(node_id)
            if node is None:
                continue
            text = str(node.text or "").strip()
            if text:
                context_chunks.append(f"[{node_id}] {text}")
            retrieved_nodes.append(
                RetrievedNode(
                    id=node.id,
                    text=text,
                    score=seed_score_by_id.get(node_id),
                    metadata=dict(node.metadata),
                )
            )

        context_chunks = dedup_preserve_order(context_chunks)

        if visualizer is not None and session_id:
            for node_id in one_hop_ids:
                if node_id in seed_ids:
                    continue
                visualizer.update_session(
                    session_id,
                    nodes=[
                        {
                            "id": node_id,
                            "style": {
                                "color": {"background": "#c8e6c9", "border": "#4caf50"},
                                "borderWidth": 3,
                            },
                        }
                    ],
                )
                time.sleep(0.1)  
            visualizer.update_session(
                session_id,
                edges=[
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "relation": edge["relation"],
                        "style": {"width": 3},
                    } for edge in one_hop_edges
                ],
            )

            visualizer.update_session(
                session_id,
                progress={"current": 100, "total": 100, "message": "Retrieval complete"},
            )

        return RetrievalResult(
            method=self.name,
            query=query,
            seed_nodes=seed_ids,
            nodes=retrieved_nodes,
            edges=one_hop_edges,
            context_chunks=context_chunks,
            metadata={
                "index_name": index_name,
                "top_k": top_k,
                "seed_count": len(seed_ids),
                "edge_count": len(one_hop_edges),
                "keyword_fallback_used": bool(not query_vector or vector_search_error is not None),
                "vector_search_error": vector_search_error,
            },
        )
