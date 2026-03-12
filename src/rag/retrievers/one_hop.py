from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core import SearchableGraphContainer
from ..contracts import RetrievedNode, RetrievalResult
from .base import BaseRetriever

import time

def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _normalize_error_policy(value: Any) -> str:
    policy = str(value or "raise").strip().lower()
    if policy in {"fallback", "soft"}:
        return "fallback"
    return "raise"


def _embed_query(
    *,
    query: str,
    embedding_service: Optional[Any],
    embedding_provider: str,
    embedding_model: str,
    embedding_error_policy: str,
    visualizer: Optional[Any],
    session_id: Optional[str],
    retriever_name: str,
) -> List[float]:
    if embedding_service is None:
        return []

    try:
        vector = embedding_service.embed(
            query,
            provider=embedding_provider,
            model=embedding_model,
        )
        return [float(x) for x in vector]
    except Exception as exc:
        message = (
            "Embedding failed "
            f"(provider={embedding_provider}, model={embedding_model}, retrieval={retriever_name}, "
            f"error={type(exc).__name__}: {exc})"
        )
        if visualizer is not None and session_id:
            visualizer.update_session(
                session_id,
                metadata={"embedding_error": message},
                progress={"current": 22, "total": 100, "message": "Embedding failed"},
            )
        if _normalize_error_policy(embedding_error_policy) == "fallback":
            return []
        raise RuntimeError(message) from exc


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
        query_vector = _embed_query(
            query=query,
            embedding_service=embedding_service,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_error_policy=str(embedding_error_policy),
            visualizer=visualizer,
            session_id=session_id,
            retriever_name=self.name,
        )

        if query_vector:
            # ── Vector search path (normal) ──────────────────────────────
            seeds = graph.search(index_name, list(query_vector), k=top_k, **kwargs)
            for item in seeds:
                if not isinstance(item, dict):
                    continue
                if item.get("id") is None:
                    continue
                node_id = str(item["id"])
                seed_ids.append(node_id)
                raw_distance = item.get("distance")
                score: Optional[float] = None
                if isinstance(raw_distance, (int, float)):
                    score = 1.0 - float(raw_distance)
                seed_score_by_id[node_id] = score
        else:
            # ── Keyword fallback (no embedding available) ─────────────────
            # Score nodes by how many query tokens appear in their id/text.
            tokens = [t.lower() for t in query.split() if t.strip()]
            scored: List[tuple] = []
            for node_id, node in graph.nodes.items():
                haystack = f"{node_id} {node.text or ''}".lower()
                hits = sum(1 for tok in tokens if tok in haystack)
                if hits > 0:
                    scored.append((hits, node_id))
            scored.sort(key=lambda x: -x[0])
            for hits, node_id in scored[:top_k]:
                seed_ids.append(node_id)
                seed_score_by_id[node_id] = float(hits) / max(len(tokens), 1)

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

        one_hop_ids = _dedup_preserve_order(one_hop_ids)
        ordered_node_ids = _dedup_preserve_order(seed_ids + one_hop_ids)

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

        context_chunks = _dedup_preserve_order(context_chunks)

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
            },
        )
