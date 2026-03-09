from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ...core import SearchableGraphContainer
from ..contracts import RetrievedNode, RetrievalResult
from .base import BaseRetriever


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


class OneHopRetriever(BaseRetriever):
    name = "one-hop"

    def retrieve(
        self,
        graph: SearchableGraphContainer,
        query: str,
        query_vector: Sequence[float],
        *,
        index_name: str,
        top_k: int,
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

        seeds = graph.search(index_name, list(query_vector), k=top_k, **kwargs)
        seed_ids: List[str] = []
        seed_score_by_id: Dict[str, Optional[float]] = {}

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

        seed_ids = _dedup_preserve_order(seed_ids)
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
            visualizer.update_session(
                session_id,
                nodes=[
                    {
                        "id": node_id,
                        "style": {
                            "color": {"background": "#ffecb3", "border": "#fb8c00"},
                            "borderWidth": 3,
                        },
                    }
                    for node_id in one_hop_ids
                ],
                edges=[
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "relation": edge["relation"],
                        "style": {"width": 3},
                    }
                    for edge in one_hop_edges
                ],
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
