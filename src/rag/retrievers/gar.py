from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ...core import SearchableGraphContainer
from ..contracts import RetrievedNode, RetrievalResult
from .fastinsight import (
    FastInsightRetriever,
    _build_graph_stats,
    _normalize_graph_construction_method,
    _score_of,
    _title_from_content,
    _to_float_list,
    _vector_search,
)
from .utils import dedup_preserve_order, embed_query, keyword_fallback_seed_scores, vector_seed_scores


def _rerank_nodes(
    *,
    query: str,
    nodes: List[Dict[str, Any]],
    ranker: "_GARRanker",
) -> List[Dict[str, Any]]:
    if not nodes:
        return []

    scores = ranker.predict(query, nodes)
    probabilities = scores.sigmoid()

    score_list = scores.detach().cpu().tolist()
    probability_list = probabilities.detach().cpu().tolist()
    for idx, node in enumerate(nodes):
        node["score"] = float(score_list[idx])
        node["probability"] = float(probability_list[idx])

    nodes.sort(key=_score_of, reverse=True)
    return nodes


class _GARRanker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", gpu_id: int = 0):
        self.model_name = model_name
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        query: str,
        nodes: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        if not nodes:
            return torch.tensor([], device=self.device)

        pairs = [[query, str(node.get("content", ""))] for node in nodes]
        all_scores: List[torch.Tensor] = []

        with torch.no_grad():
            for idx in range(0, len(pairs), batch_size):
                batch_pairs = pairs[idx : idx + batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                else:
                    logits = logits.view(-1)
                all_scores.append(logits)

        if not all_scores:
            return torch.tensor([], device=self.device)
        return torch.cat(all_scores, dim=0)


def _edge_payload(edge: Any) -> Dict[str, Any]:
    return {
        "source": str(edge.source),
        "target": str(edge.target),
        "relation": str(edge.relation),
        "weight": float(edge.weight),
    }


def _dedup_node_pool(
    *,
    nodes: List[Dict[str, Any]],
    database_construction_method: str,
) -> List[Dict[str, Any]]:
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = _title_from_content(node, database_construction_method)
        if node_id in candidate_by_id:
            existing_origin = str(candidate_by_id[node_id].get("origin", ""))
            incoming_origin = str(node.get("origin", ""))
            if existing_origin != incoming_origin and incoming_origin:
                candidate_by_id[node_id]["origin"] = "+".join(
                    dedup_preserve_order([existing_origin, incoming_origin])
                )
            if not candidate_by_id[node_id].get("content") and node.get("content"):
                candidate_by_id[node_id]["content"] = node.get("content")
            if candidate_by_id[node_id].get("metadata") is None and node.get("metadata") is not None:
                candidate_by_id[node_id]["metadata"] = node.get("metadata")
            if candidate_by_id[node_id].get("embedding") is None and node.get("embedding") is not None:
                candidate_by_id[node_id]["embedding"] = node.get("embedding")
            if candidate_by_id[node_id].get("distance") is None and node.get("distance") is not None:
                candidate_by_id[node_id]["distance"] = node.get("distance")
            continue
        candidate_by_id[node_id] = dict(node)
    return list(candidate_by_id.values())


def _ordered_remaining_nodes(
    *,
    ordered_ids: List[str],
    node_by_id: Dict[str, Dict[str, Any]],
    excluded_ids: Set[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for node_id in ordered_ids:
        if node_id in excluded_ids:
            continue
        node = node_by_id.get(node_id)
        if node is None:
            continue
        output.append(node)
    return output


def _merge_origin(existing_origin: Any, incoming_origin: Any) -> str:
    return "+".join(
        dedup_preserve_order(
            [str(existing_origin or "").strip(), str(incoming_origin or "").strip()]
        )
    ).strip("+")


class GARRetriever(FastInsightRetriever):
    name = "gar"

    def __init__(self) -> None:
        super().__init__()
        self._gar_ranker: Optional[_GARRanker] = None
        self._gar_ranker_gpu_id: Optional[int] = None

    def _ensure_ranker(
        self,
        *,
        gpu_id: int,
        ranker: Optional[Any],
        verbose: bool,
    ) -> _GARRanker:
        if ranker is not None and all(hasattr(ranker, attr) for attr in ("predict", "device")):
            self._gar_ranker = ranker
            self._gar_ranker_gpu_id = gpu_id
            return ranker

        if isinstance(ranker, _GARRanker):
            self._gar_ranker = ranker
            self._gar_ranker_gpu_id = gpu_id
            return ranker

        if self._gar_ranker is not None and self._gar_ranker_gpu_id == gpu_id:
            return self._gar_ranker

        vprint = print if verbose else (lambda *args, **kwargs: None)
        new_ranker = _GARRanker(gpu_id=gpu_id)
        vprint("GAR ranker warm-up...")
        _ = new_ranker.predict("warmup_query", [{"content": "warmup_doc"}])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._gar_ranker = new_ranker
        self._gar_ranker_gpu_id = gpu_id
        return new_ranker

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
                progress={"message": "Running GAR retrieval"},
            )

        seed_top_k = int(kwargs.pop("seed_top_k", max(top_k, 10)))
        graph_top_k = int(kwargs.pop("graph_top_k", max(top_k, 10)))
        final_top_k = int(kwargs.pop("final_top_k", top_k))
        batch_size = int(kwargs.pop("batch_size", 1))
        database_construction_method = _normalize_graph_construction_method(
            kwargs.pop("database_construction_method", "component_graph")
        )
        verbose = bool(kwargs.pop("verbose", False))
        gpu_id = int(kwargs.pop("gpu_id", 0))
        ranker = kwargs.pop("ranker", kwargs.pop("granker", None))
        query_vector_override = _to_float_list(kwargs.pop("query_vector", None))
        embedding_provider = str(kwargs.pop("embedding_provider", "hf")).strip().lower()
        embedding_model = str(kwargs.pop("embedding_model", "BAAI/bge-m3")).strip()
        embedding_error_policy = str(kwargs.pop("embedding_error_policy", "raise"))

        if seed_top_k <= 0:
            seed_top_k = max(top_k, 10)
        if graph_top_k <= 0:
            graph_top_k = max(top_k, 10)
        if final_top_k <= 0:
            final_top_k = top_k
        if batch_size <= 0:
            batch_size = 1

        self._maybe_populate_graph_embeddings(graph=graph, index_name=index_name)

        if query_vector_override is not None:
            query_vector = query_vector_override
        else:
            query_vector = embed_query(
                query=query,
                embedding_service=embedding_service,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_error_policy=embedding_error_policy,
                visualizer=visualizer,
                session_id=session_id,
                retriever_name=self.name,
            )

        vector_search_error: Optional[str] = None
        vector_results: List[Dict[str, Any]] = []
        seed_ids: List[str] = []
        seed_score_by_id: Dict[str, Optional[float]] = {}

        if query_vector:
            vector_results = _vector_search(
                query_vec=query_vector,
                graph=graph,
                index_name=index_name,
                top_k=seed_top_k,
                database_construction_method=database_construction_method,
                search_kwargs=kwargs,
            )
            seed_ids = dedup_preserve_order(
                _title_from_content(node, database_construction_method) for node in vector_results
            )
            vector_seed_ids, vector_seed_scores_by_id, vector_search_error = vector_seed_scores(
                graph,
                index_name=index_name,
                query_vector=list(query_vector),
                top_k=seed_top_k,
                search_kwargs=kwargs,
            )
            if not seed_ids:
                seed_ids = dedup_preserve_order(vector_seed_ids)
            seed_score_by_id.update(vector_seed_scores_by_id)

        if not query_vector or (vector_search_error is not None and not seed_ids):
            fallback_ids, fallback_scores = keyword_fallback_seed_scores(graph, query=query, top_k=seed_top_k)
            seed_ids = dedup_preserve_order(seed_ids + fallback_ids)
            seed_score_by_id.update(fallback_scores)
            if not vector_results:
                for node_id in fallback_ids:
                    node = graph.get_node(node_id)
                    if node is None:
                        continue
                    vector_results.append(
                        {
                            "id": node_id,
                            "title": node_id,
                            "content": str(getattr(node, "text", "") or ""),
                            "metadata": dict(getattr(node, "metadata", {}) or {}),
                            "embedding": getattr(node, "embedding", None),
                            "origin": "VS",
                        }
                    )

        if visualizer is not None and session_id and vector_search_error is not None:
            visualizer.update_session(
                session_id,
                metadata={"vector_search_fallback_reason": vector_search_error},
            )

        for node in vector_results:
            node["origin"] = "VS"

        if visualizer is not None and session_id:
            visualizer.record(
                session_id,
                seed_ids,
                style={
                    "color": {"background": "#bbdefb", "border": "#1565c0"},
                    "borderWidth": 4,
                },
                message=f"GAR vector seeds: {len(seed_ids)}",
            )

        ranker = self._ensure_ranker(gpu_id=gpu_id, ranker=ranker, verbose=verbose)
        initial_pool = _dedup_node_pool(
            nodes=vector_results,
            database_construction_method=database_construction_method,
        )
        initial_by_id: Dict[str, Dict[str, Any]] = {
            _title_from_content(node, database_construction_method): dict(node) for node in initial_pool
        }
        for node_id in seed_ids:
            if node_id in initial_by_id:
                continue
            graph_node = graph.get_node(node_id)
            if graph_node is None:
                continue
            initial_by_id[node_id] = {
                "id": node_id,
                "title": node_id,
                "content": str(getattr(graph_node, "text", "") or ""),
                "metadata": dict(getattr(graph_node, "metadata", {}) or {}),
                "embedding": getattr(graph_node, "embedding", None),
                "origin": "VS",
            }
        initial_ranking_ids = [
            node_id for node_id in seed_ids if node_id in initial_by_id
        ]
        scored_ids: Set[str] = set()
        reranked_pool: List[Dict[str, Any]] = []
        frontier_by_id: Dict[str, Dict[str, Any]] = {}
        frontier_order: List[str] = []
        frontier_priority_by_id: Dict[str, float] = {}
        frontier_source_by_id: Dict[str, str] = {}
        edges_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        graph_expansion_ids: List[str] = []
        batch_history: List[Dict[str, Any]] = []
        out_neighbors, degree_by_id, _ = _build_graph_stats(graph)

        current_pool_name = "R0"
        while len(reranked_pool) < final_top_k:
            remaining_budget = final_top_k - len(reranked_pool)
            current_batch_size = min(batch_size, remaining_budget)
            if current_batch_size <= 0:
                break

            if current_pool_name == "R0":
                pool_nodes = _ordered_remaining_nodes(
                    ordered_ids=initial_ranking_ids,
                    node_by_id=initial_by_id,
                    excluded_ids=scored_ids,
                )
                next_pool_name = "F"
            else:
                pool_nodes = _ordered_remaining_nodes(
                    ordered_ids=frontier_order,
                    node_by_id=frontier_by_id,
                    excluded_ids=scored_ids,
                )
                next_pool_name = "R0"

            if not pool_nodes:
                alternate_nodes = _ordered_remaining_nodes(
                    ordered_ids=initial_ranking_ids if current_pool_name == "F" else frontier_order,
                    node_by_id=initial_by_id if current_pool_name == "F" else frontier_by_id,
                    excluded_ids=scored_ids,
                )
                if not alternate_nodes:
                    break
                current_pool_name = next_pool_name
                continue

            candidate_batch = pool_nodes[:current_batch_size]
            reranked_batch = _rerank_nodes(
                query=query,
                nodes=[dict(node) for node in candidate_batch],
                ranker=ranker,
            )

            batch_ids: List[str] = []
            for node in reranked_batch:
                node_id = _title_from_content(node, database_construction_method)
                batch_ids.append(node_id)
                scored_ids.add(node_id)
                reranked_pool.append(node)
                if node_id in initial_by_id:
                    initial_by_id[node_id]["score"] = node.get("score")
                    initial_by_id[node_id]["probability"] = node.get("probability")
                    initial_by_id[node_id]["origin"] = _merge_origin(
                        initial_by_id[node_id].get("origin"),
                        node.get("origin"),
                    )
                frontier_by_id.pop(node_id, None)
                frontier_priority_by_id.pop(node_id, None)
                frontier_source_by_id.pop(node_id, None)

            frontier_order = [node_id for node_id in frontier_order if node_id not in scored_ids]

            if visualizer is not None and session_id:
                visualizer.record(
                    session_id,
                    batch_ids,
                    style={
                        "color": {"background": "#fff59d", "border": "#ef6c00"},
                        "borderWidth": 5,
                    },
                    message=f"GAR scored batch from {current_pool_name}: {len(batch_ids)}",
                )

            new_frontier_ids: List[str] = []
            for source_node in reranked_batch:
                source_id = _title_from_content(source_node, database_construction_method)
                source_score = _score_of(source_node)
                for edge in graph.get_neighbors(source_id):
                    target_id = str(edge.target)
                    if target_id in scored_ids:
                        continue

                    edges_by_key[(str(edge.source), target_id, str(edge.relation))] = _edge_payload(edge)

                    target_node = graph.get_node(target_id)
                    if target_node is None:
                        continue

                    target_payload = {
                        "id": target_id,
                        "title": target_id,
                        "content": str(getattr(target_node, "text", "") or ""),
                        "metadata": dict(getattr(target_node, "metadata", {}) or {}),
                        "embedding": getattr(target_node, "embedding", None),
                        "origin": "GS",
                    }

                    if target_id in initial_by_id:
                        target_payload["origin"] = _merge_origin(
                            initial_by_id[target_id].get("origin", "VS"),
                            "GS",
                        )

                    existing_frontier = frontier_by_id.get(target_id)
                    if existing_frontier is not None:
                        existing_frontier["origin"] = _merge_origin(
                            existing_frontier.get("origin"),
                            target_payload.get("origin"),
                        )
                        if existing_frontier.get("content") in {"", None} and target_payload.get("content"):
                            existing_frontier["content"] = target_payload["content"]
                        if existing_frontier.get("embedding") is None and target_payload.get("embedding") is not None:
                            existing_frontier["embedding"] = target_payload["embedding"]
                    else:
                        frontier_by_id[target_id] = target_payload
                        new_frontier_ids.append(target_id)

                    previous_priority = frontier_priority_by_id.get(target_id)
                    if previous_priority is None or source_score > previous_priority:
                        frontier_priority_by_id[target_id] = source_score
                        frontier_source_by_id[target_id] = source_id

                    if target_id in initial_by_id:
                        initial_by_id[target_id]["origin"] = _merge_origin(
                            initial_by_id[target_id].get("origin"),
                            "GS",
                        )

            frontier_order = [
                node_id
                for node_id in sorted(
                    frontier_by_id.keys(),
                    key=lambda node_id: (
                        -frontier_priority_by_id.get(node_id, float("-inf")),
                        degree_by_id.get(node_id, 0),
                        node_id,
                    ),
                )
                if node_id not in scored_ids
            ][:graph_top_k]
            frontier_by_id = {node_id: frontier_by_id[node_id] for node_id in frontier_order}
            frontier_priority_by_id = {
                node_id: frontier_priority_by_id[node_id] for node_id in frontier_order if node_id in frontier_priority_by_id
            }
            frontier_source_by_id = {
                node_id: frontier_source_by_id[node_id] for node_id in frontier_order if node_id in frontier_source_by_id
            }
            graph_expansion_ids.extend(new_frontier_ids)
            batch_history.append(
                {
                    "pool": current_pool_name,
                    "batch_ids": batch_ids,
                    "frontier_size": len(frontier_order),
                }
            )

            if visualizer is not None and session_id and new_frontier_ids:
                visualizer.record(
                    session_id,
                    new_frontier_ids[:graph_top_k],
                    style={
                        "color": {"background": "#c8e6c9", "border": "#4caf50"},
                        "borderWidth": 3,
                    },
                    message=f"GAR frontier update: {len(new_frontier_ids)}",
                )

            current_pool_name = next_pool_name

        if len(reranked_pool) < final_top_k:
            for node in _ordered_remaining_nodes(
                ordered_ids=initial_ranking_ids,
                node_by_id=initial_by_id,
                excluded_ids=scored_ids,
            ):
                reranked_pool.append(dict(node))
                scored_ids.add(_title_from_content(node, database_construction_method))
                if len(reranked_pool) >= final_top_k:
                    break

        selected_nodes = reranked_pool[:final_top_k]
        edges = list(edges_by_key.values())

        retrieved_nodes: List[RetrievedNode] = []
        context_chunks: List[str] = []
        origin_stats: Dict[str, int] = {}
        for raw_node in selected_nodes:
            node_id = _title_from_content(raw_node, database_construction_method)
            text = str(raw_node.get("content", "") or "").strip()
            if not text:
                graph_node = graph.get_node(node_id)
                text = str(getattr(graph_node, "text", "") or "").strip()

            metadata: Dict[str, Any] = {}
            if isinstance(raw_node.get("metadata"), dict):
                metadata.update(raw_node["metadata"])
            if raw_node.get("origin") is not None:
                metadata["origin"] = raw_node.get("origin")
            if isinstance(raw_node.get("probability"), (int, float)):
                metadata["probability"] = float(raw_node["probability"])
            if isinstance(raw_node.get("distance"), (int, float)):
                metadata["distance"] = float(raw_node["distance"])
            if node_id in seed_score_by_id and seed_score_by_id[node_id] is not None:
                metadata["vector_seed_score"] = float(seed_score_by_id[node_id])

            retrieved_nodes.append(
                RetrievedNode(
                    id=node_id,
                    text=text,
                    score=float(raw_node["score"]) if isinstance(raw_node.get("score"), (int, float)) else None,
                    metadata=metadata,
                )
            )
            if text:
                context_chunks.append(f"[{node_id}] {text}")

            origin = str(raw_node.get("origin", "NULL"))
            origin_stats[origin] = origin_stats.get(origin, 0) + 1

        context_chunks = dedup_preserve_order(context_chunks)

        if visualizer is not None and session_id:
            visualizer.record(
                session_id,
                [node.id for node in retrieved_nodes],
                style={
                    "color": {"background": "#fff176", "border": "#d84315"},
                    "borderWidth": 7,
                },
                message="GAR retrieval complete",
            )

        return RetrievalResult(
            method=self.name,
            query=query,
            seed_nodes=seed_ids,
            nodes=retrieved_nodes,
            edges=edges,
            context_chunks=context_chunks,
            metadata={
                "index_name": index_name,
                "top_k": top_k,
                "seed_top_k": seed_top_k,
                "graph_top_k": graph_top_k,
                "final_top_k": final_top_k,
                "batch_size": batch_size,
                "seed_count": len(seed_ids),
                "graph_candidate_count": len(dedup_preserve_order(graph_expansion_ids)),
                "node_count": len(retrieved_nodes),
                "edge_count": len(edges),
                "origin_stats": origin_stats,
                "batch_history": batch_history,
                "frontier_size": len(frontier_order),
                "keyword_fallback_used": bool(not query_vector or (vector_search_error is not None and not seed_ids)),
                "vector_search_error": vector_search_error,
                "note": (
                    "GAR follows the adaptive re-ranking loop: initialize R0 from seed retrieval, "
                    "alternate scoring batches from R0 and the frontier F, expand F with 1-hop neighbors "
                    "prioritized by source scores, then backfill from R0 if the budget is not exhausted."
                ),
            },
        )
