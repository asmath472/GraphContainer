from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ...core import SearchableGraphContainer
from ..contracts import RetrievedNode, RetrievalResult
from .base import BaseRetriever
from .utils import dedup_preserve_order, embed_query


def _title_from_content(node: Dict[str, Any], construction_method: str = "fastinsight") -> str:
    title = node.get("title")
    if isinstance(title, str) and title:
        return title

    if construction_method == "lightrag":
        content = str(node.get("content", ""))
        node["title"] = content.split("\n", 1)[0].strip()
    else:
        node["title"] = str(node.get("id", ""))
    return str(node["title"])


def _score_of(node: Dict[str, Any]) -> float:
    raw = node.get("score", node.get("probability", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _to_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        output: List[float] = []
        for item in value:
            try:
                output.append(float(item))
            except (TypeError, ValueError):
                return None
        return output
    return None


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _node_embedding(node: Any) -> Optional[List[float]]:
    if node is None:
        return None
    emb = _to_float_list(getattr(node, "embedding", None))
    if emb is not None:
        return emb
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        emb = _to_float_list(metadata.get("embedding"))
        if emb is not None:
            return emb
    return None


def _build_graph_stats(
    graph: SearchableGraphContainer,
) -> Tuple[Dict[str, Set[str]], Dict[str, int], Set[str]]:
    out_neighbors: Dict[str, Set[str]] = {}
    degree_by_id: Dict[str, int] = {}
    node_ids = {str(node_id) for node_id in graph.nodes.keys()}

    for edge in graph.edges:
        source = str(edge.source)
        target = str(edge.target)

        out_neighbors.setdefault(source, set()).add(target)

        # Match networkx DiGraph.degree behavior (in + out).
        degree_by_id[source] = degree_by_id.get(source, 0) + 1
        degree_by_id[target] = degree_by_id.get(target, 0) + 1

    for node_id in node_ids:
        out_neighbors.setdefault(node_id, set())
        degree_by_id.setdefault(node_id, 0)

    return out_neighbors, degree_by_id, node_ids


def _vector_search(
    *,
    query_vec: Sequence[float],
    graph: SearchableGraphContainer,
    index_name: str,
    top_k: int,
    database_construction_method: str,
    search_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    query_list = [float(x) for x in query_vec]
    results = graph.search(index_name, query_list, k=top_k, **(search_kwargs or {}))

    parsed_results: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if raw_id is None:
            continue

        node_id = str(raw_id)
        content = item.get("document")
        if content is None:
            node = graph.get_node(node_id)
            content = getattr(node, "text", "") if node is not None else ""

        embedding = _to_float_list(item.get("embedding"))
        if embedding is None:
            embedding = _node_embedding(graph.get_node(node_id))

        row: Dict[str, Any] = {
            "id": node_id,
            "content": str(content or ""),
            "metadata": item.get("metadata"),
            "distance": item.get("distance"),
            "embedding": embedding,
        }

        if database_construction_method == "lightrag":
            row["title"] = row["content"].split("\n", 1)[0].strip()
        else:
            row["title"] = node_id

        parsed_results.append(row)

    return parsed_results


def _find_neighbors(
    *,
    target_titles: List[str],
    graph: SearchableGraphContainer,
    retrieved_titles_set: Set[str],
    out_neighbors: Dict[str, Set[str]],
    degree_by_id: Dict[str, int],
) -> List[Dict[str, Any]]:
    neighbors_set: Set[str] = set()
    for title in target_titles:
        neighbors_set |= out_neighbors.get(title, set())

    candidate_titles = [
        node_id for node_id in neighbors_set if node_id in graph.nodes and node_id not in retrieved_titles_set
    ]
    candidate_titles.sort(key=lambda node_id: degree_by_id.get(node_id, 0), reverse=False)

    candidate_nodes: List[Dict[str, Any]] = []
    for node_title in candidate_titles:
        node = graph.get_node(node_title)
        candidate_nodes.append(
            {
                "title": node_title,
                "content": str(getattr(node, "text", "") or ""),
                "embedding": _node_embedding(node),
            }
        )
    return candidate_nodes


class _STeX:
    def calculate_vgs_score(
        self,
        *,
        node: Dict[str, Any],
        retrieved_node_map: Dict[str, Dict[str, Any]],
        retrieved_nodes_title: List[str],
        query_vector: Sequence[float],
        stex_params: Optional[Dict[str, Any]],
        out_neighbors: Dict[str, Set[str]],
        degree_by_id: Dict[str, int],
        graph_nodes_set: Set[str],
    ) -> float:
        beta = 1.0 if stex_params is None else float(stex_params.get("beta", 1.0))

        r_max = len(retrieved_nodes_title)
        node_title = str(node.get("title", ""))
        if node_title not in graph_nodes_set:
            return 0.0

        best_rank = float("inf")
        num_adj_retrieved = 0

        for title in retrieved_nodes_title:
            if title not in graph_nodes_set:
                continue
            if node_title in out_neighbors.get(title, set()):
                neighbor_node_data = retrieved_node_map[title]
                num_adj_retrieved += 1
                best_rank = min(best_rank, float(neighbor_node_data.get("rank", r_max + 1)))

        if num_adj_retrieved == 0:
            return 0.0

        if best_rank == float("inf"):
            best_rank = float(r_max + 1)

        if r_max > 1:
            i_rank = 1.0 - (best_rank - 1.0) / float(r_max - 1)
        else:
            i_rank = 0.0

        node_degree = degree_by_id.get(node_title, 0)
        c_max = min(node_degree, r_max)
        if c_max <= 1:
            i_bridge = 0.0
        else:
            i_bridge = float(num_adj_retrieved - 1) / float(c_max - 1)

        i_sim = 0.0
        embedding = _to_float_list(node.get("embedding"))
        if embedding is not None:
            vec_node = np.array(embedding, dtype=np.float32)
            vec_query = np.array(query_vector, dtype=np.float32)
            if vec_node.shape == vec_query.shape:
                i_sim = float(np.dot(vec_node, vec_query))

        return beta * (i_rank + i_bridge) + i_sim


class _GRanker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", gpu_id: int = 0):
        self.model_name = model_name
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model, "roberta"):
            self.base_model = self.model.roberta
        elif hasattr(self.model, "bert"):
            self.base_model = self.model.bert
        else:
            self.base_model = getattr(self.model, "base_model", self.model)

        self.classifier = self.model.classifier

    def get_cross_encoder_vectors(
        self,
        query: str,
        nodes: List[Dict[str, Any]],
        batch_size: int = 64,
    ) -> torch.Tensor:
        if not nodes:
            return torch.tensor([], device=self.device)

        pairs = [[query, str(node.get("content", ""))] for node in nodes]
        all_cls_vectors: List[torch.Tensor] = []

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
                outputs = self.base_model(**inputs)
                cls_vectors = outputs.last_hidden_state[:, 0, :]
                all_cls_vectors.append(cls_vectors)

        if not all_cls_vectors:
            return torch.tensor([], device=self.device)
        return torch.cat(all_cls_vectors, dim=0)

    def predict_score_from_vectors(self, cls_vectors: torch.Tensor) -> torch.Tensor:
        if cls_vectors.numel() == 0:
            return torch.tensor([], device=self.device)

        with torch.no_grad():
            x = cls_vectors
            x = self.classifier.dropout(x)
            x = self.classifier.dense(x)
            x = torch.tanh(x)
            x = self.classifier.dropout(x)
            x = self.classifier.out_proj(x)
            return x.view(-1)


def _collecting_new(
    *,
    query: str,
    query_vec: Sequence[float],
    nodes: List[Dict[str, Any]],
    graph: SearchableGraphContainer,
    granker: _GRanker,
    gcn_filter: bool = False,
    gcn_alpha: float = 0.7,
    database_construction_method: str = "fastinsight",
    stex_params: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    visualizer: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], None]:
    # Kept for parity with the original implementation.

    slide_size = 10
    max_reranking = len(nodes)
    if max_reranking == 0:
        return [], None

    stex = _STeX()
    node_vector_cache: Dict[str, torch.Tensor] = {}
    out_neighbors, degree_by_id, graph_nodes_set = _build_graph_stats(graph)

    def update_nodes_score_with_fusion(target_nodes: List[Dict[str, Any]]) -> None:
        if not target_nodes:
            return

        nodes_to_encode: List[Dict[str, Any]] = []
        titles_to_encode: List[str] = []
        for node in target_nodes:
            title = _title_from_content(node, database_construction_method)
            if title not in node_vector_cache:
                nodes_to_encode.append(node)
                titles_to_encode.append(title)

        if nodes_to_encode:
            new_vectors = granker.get_cross_encoder_vectors(query, nodes_to_encode)
            for title, vec in zip(titles_to_encode, new_vectors):
                node_vector_cache[title] = vec.view(1, -1)

        current_titles = [_title_from_content(node, database_construction_method) for node in target_nodes]
        feature_matrix = torch.cat([node_vector_cache[title] for title in current_titles], dim=0)

        if gcn_filter and len(target_nodes) > 1:
            title_to_idx = {title: i for i, title in enumerate(current_titles)}
            n_nodes = len(current_titles)
            adj = torch.eye(n_nodes, device=granker.device)

            edges: List[Tuple[int, int]] = []
            for source in current_titles:
                for target in out_neighbors.get(source, set()):
                    if target in title_to_idx:
                        edges.append((title_to_idx[source], title_to_idx[target]))

            if edges:
                edge_idx = torch.tensor(edges, device=granker.device).t()
                adj[edge_idx[0], edge_idx[1]] = 1.0

            degree = adj.sum(dim=1, keepdim=True)
            degree = torch.where(degree == 0, torch.tensor(1.0, device=granker.device), degree)
            norm_adj = adj / degree

            smoothed_features = torch.matmul(norm_adj, feature_matrix)
            feature_matrix = (1.0 - gcn_alpha) * feature_matrix + gcn_alpha * smoothed_features

        scores = granker.predict_score_from_vectors(feature_matrix)
        probs = torch.sigmoid(scores)

        scores_list = scores.detach().cpu().tolist()
        probs_list = probs.detach().cpu().tolist()
        for idx, node in enumerate(target_nodes):
            node["score"] = scores_list[idx]
            node["probability"] = probs_list[idx]

    nodes_contents_processed_index = int(slide_size)
    retrieved_nodes = nodes[:nodes_contents_processed_index]

    update_nodes_score_with_fusion(retrieved_nodes)
    for node in retrieved_nodes:
        node.setdefault("origin", "VS")
    retrieved_nodes.sort(key=_score_of, reverse=True)

    while True:
        if len(retrieved_nodes) >= max_reranking:
            break

        retrieved_node_map = {
            _title_from_content(node, database_construction_method): node for node in retrieved_nodes
        }
        retrieved_nodes_title = list(retrieved_node_map.keys())
        retrieved_titles_set = set(retrieved_nodes_title)

        target_expansion_titles = [title for title in retrieved_nodes_title if title in graph.nodes]
        candidate_nodes = _find_neighbors(
            target_titles=target_expansion_titles,
            graph=graph,
            retrieved_titles_set=retrieved_titles_set,
            out_neighbors=out_neighbors,
            degree_by_id=degree_by_id,
        )

        unique_candidates = {
            _title_from_content(node, database_construction_method): node for node in candidate_nodes
        }
        candidate_nodes = list(unique_candidates.values())

        for node in candidate_nodes:
            score = stex.calculate_vgs_score(
                node=node,
                retrieved_node_map=retrieved_node_map,
                retrieved_nodes_title=retrieved_nodes_title,
                query_vector=query_vec,
                stex_params=stex_params,
                out_neighbors=out_neighbors,
                degree_by_id=degree_by_id,
                graph_nodes_set=graph_nodes_set,
            )
            node["gs_score"] = score

        candidate_nodes.sort(key=lambda node: node.get("gs_score", -float("inf")), reverse=True)
        expanded_nodes = candidate_nodes[:slide_size]

        if len(retrieved_nodes) + len(expanded_nodes) > max_reranking:
            expanded_nodes = expanded_nodes[: max_reranking - len(retrieved_nodes)]
        if not expanded_nodes:
            break

        for node in expanded_nodes:
            node["origin"] = "GS"

        if visualizer is not None and session_id:
            node_id = _title_from_content(node, database_construction_method)
            visualizer.update_session(
                session_id,
                nodes=[
                    {
                        "id": _title_from_content(node, database_construction_method),
                        "style": {
                            "color": {"background": "#c8e6c9", "border": "#4caf50"},
                            "borderWidth": 3,
                        },
                    }
                    for node in expanded_nodes
                ],
                progress={
                    "current": len(retrieved_nodes) + len(expanded_nodes),
                    "total": max_reranking + 10,
                    "message": f"Collecting node: {node_id}",
                },
            )

        retrieved_nodes.extend(expanded_nodes)
        update_nodes_score_with_fusion(retrieved_nodes)
        retrieved_nodes.sort(key=_score_of, reverse=True)

    return retrieved_nodes, None


class FastInsightRetriever(BaseRetriever):
    name = "fastinsight"

    def __init__(self) -> None:
        self._granker: Optional[_GRanker] = None
        self._granker_gpu_id: Optional[int] = None
        self._graph_embedding_populated: Set[int] = set()

    def _ensure_granker(self, *, gpu_id: int, granker: Optional[Any], verbose: bool) -> _GRanker:
        if granker is not None and all(
            hasattr(granker, attr) for attr in ("get_cross_encoder_vectors", "predict_score_from_vectors", "device")
        ):
            self._granker = granker
            self._granker_gpu_id = gpu_id
            return granker

        if isinstance(granker, _GRanker):
            self._granker = granker
            self._granker_gpu_id = gpu_id
            return granker

        if self._granker is not None and self._granker_gpu_id == gpu_id:
            return self._granker

        vprint = print if verbose else (lambda *args, **kwargs: None)
        new_ranker = _GRanker(gpu_id=gpu_id)
        vprint("🔥 GPU Warm-up Triggered...")
        _ = new_ranker.get_cross_encoder_vectors("warmup_query", [{"content": "warmup_doc"}])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vprint("🔥 GPU Warm-up Done.")

        self._granker = new_ranker
        self._granker_gpu_id = gpu_id
        return new_ranker

    def _maybe_populate_graph_embeddings(
        self,
        *,
        graph: SearchableGraphContainer,
        index_name: str,
    ) -> None:
        graph_identity = id(graph)
        if graph_identity in self._graph_embedding_populated:
            return

        indexer = graph.get_index(index_name)
        collection = getattr(indexer, "collection", None)
        if collection is None or not hasattr(collection, "get"):
            self._graph_embedding_populated.add(graph_identity)
            return

        missing_ids = [str(node_id) for node_id, node in graph.nodes.items() if _node_embedding(node) is None]
        if not missing_ids:
            self._graph_embedding_populated.add(graph_identity)
            return

        batch_size = 5000
        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start : start + batch_size]
            try:
                result = collection.get(ids=batch_ids, include=["embeddings"])
            except Exception:
                continue

            ids = _to_list(result.get("ids"))
            embeddings = _to_list(result.get("embeddings"))
            for node_id, emb in zip(ids, embeddings):
                emb_list = _to_float_list(emb)
                if emb_list is None:
                    continue
                node = graph.get_node(str(node_id))
                if node is not None:
                    node.embedding = emb_list

        self._graph_embedding_populated.add(graph_identity)

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
                progress={"current": 0, "total": 100, "message": "Running fastinsight retrieval"},
            )

        seed_top_k = int(kwargs.pop("seed_top_k", 10))
        diving_top_k = int(kwargs.pop("diving_top_k", 100))
        final_top_k = int(kwargs.pop("final_top_k", 10))
        database_construction_method = str(kwargs.pop("database_construction_method", "fastinsight"))
        verbose = bool(kwargs.pop("verbose", False))
        gcn_filter = bool(kwargs.pop("gcn_filter", False))
        gcn_alpha = float(kwargs.pop("gcn_alpha", 0.7))
        gpu_id = int(kwargs.pop("gpu_id", 0))
        stex_params = kwargs.pop("stex_params", None)
        granker = kwargs.pop("granker", None)
        query_vector_override = _to_float_list(kwargs.pop("query_vector", None))
        embedding_provider = str(kwargs.pop("embedding_provider", "hf")).strip().lower()
        embedding_model = str(kwargs.pop("embedding_model", "BAAI/bge-m3")).strip()
        embedding_error_policy = str(kwargs.pop("embedding_error_policy", "raise"))

        if seed_top_k <= 0:
            seed_top_k = 10
        if diving_top_k <= 0:
            diving_top_k = 100
        if final_top_k <= 0:
            final_top_k = 10

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
            if not query_vector:
                raise RuntimeError(
                    "FastInsight retrieval requires a query embedding. "
                    "Provide `query_vector` or configure `embedding_service`."
                )

        search_results = _vector_search(
            query_vec=query_vector,
            graph=graph,
            index_name=index_name,
            top_k=diving_top_k,
            database_construction_method=database_construction_method,
            search_kwargs=kwargs,
        )

        seed_ids = dedup_preserve_order(
            _title_from_content(node, database_construction_method) for node in search_results[:seed_top_k]
        )

        if visualizer is not None and session_id:
            visualizer.update_session(
                session_id,
                nodes=[
                    {
                        "id": seed_id,
                        "style": {
                            "color": {"background": "#bbdefb", "border": "#1565c0"},
                            "borderWidth": 4,
                        },
                    }
                    for seed_id in seed_ids
                ],
                progress={"current": len(seed_ids), "total": 110, "message": f"Seed nodes found: {len(seed_ids)}"},
            )

        ranker = self._ensure_granker(gpu_id=gpu_id, granker=granker, verbose=verbose)
        expanded_nodes, _ = _collecting_new(
            query=query,
            query_vec=query_vector,
            nodes=search_results,
            graph=graph,
            granker=ranker,
            gcn_filter=gcn_filter,
            gcn_alpha=gcn_alpha,
            database_construction_method=database_construction_method,
            stex_params=stex_params,
            session_id=session_id,
            visualizer=visualizer,
        )
        selected_nodes = expanded_nodes[:final_top_k]

        retrieved_nodes: List[RetrievedNode] = []
        context_chunks: List[str] = []

        for raw_node in selected_nodes:
            node_id = _title_from_content(raw_node, database_construction_method)
            text = str(raw_node.get("content", "") or "").strip()
            if not text:
                graph_node = graph.get_node(node_id)
                text = str(getattr(graph_node, "text", "") or "").strip()

            score = raw_node.get("score")
            score_value: Optional[float] = None
            if isinstance(score, (int, float)):
                score_value = float(score)

            metadata: Dict[str, Any] = {}
            if isinstance(raw_node.get("metadata"), dict):
                metadata.update(raw_node["metadata"])
            if raw_node.get("origin") is not None:
                metadata["origin"] = raw_node.get("origin")
            if isinstance(raw_node.get("probability"), (int, float)):
                metadata["probability"] = float(raw_node["probability"])
            if isinstance(raw_node.get("distance"), (int, float)):
                metadata["distance"] = float(raw_node["distance"])
            if isinstance(raw_node.get("gs_score"), (int, float)):
                metadata["gs_score"] = float(raw_node["gs_score"])

            retrieved_nodes.append(
                RetrievedNode(
                    id=node_id,
                    text=text,
                    score=score_value,
                    metadata=metadata,
                )
            )
            if text:
                context_chunks.append(f"[{node_id}] {text}")

        context_chunks = dedup_preserve_order(context_chunks)

        highlight_node_ids = dedup_preserve_order(node.id for node in retrieved_nodes[:10])
        if visualizer is not None and session_id:
            visualizer.update_session(
                session_id,
                nodes=[
                    {
                        "id": node_id,
                        "style": {
                            "color": {"background": "#fff176", "border": "#d84315"},
                            "borderWidth": 7,
                        },
                    }
                    for node_id in highlight_node_ids
                ],
                progress={"current": 110, "total": 110, "message": "FastInsight retrieval complete"},
            )

        origin_stats: Dict[str, int] = {}
        for node in selected_nodes:
            origin = str(node.get("origin", "NULL"))
            origin_stats[origin] = origin_stats.get(origin, 0) + 1

        return RetrievalResult(
            method=self.name,
            query=query,
            seed_nodes=seed_ids,
            nodes=retrieved_nodes,
            edges=[],
            context_chunks=context_chunks,
            metadata={
                "index_name": index_name,
                "top_k": top_k,
                "seed_top_k": seed_top_k,
                "diving_top_k": diving_top_k,
                "final_top_k": final_top_k,
                "gcn_filter": gcn_filter,
                "gcn_alpha": gcn_alpha,
                "seed_count": len(seed_ids),
                "node_count": len(retrieved_nodes),
                "origin_stats": origin_stats,
                "note": "FastInsight runs with default seed_top_k=10, diving_top_k=100, final_top_k=10 unless overridden.",
            },
        )


def fastinsight_search(
    graph: SearchableGraphContainer,
    query: str,
    query_vector: Sequence[float],
    *,
    index_name: str,
    k: int = 5,
) -> Dict[str, Any]:
    retriever = FastInsightRetriever()
    result = retriever.retrieve(
        graph=graph,
        query=query,
        query_vector=query_vector,
        index_name=index_name,
        top_k=k,
    )
    return {
        "index_name": index_name,
        "seed_nodes": result.seed_nodes,
        "nodes": [node.to_dict() for node in result.nodes],
        "context_chunks": result.context_chunks,
        "metadata": result.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    from ...adapters.fastinsight import import_graph_from_fastinsight
    from FlagEmbedding import BGEM3FlagModel

    graph = import_graph_from_fastinsight(args.source)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    query_vector = model.encode(args.query, return_dense=True)["dense_vecs"].tolist()

    result = fastinsight_search(
        graph,
        args.query,
        query_vector,
        index_name="node_vector",
        k=args.k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
