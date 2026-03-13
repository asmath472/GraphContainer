from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def dedup_preserve_order(items: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for item in items:
        value = str(item)
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_error_policy(value: Any) -> str:
    policy = str(value or "raise").strip().lower()
    if policy in {"fallback", "soft"}:
        return "fallback"
    return "raise"


def embed_query(
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
        if normalize_error_policy(embedding_error_policy) == "fallback":
            return []
        raise RuntimeError(message) from exc


def keyword_fallback_seed_scores(
    graph: Any,
    *,
    query: str,
    top_k: int,
) -> Tuple[List[str], Dict[str, float]]:
    tokens = [t.lower() for t in str(query).split() if t.strip()]
    if top_k <= 0 or not tokens:
        return [], {}

    scored: List[Tuple[int, str]] = []
    for raw_node_id, node in graph.nodes.items():
        node_id = str(raw_node_id)
        text = str(getattr(node, "text", "") or "")
        haystack = f"{node_id} {text}".lower()
        hits = sum(1 for tok in tokens if tok in haystack)
        if hits > 0:
            scored.append((hits, node_id))

    scored.sort(key=lambda x: -x[0])
    denom = max(len(tokens), 1)
    seed_ids: List[str] = []
    seed_score_by_id: Dict[str, float] = {}
    for hits, node_id in scored[:top_k]:
        seed_ids.append(node_id)
        seed_score_by_id[node_id] = float(hits) / float(denom)
    return seed_ids, seed_score_by_id


def vector_seed_scores(
    graph: Any,
    *,
    index_name: str,
    query_vector: List[float],
    top_k: int,
    search_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Optional[float]], Optional[str]]:
    if top_k <= 0 or not query_vector:
        return [], {}, None

    get_index = getattr(graph, "get_index", None)
    if callable(get_index):
        try:
            if get_index(index_name) is None:
                return [], {}, f"Index '{index_name}' is not attached."
        except Exception:
            # Best-effort precheck only; fallback to direct search call.
            pass

    try:
        rows = graph.search(index_name, list(query_vector), k=top_k, **(search_kwargs or {}))
    except Exception as exc:
        return [], {}, f"{type(exc).__name__}: {exc}"

    seed_ids: List[str] = []
    seed_score_by_id: Dict[str, Optional[float]] = {}
    for item in rows:
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

    return seed_ids, seed_score_by_id, None
