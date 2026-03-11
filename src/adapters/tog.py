from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _append_unique(metadata: Dict[str, Any], key: str, value: Any) -> None:
    current = metadata.get(key)
    if current is None:
        metadata[key] = [value]
        return
    if not isinstance(current, list):
        current = [current]
    if value not in current:
        current.append(value)
    metadata[key] = current


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise UnsupportedSourceError(f"ToG result file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    records.append(row)
        return records

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    raise UnsupportedSourceError(
        f"Unsupported ToG JSON payload type: {type(payload)} (expected object or list of objects)"
    )


def _entity_id(item: Dict[str, Any], *, fallback_prefix: str) -> Optional[str]:
    for key in ("id", "entity_id", "qid"):
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    name = item.get("name", item.get("entity_name"))
    if name is not None and str(name).strip() != "":
        return f"{fallback_prefix}:{str(name).strip()}"
    return None


def _entity_name(item: Dict[str, Any]) -> Optional[str]:
    for key in ("name", "entity_name", "label"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return None


class ToGAdapter(GraphAdapter):
    """Adapter for ToG-2 explored-subgraph result JSON files (Option B)."""

    def __init__(self):
        super().__init__(name="tog", version="0.1.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            return src.exists() and src.suffix.lower() in {".json", ".jsonl"}
        except Exception:
            return False

    def import_graph(
        self,
        source: Any,
        container: Optional[SimpleGraphContainer] = None,
        *,
        keep_source_reference: bool = False,
        **kwargs: Any,
    ) -> SearchableGraphContainer:
        src_path = _normalize_source(source)
        records = _load_records(src_path)

        graph = container_or_new(container)
        edge_key_to_idx: Dict[Tuple[str, str, str], int] = {}

        for record_idx, record in enumerate(records):
            search_entity_list = record.get("search_entity_list", [])
            if not isinstance(search_entity_list, list):
                continue

            question = record.get("question")

            for depth_item in search_entity_list:
                if not isinstance(depth_item, dict):
                    continue
                depth = depth_item.get("depth")
                relation_items = depth_item.get("each_relation_right_entityList", [])
                if not isinstance(relation_items, list):
                    continue

                for rel_item in relation_items:
                    if not isinstance(rel_item, dict):
                        continue
                    current_relation = rel_item.get("current_relation", {})
                    if not isinstance(current_relation, dict):
                        continue

                    curr_id = _entity_id(current_relation, fallback_prefix="name")
                    curr_name = _entity_name(current_relation)
                    if curr_id is None:
                        continue

                    curr_metadata: Dict[str, Any] = {}
                    if keep_source_reference:
                        curr_metadata["_source_path"] = str(src_path)
                        curr_metadata["_source_style"] = "tog"
                    if question is not None:
                        curr_metadata["question"] = question

                    if curr_id not in graph.nodes:
                        graph.add_node(
                            NodeRecord(
                                id=curr_id,
                                type="Entity",
                                text=curr_name,
                                metadata=curr_metadata,
                            )
                        )
                    else:
                        node = graph.nodes[curr_id]
                        if node.text is None and curr_name is not None:
                            node.text = curr_name

                    relation_text = str(current_relation.get("relation", "related"))
                    is_head = bool(current_relation.get("head", True))

                    right_entities = rel_item.get("right_entity", [])
                    if not isinstance(right_entities, list):
                        continue

                    for right in right_entities:
                        if not isinstance(right, dict):
                            continue
                        right_id = _entity_id(right, fallback_prefix="name")
                        right_name = _entity_name(right)
                        if right_id is None:
                            continue

                        right_metadata: Dict[str, Any] = {}
                        if keep_source_reference:
                            right_metadata["_source_path"] = str(src_path)
                            right_metadata["_source_style"] = "tog"

                        related_paragraphs = right.get("related_paragraphs")
                        if related_paragraphs is not None:
                            right_metadata["related_paragraphs"] = related_paragraphs

                        if right_id not in graph.nodes:
                            graph.add_node(
                                NodeRecord(
                                    id=right_id,
                                    type="Entity",
                                    text=right_name,
                                    metadata=right_metadata,
                                )
                            )
                        else:
                            right_node = graph.nodes[right_id]
                            if right_node.text is None and right_name is not None:
                                right_node.text = right_name
                            if related_paragraphs is not None:
                                _append_unique(
                                    right_node.metadata,
                                    "related_paragraphs",
                                    related_paragraphs,
                                )

                        source_id, target_id = (curr_id, right_id) if is_head else (right_id, curr_id)
                        edge_key = (source_id, relation_text, target_id)
                        if edge_key in edge_key_to_idx:
                            edge_idx = edge_key_to_idx[edge_key]
                            edge = graph.edges[edge_idx]
                            if depth is not None:
                                _append_unique(edge.metadata, "depths", depth)
                            _append_unique(edge.metadata, "record_indices", record_idx)
                            continue

                        edge_metadata: Dict[str, Any] = {"record_indices": [record_idx]}
                        if depth is not None:
                            edge_metadata["depths"] = [depth]
                        if keep_source_reference:
                            edge_metadata["_source_path"] = str(src_path)
                            edge_metadata["_source_style"] = "tog"

                        graph.add_edge(
                            EdgeRecord(
                                source=source_id,
                                target=target_id,
                                relation=relation_text,
                                weight=1.0,
                                metadata=edge_metadata,
                            )
                        )
                        edge_key_to_idx[edge_key] = len(graph.edges) - 1

        return graph

    def export_graph(
        self,
        container: SimpleGraphContainer,
        destination: Any,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        dest = Path(destination)
        if dest.exists() and not overwrite and any(dest.iterdir()):
            raise GraphAdapterError(f"Destination is not empty: {dest}")
        dest.mkdir(parents=True, exist_ok=True)

        node_path = dest / "nodes.jsonl"
        edge_path = dest / "edges.jsonl"

        with node_path.open("w", encoding="utf-8") as f:
            for node in container.nodes.values():
                payload = {
                    "id": node.id,
                    "text": node.text,
                    "type": node.type,
                    **node.metadata,
                }
                if node.embedding is not None:
                    payload["embedding"] = (
                        node.embedding.tolist()
                        if hasattr(node.embedding, "tolist")
                        else list(node.embedding)
                    )
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with edge_path.open("w", encoding="utf-8") as f:
            for edge in container.edges:
                payload = {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    **edge.metadata,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        store_info = None
        if isinstance(container, SearchableGraphContainer):
            idx = container.get_index("node_vector")
            if idx is not None and hasattr(idx, "describe_store"):
                store_info = idx.describe_store()
        if store_info is not None:
            (dest / "manifest.json").write_text(
                json.dumps({"vector_store": store_info}, indent=2), encoding="utf-8"
            )

        return {
            "destination": str(dest),
            "node_file": str(node_path),
            "edge_file": str(edge_path),
            "nodes": len(container.nodes),
            "edges": len(container.edges),
        }


def import_graph_from_tog(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
    **kwargs: Any,
) -> SearchableGraphContainer:
    return ToGAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        **kwargs,
    )
