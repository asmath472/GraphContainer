from __future__ import annotations

import csv
import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise UnsupportedSourceError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                yield row


def _resolve_csv_paths(source_path: Path, *, graph_id: Optional[str]) -> Tuple[Path, Path]:
    direct_nodes = source_path / "nodes.csv"
    direct_edges = source_path / "edges.csv"
    if direct_nodes.exists() and direct_edges.exists():
        return direct_nodes, direct_edges

    nodes_dir = source_path / "nodes"
    edges_dir = source_path / "edges"
    if not (nodes_dir.is_dir() and edges_dir.is_dir()):
        raise UnsupportedSourceError(
            "Expected either (nodes.csv, edges.csv) or (nodes/<id>.csv, edges/<id>.csv)."
        )

    if graph_id is not None:
        node_csv = nodes_dir / f"{graph_id}.csv"
        edge_csv = edges_dir / f"{graph_id}.csv"
        if node_csv.exists() and edge_csv.exists():
            return node_csv, edge_csv
        raise UnsupportedSourceError(
            f"Could not find nodes/edges CSV files for graph_id='{graph_id}' under: {source_path}"
        )

    node_ids = {p.stem for p in nodes_dir.glob("*.csv")}
    edge_ids = {p.stem for p in edges_dir.glob("*.csv")}
    common_ids = sorted(node_ids & edge_ids)

    if len(common_ids) == 1:
        single_id = common_ids[0]
        return nodes_dir / f"{single_id}.csv", edges_dir / f"{single_id}.csv"

    if not common_ids:
        raise UnsupportedSourceError(
            f"No matching nodes/edges CSV pairs found under: {source_path}"
        )

    raise UnsupportedSourceError(
        f"Multiple graph CSV pairs found under {source_path}; pass graph_id to select one. "
        f"Found ids: {common_ids[:10]}{'...' if len(common_ids) > 10 else ''}"
    )


def _list_graph_ids(source_path: Path) -> list[str]:
    nodes_dir = source_path / "nodes"
    edges_dir = source_path / "edges"
    if not (nodes_dir.is_dir() and edges_dir.is_dir()):
        raise UnsupportedSourceError(
            f"Expected directories '{nodes_dir}' and '{edges_dir}' for graph_id='all'."
        )

    node_ids = {p.stem for p in nodes_dir.glob("*.csv")}
    edge_ids = {p.stem for p in edges_dir.glob("*.csv")}
    common_ids = sorted(node_ids & edge_ids)
    if not common_ids:
        raise UnsupportedSourceError(
            f"No matching nodes/edges CSV pairs found under: {source_path}"
        )
    return common_ids


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _append_unique(metadata: Dict[str, Any], key: str, value: str) -> None:
    current = metadata.get(key)
    if current is None:
        metadata[key] = [value]
        return
    if not isinstance(current, list):
        current = [str(current)]
    if value not in current:
        current.append(value)
    metadata[key] = current


class GRetrieverAdapter(GraphAdapter):
    def __init__(self):
        super().__init__(name="g_retriever", version="0.1.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            if (src / "nodes.csv").exists() and (src / "edges.csv").exists():
                return True
            return (src / "nodes").is_dir() and (src / "edges").is_dir()
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
        graph_id = kwargs.get("graph_id")
        graph_id_str = str(graph_id) if graph_id is not None else None
        graph = container_or_new(container)

        if graph_id_str == "all":
            graph_ids = _list_graph_ids(src_path)
            node_key_to_global: Dict[Tuple[str, str], str] = {}
            edge_key_to_idx: Dict[Tuple[str, str, str], int] = {}

            for gid in graph_ids:
                node_csv, edge_csv = _resolve_csv_paths(src_path, graph_id=gid)
                local_to_global: Dict[str, str] = {}

                for raw in _iter_csv_rows(node_csv):
                    node_id = raw.get("node_id") or raw.get("id")
                    if node_id is None or str(node_id).strip() == "":
                        continue

                    node_text = raw.get("node_attr", raw.get("text"))
                    node_type = str(raw.get("type", "Entity"))
                    local_id = str(node_id)
                    node_key_text = _normalize_text(node_text)
                    if node_key_text:
                        node_key = (node_type.lower(), node_key_text)
                        default_global_id = f"node-{md5(f'{node_key[0]}|{node_key[1]}'.encode()).hexdigest()}"
                    else:
                        node_key = ("__fallback__", f"{gid}:{local_id}")
                        default_global_id = f"{gid}:{local_id}"
                    global_id = node_key_to_global.get(node_key, default_global_id)
                    local_to_global[local_id] = global_id
                    if global_id in graph.nodes:
                        existing_meta = graph.nodes[global_id].metadata
                        _append_unique(existing_meta, "graph_ids", str(gid))
                        _append_unique(existing_meta, "original_node_ids", f"{gid}:{local_id}")
                        continue

                    metadata = {
                        k: v
                        for k, v in raw.items()
                        if k not in {"node_id", "id", "node_attr", "text", "type"}
                    }
                    metadata["graph_ids"] = [str(gid)]
                    metadata["original_node_ids"] = [f"{gid}:{local_id}"]
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(src_path))
                        metadata.setdefault("_source_style", "g_retriever")

                    graph.add_node(
                        NodeRecord(
                            id=global_id,
                            type=node_type,
                            text=node_text,
                            metadata=metadata,
                        )
                    )
                    node_key_to_global[node_key] = global_id

                for raw in _iter_csv_rows(edge_csv):
                    source_id = raw.get("src", raw.get("source"))
                    target_id = raw.get("dst", raw.get("target"))
                    if source_id is None or target_id is None:
                        continue

                    src_local = str(source_id)
                    tgt_local = str(target_id)
                    source_global = local_to_global.get(src_local, f"{gid}:{src_local}")
                    target_global = local_to_global.get(tgt_local, f"{gid}:{tgt_local}")

                    relation = str(raw.get("edge_attr", raw.get("relation", "related")))
                    try:
                        weight = float(raw.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0

                    edge_key = (source_global, relation, target_global)
                    if edge_key in edge_key_to_idx:
                        edge_idx = edge_key_to_idx[edge_key]
                        existing_edge = graph.edges[edge_idx]
                        if existing_edge.weight < weight:
                            existing_edge.weight = weight
                        _append_unique(existing_edge.metadata, "graph_ids", str(gid))
                        _append_unique(
                            existing_edge.metadata,
                            "original_edge_ids",
                            f"{gid}:{src_local}->{tgt_local}",
                        )
                        continue

                    metadata = {
                        k: v
                        for k, v in raw.items()
                        if k
                        not in {
                            "src",
                            "source",
                            "dst",
                            "target",
                            "edge_attr",
                            "relation",
                            "weight",
                        }
                    }
                    metadata["graph_ids"] = [str(gid)]
                    metadata["original_edge_ids"] = [f"{gid}:{src_local}->{tgt_local}"]
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(src_path))
                        metadata.setdefault("_source_style", "g_retriever")

                    graph.add_edge(
                        EdgeRecord(
                            source=source_global,
                            target=target_global,
                            relation=relation,
                            weight=weight,
                            metadata=metadata,
                        )
                    )
                    edge_key_to_idx[edge_key] = len(graph.edges) - 1
        else:
            node_csv, edge_csv = _resolve_csv_paths(src_path, graph_id=graph_id_str)

            for raw in _iter_csv_rows(node_csv):
                node_id = raw.get("node_id") or raw.get("id")
                if node_id is None or str(node_id).strip() == "":
                    continue

                node_text = raw.get("node_attr", raw.get("text"))
                node_type = str(raw.get("type", "Entity"))
                metadata = {
                    k: v
                    for k, v in raw.items()
                    if k not in {"node_id", "id", "node_attr", "text", "type"}
                }
                if graph_id_str is not None:
                    metadata.setdefault("graph_id", graph_id_str)
                if keep_source_reference:
                    metadata.setdefault("_source_path", str(src_path))
                    metadata.setdefault("_source_style", "g_retriever")

                graph.add_node(
                    NodeRecord(
                        id=str(node_id),
                        type=node_type,
                        text=node_text,
                        metadata=metadata,
                    )
                )

            for raw in _iter_csv_rows(edge_csv):
                source_id = raw.get("src", raw.get("source"))
                target_id = raw.get("dst", raw.get("target"))
                if source_id is None or target_id is None:
                    continue

                relation = str(raw.get("edge_attr", raw.get("relation", "related")))
                try:
                    weight = float(raw.get("weight", 1.0))
                except (TypeError, ValueError):
                    weight = 1.0

                metadata = {
                    k: v
                    for k, v in raw.items()
                    if k not in {"src", "source", "dst", "target", "edge_attr", "relation", "weight"}
                }
                if graph_id_str is not None:
                    metadata.setdefault("graph_id", graph_id_str)
                if keep_source_reference:
                    metadata.setdefault("_source_path", str(src_path))
                    metadata.setdefault("_source_style", "g_retriever")

                graph.add_edge(
                    EdgeRecord(
                        source=str(source_id),
                        target=str(target_id),
                        relation=relation,
                        weight=weight,
                        metadata=metadata,
                    )
                )

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


def import_graph_from_g_retriever(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
    **kwargs: Any,
) -> SearchableGraphContainer:
    return GRetrieverAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        **kwargs,
    )
