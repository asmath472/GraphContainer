from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError
from ..indexers import ChromaCollectionIndexer


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        raise UnsupportedSourceError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                # edges.jsonl may use list rows.
                yield {"_list": item}


def _parse_edge(raw: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if "_list" in raw:
        row = raw["_list"]
        if isinstance(row, list) and len(row) >= 2:
            return str(row[0]), str(row[1])
        return None

    source = raw.get("src", raw.get("source"))
    target = raw.get("tgt", raw.get("target"))
    if source is None or target is None:
        return None
    return str(source), str(target)


class FastInsightAdapter(GraphAdapter):
    def __init__(self):
        super().__init__(name="component_graph", version="0.1.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            return (src / "nodes.jsonl").exists() and (src / "edges.jsonl").exists()
        except Exception:
            return False

    def _get_manifest(self, path: Path) -> Dict[str, Any]:
        for name in ("manifest.json", "menifest.json"):
            p = path / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def import_graph(
        self,
        source: Any,
        container: Optional[SimpleGraphContainer] = None,
        *,
        keep_source_reference: bool = False,
        **kwargs: Any,
    ) -> SearchableGraphContainer:
        src_path = _normalize_source(source)
        manifest = self._get_manifest(src_path)
        v_cfg = manifest.get("vector_store", {})

        # Priority: kwargs > manifest > default
        col_name = kwargs.get("collection_name") or v_cfg.get("collection_name")
        db_path = kwargs.get("vector_store_path") or v_cfg.get("path")
        distance_metric = str(kwargs.get("distance_metric", v_cfg.get("distance_metric", "cosine")))

        graph = container_or_new(container)

        for raw in _iter_jsonl(src_path / "nodes.jsonl"):
            node_id = raw.get("id")
            if node_id is None:
                continue
            metadata = {k: v for k, v in raw.items() if k not in {"id", "text", "type"}}
            if keep_source_reference:
                metadata.setdefault("_source_path", str(src_path))
                metadata.setdefault("_source_style", "component_graph")
            graph.add_node(
                NodeRecord(
                    id=str(node_id),
                    type=str(raw.get("type", "Chunk")),
                    text=raw.get("text"),
                    metadata=metadata,
                )
            )

        for raw in _iter_jsonl(src_path / "edges.jsonl"):
            endpoints = _parse_edge(raw)
            if endpoints is None:
                continue
            source_id, target_id = endpoints
            graph.add_edge(EdgeRecord(source=source_id, target=target_id))

        if col_name:
            import chromadb

            if not db_path:
                db_path = f"./data/database/chroma_db/{col_name}"
            client = chromadb.PersistentClient(path=str(Path(str(db_path)).resolve()))
            col = client.get_or_create_collection(name=str(col_name))
            indexer = ChromaCollectionIndexer(
                col,
                persist_path=str(db_path),
                distance_metric=distance_metric,
            )
            graph.attach_index(str(col_name), indexer)
            graph.attach_index("node_vector", indexer)

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
            for n in container.nodes.values():
                payload = {"id": n.id, "text": n.text, "type": n.type, **n.metadata}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with edge_path.open("w", encoding="utf-8") as f:
            for e in container.edges:
                f.write(json.dumps([e.source, e.target], ensure_ascii=False) + "\n")

        store_info = None
        if isinstance(container, SearchableGraphContainer):
            idx = container.get_index("node_vector")
            if idx is not None and hasattr(idx, "describe_store"):
                store_info = idx.describe_store()

        col_name = kwargs.get("collection_name", dest.name)
        vector_path = kwargs.get("vector_store_path", f"./chroma_db/{col_name}")
        manifest = {
            "vector_store": store_info
            or {
                "library": "chromadb",
                "path": vector_path,
                "collection_name": col_name,
                "distance_metric": kwargs.get("distance_metric", "cosine"),
            }
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "destination": str(dest),
            "node_file": str(node_path),
            "edge_file": str(edge_path),
            "nodes": len(container.nodes),
            "edges": len(container.edges),
        }


def import_graph_from_fastinsight(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
    **kwargs: Any,
) -> SearchableGraphContainer:
    return FastInsightAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        **kwargs,
    )
