from __future__ import annotations

import csv
import json
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError


_TRIPLE_RE = re.compile(r"\((.*?)\)")


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return path / "train_dev.tsv"
        return path
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _iter_tsv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise UnsupportedSourceError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row:
                yield row


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _parse_triplets(graph_text: str) -> Iterator[Tuple[str, str, str]]:
    for chunk in _TRIPLE_RE.findall(graph_text or ""):
        parts = [p.strip() for p in chunk.split(";", maxsplit=2)]
        if len(parts) != 3:
            continue
        src, rel, dst = parts
        if not src or not dst:
            continue
        yield src, rel, dst


class ExplaGraphsAdapter(GraphAdapter):
    """Adapter for Expla_graphs train_dev.tsv format."""

    def __init__(self):
        super().__init__(name="expla_graphs", version="0.1.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            return src.is_file() and src.suffix.lower() in {".tsv", ".txt"}
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

        graph = container_or_new(container)

        # Deduplicate nodes/edges across rows by default.
        node_key_to_id: Dict[Tuple[str, str], str] = {}
        edge_key_to_idx: Dict[Tuple[str, str, str], int] = {}

        for row_idx, row in enumerate(_iter_tsv_rows(src_path)):
            graph_text = row.get("graph")
            if not graph_text:
                continue

            for src, rel, dst in _parse_triplets(graph_text):
                src_text = _normalize_text(src)
                dst_text = _normalize_text(dst)
                rel_text = _normalize_text(rel) or "related"

                src_key = ("entity", src_text)
                dst_key = ("entity", dst_text)

                src_id = node_key_to_id.get(src_key)
                if src_id is None:
                    src_id = f"node-{md5(f'{src_key[0]}|{src_key[1]}'.encode()).hexdigest()}"
                    node_key_to_id[src_key] = src_id
                    metadata: Dict[str, Any] = {}
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(src_path))
                        metadata.setdefault("_source_style", "expla_graphs")
                    graph.add_node(
                        NodeRecord(
                            id=src_id,
                            type="Entity",
                            text=src,
                            metadata=metadata,
                        )
                    )

                dst_id = node_key_to_id.get(dst_key)
                if dst_id is None:
                    dst_id = f"node-{md5(f'{dst_key[0]}|{dst_key[1]}'.encode()).hexdigest()}"
                    node_key_to_id[dst_key] = dst_id
                    metadata = {}
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(src_path))
                        metadata.setdefault("_source_style", "expla_graphs")
                    graph.add_node(
                        NodeRecord(
                            id=dst_id,
                            type="Entity",
                            text=dst,
                            metadata=metadata,
                        )
                    )

                edge_key = (src_id, rel_text, dst_id)
                if edge_key in edge_key_to_idx:
                    continue

                metadata = {}
                if keep_source_reference:
                    metadata.setdefault("_source_path", str(src_path))
                    metadata.setdefault("_source_style", "expla_graphs")

                graph.add_edge(
                    EdgeRecord(
                        source=src_id,
                        target=dst_id,
                        relation=rel_text,
                        weight=1.0,
                        metadata=metadata,
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

        return {
            "destination": str(dest),
            "node_file": str(node_path),
            "edge_file": str(edge_path),
            "nodes": len(container.nodes),
            "edges": len(container.edges),
        }


def import_graph_from_expla_graphs(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
    **kwargs: Any,
) -> SearchableGraphContainer:
    return ExplaGraphsAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        **kwargs,
    )
