from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError


def _normalize_source(source: Any) -> Tuple[str, Optional[Path]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            return "path", path
        return "hf", str(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _iter_hf_graphs(dataset_name: str, splits: Optional[Iterable[str]]) -> Iterator[Iterable[Tuple[str, str, str]]]:
    from datasets import load_dataset, concatenate_datasets

    dataset = load_dataset(dataset_name)
    if splits is None:
        split_names = [k for k in ("train", "validation", "test") if k in dataset]
        if not split_names:
            split_names = list(dataset.keys())
    else:
        split_names = list(splits)
    parts = [dataset[name] for name in split_names if name in dataset]
    if not parts:
        raise UnsupportedSourceError(
            f"No requested splits found in dataset '{dataset_name}'. Available: {list(dataset.keys())}"
        )
    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    for row in merged:
        graph = row.get("graph") if isinstance(row, dict) else None
        if graph:
            yield graph


class FreebaseKGAdapter(GraphAdapter):
    """Adapter for the RoG-webqsp dataset on Hugging Face (Freebase KG triples)."""

    def __init__(self):
        super().__init__(name="freebasekg", version="0.1.0")

    def can_import(self, source: Any) -> bool:
        try:
            # FreebaseKG is special; it primarily expects a HF dataset name (str)
            # or a path that we intentionally reject for this adapter.
            mode, _ = _normalize_source(source)
            return mode == "hf"
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
        mode, resolved = _normalize_source(source)
        splits = kwargs.get("splits")
        dataset_name = "rmanluo/RoG-webqsp"

        if mode == "path":
            raise UnsupportedSourceError(
                "WebQSP adapter expects a Hugging Face dataset name (e.g. "
                "'rmanluo/RoG-webqsp'). Use the g_retriever adapter for preprocessed "
                "nodes/edges CSVs."
            )
        if isinstance(resolved, str):
            dataset_name = resolved

        graph = container_or_new(container)

        node_key_to_id: Dict[str, str] = {}
        edge_key_to_idx: Dict[Tuple[str, str, str], int] = {}

        for triples in _iter_hf_graphs(dataset_name, splits):
            for tri in triples:
                if not isinstance(tri, (list, tuple)) or len(tri) != 3:
                    continue
                head, rel, tail = tri
                head_norm = _normalize_text(head)
                tail_norm = _normalize_text(tail)
                rel_norm = _normalize_text(rel) or "related"
                if not head_norm or not tail_norm:
                    continue

                head_id = node_key_to_id.get(head_norm)
                if head_id is None:
                    head_id = f"node-{md5(head_norm.encode()).hexdigest()}"
                    node_key_to_id[head_norm] = head_id
                    metadata: Dict[str, Any] = {}
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(source))
                        metadata.setdefault("_source_style", "freebasekg")
                    graph.add_node(
                        NodeRecord(
                            id=head_id,
                            type="Entity",
                            text=head,
                            metadata=metadata,
                        )
                    )

                tail_id = node_key_to_id.get(tail_norm)
                if tail_id is None:
                    tail_id = f"node-{md5(tail_norm.encode()).hexdigest()}"
                    node_key_to_id[tail_norm] = tail_id
                    metadata = {}
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(source))
                        metadata.setdefault("_source_style", "freebasekg")
                    graph.add_node(
                        NodeRecord(
                            id=tail_id,
                            type="Entity",
                            text=tail,
                            metadata=metadata,
                        )
                    )

                edge_key = (head_id, rel_norm, tail_id)
                if edge_key in edge_key_to_idx:
                    continue
                metadata = {}
                if keep_source_reference:
                    metadata.setdefault("_source_path", str(source))
                    metadata.setdefault("_source_style", "freebasekg")
                graph.add_edge(
                    EdgeRecord(
                        source=head_id,
                        target=tail_id,
                        relation=rel_norm,
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


def import_graph_from_freebasekg(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
    **kwargs: Any,
) -> SearchableGraphContainer:
    return FreebaseKGAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        **kwargs,
    )
