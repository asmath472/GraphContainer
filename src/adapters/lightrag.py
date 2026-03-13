# src/GraphContainer/adapters/lightrag.py
from __future__ import annotations

import base64
import json
import os
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import dotenv
from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError
from ..indexers import ChromaCollectionIndexer, to_float_list

dotenv.load_dotenv()  # Load environment variables from .env file, if present


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _decode_vector(encoded_str: str) -> Optional[List[float]]:
    try:
        import numpy as np
    except ImportError:
        return None

    try:
        decoded = base64.b64decode(encoded_str)
        decompressed = zlib.decompress(decoded)
        vec_f16 = np.frombuffer(decompressed, dtype=np.float16)
        return vec_f16.astype(np.float32).tolist()
    except Exception:
        return None


def _vector_from_item(item: Dict[str, Any], matrix_row: Any = None) -> Optional[List[float]]:
    for key in ("__vector__", "embedding", "vector"):
        v = item.get(key)
        vec = to_float_list(v)
        if vec is not None:
            return vec

    if isinstance(item.get("vector"), str):
        return _decode_vector(item["vector"])

    vec = to_float_list(matrix_row)
    if vec is not None:
        return vec
    return None


def _iter_vdb_items(path: Path, *, load_embeddings: bool) -> Iterable[Tuple[Dict[str, Any], Any]]:
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None

    if ijson is None:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("data", [])
        matrix = payload.get("matrix", []) if load_embeddings else []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            row = matrix[i] if i < len(matrix) else None
            yield item, row
        return

    with path.open("rb") as f:
        for item in ijson.items(f, "data.item"):
            if isinstance(item, dict):
                # Streaming path only reads "data"; embedding is taken from item fields.
                yield item, None


def _flush_upsert(
    col: Any,
    ids: List[str],
    embs: List[List[float]],
    docs: List[str],
    metas: List[Dict[str, Any]],
) -> None:
    if ids:
        col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        ids.clear()
        embs.clear()
        docs.clear()
        metas.clear()


class LightRAGAdapter(GraphAdapter):
    """Adapter for LightRAG-style NanoVectorDB graph files."""

    def __init__(self, *, version: str = "0.1.0"):
        super().__init__(name="lightrag", version=version)

    def can_import(self, source: Any) -> bool:
        try:
            source_path = _normalize_source(source)
            return (source_path / "vdb_entities.json").exists() and (source_path / "vdb_relationships.json").exists()
        except Exception:
            return False

    def import_graph(
        self,
        source: Any,
        container: Optional[SimpleGraphContainer] = None,
        *,
        keep_source_reference: bool = False,
    ) -> SearchableGraphContainer:
        source_path = _normalize_source(source)

        node_file = source_path / "vdb_entities.json"
        edge_file = source_path / "vdb_relationships.json"
        if not node_file.exists() or not edge_file.exists():
            raise UnsupportedSourceError(f"LightRAG files not found under: {source_path}")

        attach_index = _env_bool("LIGHTRAG_ATTACH_INDEX", True)
        load_embeddings = _env_bool("LIGHTRAG_LOAD_EMBEDDINGS", True)
        batch_size = int(os.getenv("LIGHTRAG_BATCH_SIZE", "1000"))

        graph = container_or_new(container)
        graph.nodes = {}
        graph.edges = []
        graph._adj = {}

        # ── Fast-path: when kv_store files are available, use them unconditionally
        # for visualization (8 MB + 12 MB vs 700 MB+ vdb JSON files).
        # Set LIGHTRAG_FORCE_VDB=1 to bypass this and load full vdb files with embeddings.
        force_vdb = _env_bool("LIGHTRAG_FORCE_VDB", False)
        kv_ent_file = source_path / "kv_store_full_entities.json"
        kv_rel_file = source_path / "kv_store_full_relations.json"
        if not force_vdb and kv_ent_file.exists() and kv_rel_file.exists():
            with kv_ent_file.open("r", encoding="utf-8") as f:
                kv_entities: Dict[str, Any] = json.load(f)
            with kv_rel_file.open("r", encoding="utf-8") as f:
                kv_relations: Dict[str, Any] = json.load(f)

            # Collect unique entity names → one node each
            seen_nodes: set = set()
            for doc_entry in kv_entities.values():
                for entity_name in doc_entry.get("entity_names", []):
                    if not entity_name or entity_name in seen_nodes:
                        continue
                    seen_nodes.add(entity_name)
                    graph.add_node(
                        NodeRecord(
                            id=str(entity_name),
                            type="Entity",
                            text=None,
                            embedding=None,
                            metadata={},
                        )
                    )

            # Collect unique relation pairs → one edge each
            seen_edges: set = set()
            for doc_entry in kv_relations.values():
                for pair in doc_entry.get("relation_pairs", []):
                    if len(pair) != 2:
                        continue
                    src, tgt = str(pair[0]), str(pair[1])
                    key = (src, tgt)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    graph.add_edge(
                        EdgeRecord(
                            source=src,
                            target=tgt,
                            relation="RELATED",
                            weight=1.0,
                            metadata={},
                        )
                    )

            if keep_source_reference:
                for node in graph.nodes.values():
                    node.metadata.setdefault("_source_path", str(source_path))
                    node.metadata.setdefault("_source_style", "lightrag")

            return graph  # type: ignore[return-value]

        # ── Full-path: load vdb files (slow for large graphs, includes embeddings)
        # Use LIGHTRAG_FORCE_VDB=1 to reach this path, or when kv_store files are absent.
        node_col = None
        edge_col = None
        db_path = os.getenv("VECTOR_STORE_PATH", "./data/database/chroma_db")
        env_col_name = source_path.name
        distance_metric = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")
        if attach_index and load_embeddings:
            import chromadb

            base_name = str(env_col_name or source_path.name)
            node_col_name = f"{base_name}_entities"
            edge_col_name = f"{base_name}_relationships"
            db_path = str(db_path or f"./chroma_db/{base_name}")
            db_dir = Path(db_path).resolve()
            db_dir.mkdir(parents=True, exist_ok=True)
            try:
                client = chromadb.PersistentClient(path=str(db_dir))
            except Exception:
                client = chromadb.EphemeralClient()
                db_path = None
            node_col = client.get_or_create_collection(name=node_col_name)
            edge_col = client.get_or_create_collection(name=edge_col_name)

        node_ids: List[str] = []
        node_embs: List[List[float]] = []
        node_docs: List[str] = []
        node_metas: List[Dict[str, Any]] = []
        edge_ids: List[str] = []
        edge_embs: List[List[float]] = []
        edge_docs: List[str] = []
        edge_metas: List[Dict[str, Any]] = []

        for raw, matrix_row in _iter_vdb_items(node_file, load_embeddings=load_embeddings):
            node_id = raw.get("entity_name") or raw.get("id") or raw.get("__id__")
            if node_id is None:
                continue
            node_id = str(node_id)

            embedding = _vector_from_item(raw, matrix_row) if load_embeddings else None
            metadata = {
                k: v
                for k, v in raw.items()
                if k not in {"entity_name", "id", "__id__", "content", "vector", "__vector__", "embedding"}
            }
            if "__id__" in raw:
                metadata.setdefault("lightrag_id", raw["__id__"])

            node = NodeRecord(
                id=node_id,
                type=str(raw.get("type", "Entity")),
                text=raw.get("content"),
                embedding=embedding,
                metadata=metadata,
            )
            graph.add_node(node)
            if node_col is not None and embedding is not None:
                node_ids.append(node_id)
                node_embs.append(embedding)
                node_docs.append(node.text or "")
                node_metas.append(dict(node.metadata))
                if len(node_ids) >= batch_size:
                    _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)

        for i, (raw, matrix_row) in enumerate(_iter_vdb_items(edge_file, load_embeddings=load_embeddings)):
            source_id = raw.get("src_id") or raw.get("source") or raw.get("src")
            target_id = raw.get("tgt_id") or raw.get("target") or raw.get("tgt")
            if source_id is None or target_id is None:
                continue
            source_id, target_id = str(source_id), str(target_id)

            try:
                weight = float(raw.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            relation = str(raw.get("relation", "RELATED"))

            metadata = {
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "src_id",
                    "tgt_id",
                    "source",
                    "target",
                    "src",
                    "tgt",
                    "relation",
                    "weight",
                    "vector",
                    "__vector__",
                    "embedding",
                }
            }
            edge = EdgeRecord(
                source=source_id,
                target=target_id,
                relation=relation,
                weight=weight,
                metadata=metadata,
            )
            graph.add_edge(edge)
            edge_embedding = _vector_from_item(raw, matrix_row) if load_embeddings else None
            if edge_col is not None and edge_embedding is not None:
                edge_id = str(raw.get("__id__", f"{source_id}->{target_id}#{i}"))
                edge_ids.append(edge_id)
                edge_embs.append(edge_embedding)
                edge_docs.append(f"{source_id} {relation} {target_id}")
                edge_metas.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "relation": relation,
                        "weight": weight,
                        **metadata,
                    }
                )
                if len(edge_ids) >= batch_size:
                    _flush_upsert(edge_col, edge_ids, edge_embs, edge_docs, edge_metas)

        if node_col is not None:
            _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)
            graph.attach_index(
                "node_vector",
                ChromaCollectionIndexer(
                    node_col,
                    persist_path=db_path,
                    distance_metric=str(distance_metric),
                ),
            )
        if edge_col is not None:
            _flush_upsert(edge_col, edge_ids, edge_embs, edge_docs, edge_metas)
            graph.attach_index(
                "edge_vector",
                ChromaCollectionIndexer(
                    edge_col,
                    persist_path=db_path,
                    distance_metric=str(distance_metric),
                ),
            )

        if keep_source_reference:
            for node in graph.nodes.values():
                node.metadata.setdefault("_source_path", str(source_path))
                node.metadata.setdefault("_source_style", "lightrag")

        return graph

    def export_graph(
        self,
        container: SimpleGraphContainer,
        destination: Any,
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        # Generic export for now: JSONL nodes/edges (not NanoVectorDB format).
        dest = Path(destination)
        if dest.exists() and not overwrite and any(dest.iterdir()):
            raise GraphAdapterError(f"Destination is not empty: {dest}")
        dest.mkdir(parents=True, exist_ok=True)

        nodes_path = dest / "nodes.jsonl"
        edges_path = dest / "edges.jsonl"

        with nodes_path.open("w", encoding="utf-8") as f:
            for node in container.nodes.values():
                payload = {"id": node.id, "text": node.text, "type": node.type, **node.metadata}
                if node.embedding is not None:
                    payload["embedding"] = node.embedding
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with edges_path.open("w", encoding="utf-8") as f:
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
            (dest / "manifest.json").write_text(json.dumps({"vector_store": store_info}, indent=2), encoding="utf-8")

        return {
            "destination": str(dest),
            "node_file": str(nodes_path),
            "edge_file": str(edges_path),
            "nodes": len(container.nodes),
            "edges": len(container.edges),
        }


def import_graph_from_lightrag(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
) -> SearchableGraphContainer:
    return LightRAGAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
    )
