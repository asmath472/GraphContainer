# src/GraphContainer/adapters/lightrag.py
from __future__ import annotations

import base64
import json
import os
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

import dotenv
from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError
from ..indexers import ChromaCollectionIndexer, to_float_list

dotenv.load_dotenv()  # Load environment variables from .env file, if present


def _log(msg: str) -> None:
    if _env_bool("LIGHTRAG_LOG", True):
        print(f"[lightrag] {msg}")


def _log_timing(label: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    _log(f"{label} in {elapsed:.2f}s")


def _maybe_pdb(label: str) -> None:
    if _env_bool("LIGHTRAG_PDB", False):
        _log(f"pdb break: {label}")
        import pdb

        pdb.set_trace()


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _decode_vector_batch(encoded_list: List[Optional[str]]) -> List[Optional[List[float]]]:
    """Batch decode Base64+zlib float16 vectors; returns None for failures."""
    try:
        import numpy as np
    except ImportError:
        return [None] * len(encoded_list)

    out: List[Optional[List[float]]] = []
    for encoded in encoded_list:
        if not encoded:
            out.append(None)
            continue
        try:
            decoded = base64.b64decode(encoded)
            decompressed = zlib.decompress(decoded)
            vec_f16 = np.frombuffer(decompressed, dtype=np.float16)
            out.append(vec_f16.astype(np.float32).tolist())
        except Exception:
            out.append(None)
    return out


def _iter_vdb_batches(
    path: Path, *, load_embeddings: bool, batch_size: int
) -> Iterable[Tuple[List[Dict[str, Any]], List[Any]]]:
    # When embeddings are required, load once and slice in memory.
    # This avoids any ambiguity about repeated file reads while batching.
    if load_embeddings:
        _log(f"using single-pass json.load path for {path.name} (embeddings enabled)")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("data", [])
        matrix = payload.get("matrix", [])
        for i in range(0, len(data), batch_size):
            raw_batch = data[i : i + batch_size]
            matrix_batch = matrix[i : i + batch_size] if isinstance(matrix, list) else []
            items: List[Dict[str, Any]] = []
            rows: List[Any] = []
            for j, item in enumerate(raw_batch):
                if not isinstance(item, dict):
                    continue
                items.append(item)
                rows.append(matrix_batch[j] if j < len(matrix_batch) else None)
            if not items:
                continue
            yield items, rows
        return

    # Embeddings disabled: prefer low-memory streaming when available.
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None

    if ijson is not None:
        _log(f"using ijson streaming path for {path.name} (embeddings disabled)")
        with path.open("rb") as f:
            parser = ijson.items(f, "data.item")
            items: List[Dict[str, Any]] = []
            for item in parser:
                if not isinstance(item, dict):
                    continue
                items.append(item)
                if len(items) >= batch_size:
                    yield items, [None] * len(items)
                    items = []
            if items:
                yield items, [None] * len(items)
        return

    _log(f"using json.load path for {path.name} (embeddings disabled)")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data", [])
    for i in range(0, len(data), batch_size):
        items = [x for x in data[i : i + batch_size] if isinstance(x, dict)]
        if not items:
            continue
        yield items, [None] * len(items)


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
        super().__init__(name="attribute_bundle_graph", version=version)

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
        load_edge_embeddings = _env_bool("LIGHTRAG_LOAD_EDGE_EMBEDDINGS", True)
        batch_size = int(os.getenv("LIGHTRAG_BATCH_SIZE", "5000"))

        import_start = time.perf_counter()
        _log(
            "import start "
            f"(attach_index={attach_index}, load_embeddings={load_embeddings}, "
            f"load_edge_embeddings={load_edge_embeddings}, batch_size={batch_size})"
        )
        _maybe_pdb("import start")

        graph = container_or_new(container)
        graph.nodes = {}
        graph.edges = []
        graph._adj = {}

        # ── Load vdb files (slow for large graphs, includes embeddings)
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

        nodes_start = time.perf_counter()
        _maybe_pdb("before node load")
        for raw_batch, matrix_batch in tqdm(
            _iter_vdb_batches(node_file, load_embeddings=load_embeddings, batch_size=batch_size),
            desc="Loading nodes",
            unit="batch",
        ):
            if load_embeddings:
                if matrix_batch and any(row is not None for row in matrix_batch):
                    embeddings = [to_float_list(row) for row in matrix_batch]
                else:
                    encoded = [
                        (r.get("__vector__") or r.get("embedding") or r.get("vector"))
                        if isinstance(r, dict)
                        else None
                        for r in raw_batch
                    ]
                    embeddings = _decode_vector_batch(encoded)
            else:
                embeddings = [None] * len(raw_batch)

            for raw, embedding in zip(raw_batch, embeddings):
                node_id = raw.get("id") or raw.get("__id__")
                if node_id is None:
                    continue
                node_id = str(node_id)

                metadata = {
                    k: v
                    for k, v in raw.items()
                    if k not in {"id", "__id__", "content", "vector", "__vector__", "embedding"}
                }

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

        _log_timing("node load complete", nodes_start)
        _maybe_pdb("after node load")

        edges_start = time.perf_counter()
        _maybe_pdb("before edge load")
        edge_i = 0
        for raw_batch, matrix_batch in tqdm(
            _iter_vdb_batches(edge_file, load_embeddings=load_embeddings, batch_size=batch_size),
            desc="Loading edges",
            unit="batch",
        ):
            if load_embeddings and load_edge_embeddings:
                if matrix_batch and any(row is not None for row in matrix_batch):
                    embeddings = [to_float_list(row) for row in matrix_batch]
                else:
                    encoded = [
                        (r.get("__vector__") or r.get("embedding") or r.get("vector"))
                        if isinstance(r, dict)
                        else None
                        for r in raw_batch
                    ]
                    embeddings = _decode_vector_batch(encoded)
            else:
                embeddings = [None] * len(raw_batch)

            for raw, edge_embedding in zip(raw_batch, embeddings):
                i = edge_i
                edge_i += 1
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

        _log_timing("edge load complete", edges_start)
        _maybe_pdb("after edge load")

        if node_col is not None:
            index_start = time.perf_counter()
            _maybe_pdb("before node index attach")
            _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)
            graph.attach_index(
                "node_vector",
                ChromaCollectionIndexer(
                    node_col,
                    persist_path=db_path,
                    distance_metric=str(distance_metric),
                ),
            )
            _log_timing("node index attach", index_start)
            _maybe_pdb("after node index attach")
        if edge_col is not None:
            index_start = time.perf_counter()
            _maybe_pdb("before edge index attach")
            _flush_upsert(edge_col, edge_ids, edge_embs, edge_docs, edge_metas)
            graph.attach_index(
                "edge_vector",
                ChromaCollectionIndexer(
                    edge_col,
                    persist_path=db_path,
                    distance_metric=str(distance_metric),
                ),
            )
            _log_timing("edge index attach", index_start)
            _maybe_pdb("after edge index attach")

        if keep_source_reference:
            for node in graph.nodes.values():
                node.metadata.setdefault("_source_path", str(source_path))
                node.metadata.setdefault("_source_style", "attribute_bundle_graph")

        _log_timing("import complete", import_start)
        _maybe_pdb("import complete")

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
