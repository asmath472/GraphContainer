from __future__ import annotations

import csv
import json
import os
import sys
import types
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError
from tqdm import tqdm

from ..indexers import ChromaCollectionIndexer


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


def _flush_upsert(
    col: Any,
    ids: List[str],
    embs: List[List[float]],
    docs: List[str],
    metas: List[Dict[str, Any]],
) -> None:
    """Upsert buffered items into a ChromaDB collection then clear the buffers."""
    if ids:
        col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        ids.clear()
        embs.clear()
        docs.clear()
        metas.clear()


# ---------------------------------------------------------------------------
# Install torch_geometric stubs once at module load time so that the 74k+
# per-graph torch.load() calls don't pay a stub-install/teardown overhead.
# ---------------------------------------------------------------------------
_TG_SUBMODULES = [
    "torch_geometric",
    "torch_geometric.data",
    "torch_geometric.data.data",
    "torch_geometric.data.storage",
    "torch_geometric.data.feature_store",
    "torch_geometric.transforms",
    "torch_geometric.utils",
    "torch_geometric.nn",
]
_TG_CLASSES = [
    "Data", "HeteroData", "DataEdgeAttr", "DataTensorAttr",
    "GlobalStorage", "NodeStorage", "EdgeStorage",
]

def _ensure_tg_stubs() -> None:
    """Install lightweight torch_geometric stubs if the real package is absent."""
    if "torch_geometric" in sys.modules:
        return  # real package (or prior stubs) already installed
    for mod_name in _TG_SUBMODULES:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m
    for mod_name in _TG_SUBMODULES:
        m = sys.modules[mod_name]
        for cls_name in _TG_CLASSES:
            cls = type(cls_name, (), {
                "__init__": lambda self, *a, **kw: self.__dict__.update(kw),
                "__setattr__": lambda self, k, v: self.__dict__.__setitem__(k, v),
            })
            setattr(m, cls_name, cls)


_ensure_tg_stubs()
# Cache the torch module reference after stubs are installed.
_torch: Any = None
try:
    import torch as _torch  # noqa: E402
except ImportError:
    _torch = None


def _load_pt_embeddings(pt_path: Path) -> Tuple[Optional[Any], Optional[Any]]:
    """Load node and edge embeddings from a torch_geometric .pt file.

    Returns ``(x, edge_attr)`` tensors or ``(None, None)`` on any failure.
    ``torch_geometric`` is **not** a required runtime dependency — lightweight
    stubs are installed once at import time for pickle compatibility.
    """
    if _torch is None or not pt_path.exists():
        return None, None

    try:
        data = _torch.load(str(pt_path), map_location="cpu", weights_only=False)
        mapping: Dict[str, Any] = {}
        store = data.__dict__.get("_store")
        if store is not None:
            mapping = store.__dict__.get("_mapping", {})
        elif isinstance(data, dict):
            mapping = data

        return mapping.get("x"), mapping.get("edge_attr")
    except Exception:
        return None, None


def _iter_graph_batches(
    graph_ids: List[str],
    src_path: Path,
    graphs_dir: Path,
    batch_size: int,
) -> Iterable[Tuple[List[str], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[Any], List[Any]]]:
    """Yield batches of ``batch_size`` graphs at a time.

    Each yielded tuple contains:
    - ``gid_batch``       – list of graph IDs in this batch
    - ``node_rows_batch`` – per-graph list of node CSV row dicts
    - ``edge_rows_batch`` – per-graph list of edge CSV row dicts
    - ``x_batch``         – per-graph node-embedding tensors (or None)
    - ``ea_batch``        – per-graph edge-embedding tensors (or None)

    Mimics :func:`lightrag._iter_vdb_batches`: all data for the batch is
    loaded eagerly so the caller can iterate without worrying about file
    handles or re-reads.
    """
    nodes_dir = src_path / "nodes"
    edges_dir = src_path / "edges"

    for i in range(0, len(graph_ids), batch_size):
        gid_batch = graph_ids[i : i + batch_size]
        node_rows_batch: List[List[Dict[str, Any]]] = []
        edge_rows_batch: List[List[Dict[str, Any]]] = []
        x_batch: List[Any] = []
        ea_batch: List[Any] = []

        for gid in gid_batch:
            # Load CSV rows eagerly
            node_csv = nodes_dir / f"{gid}.csv"
            edge_csv = edges_dir / f"{gid}.csv"
            node_rows: List[Dict[str, Any]] = []
            edge_rows: List[Dict[str, Any]] = []
            if node_csv.exists():
                with node_csv.open("r", encoding="utf-8", newline="") as f:
                    node_rows = [r for r in csv.DictReader(f) if r]
            if edge_csv.exists():
                with edge_csv.open("r", encoding="utf-8", newline="") as f:
                    edge_rows = [r for r in csv.DictReader(f) if r]

            # Load .pt embeddings
            x, ea = _load_pt_embeddings(graphs_dir / f"{gid}.pt")

            node_rows_batch.append(node_rows)
            edge_rows_batch.append(edge_rows)
            x_batch.append(x)
            ea_batch.append(ea)

        yield gid_batch, node_rows_batch, edge_rows_batch, x_batch, ea_batch


class GRetrieverAdapter(GraphAdapter):
    def __init__(self):
        super().__init__(name="subgraph_union_graph", version="0.1.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            # Require nodes/ and edges/ CSV directories.
            has_csv_dirs = (src / "nodes").is_dir() and (src / "edges").is_dir()
            has_flat_csvs = (src / "nodes.csv").exists() and (src / "edges.csv").exists()
            if not (has_csv_dirs or has_flat_csvs):
                return False
            # Require graphs/ directory with at least one .pt embedding file.
            graphs_dir = src / "graphs"
            if not graphs_dir.is_dir():
                return False
            if not any(graphs_dir.glob("*.pt")):
                return False
            return True
        except Exception:
            return False


    def import_graph(
        self,
        source: Any,
        container: Optional[SimpleGraphContainer] = None,
        *,
        keep_source_reference: bool = False,
        attach_index: Optional[bool] = None,
        **kwargs: Any,
    ) -> SearchableGraphContainer:
        """Import a GRetriever-style graph from *source*.

        Args:
            source: Path to the dataset root that contains ``nodes/`` and
                ``edges/`` subdirectories (or ``nodes.csv`` / ``edges.csv``
                directly).
            container: Optional pre-existing container to populate.
            keep_source_reference: Store ``_source_path``/``_source_style``
                on every node and edge.
            attach_index: If True (default), load per-graph ``graphs/<id>.pt``
                files and upsert the ``x`` (node) and ``edge_attr`` (edge)
                tensors into ChromaDB, attaching them as the ``node_vector``
                and ``edge_vector`` indices.  Falls back to the
                ``GRETRIEVER_ATTACH_INDEX`` env-var (default True).
        """
        # ------------------------------------------------------------------ #
        # 0. Settings                                                          #
        # ------------------------------------------------------------------ #
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        if attach_index is None:
            attach_index = _env_bool("GRETRIEVER_ATTACH_INDEX", True)
        batch_size = int(os.getenv("GRETRIEVER_BATCH_SIZE", "2500"))
        db_path = os.getenv("VECTOR_STORE_PATH", "./data/database/chroma_db")
        distance_metric = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")

        src_path = _normalize_source(source)
        graph = container_or_new(container)

        # ------------------------------------------------------------------ #
        # 1. Set up ChromaDB collections                                       #
        # ------------------------------------------------------------------ #
        node_col = None
        edge_col = None
        if attach_index:
            try:
                import chromadb

                base_name = src_path.resolve().name
                db_dir = Path(db_path).resolve()
                db_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _chroma_client = chromadb.PersistentClient(path=str(db_dir))
                except Exception:
                    _chroma_client = chromadb.EphemeralClient()
                    db_path = None
                node_col = _chroma_client.get_or_create_collection(
                    name=f"{base_name}_nodes"
                )
                edge_col = _chroma_client.get_or_create_collection(
                    name=f"{base_name}_edges"
                )
            except ImportError:
                node_col = None
                edge_col = None

        # Upsert accumulation buffers
        node_ids: List[str] = []
        node_embs: List[List[float]] = []
        node_docs: List[str] = []
        node_metas: List[Dict[str, Any]] = []
        node_ids_seen: set = set()  # dedup across graphs in-flight

        edge_ids_buf: List[str] = []
        edge_embs_buf: List[List[float]] = []
        edge_docs_buf: List[str] = []
        edge_metas_buf: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------ #
        # 2. Load all graph-ids                                                #
        # ------------------------------------------------------------------ #
        graph_ids = _list_graph_ids(src_path)
        node_key_to_global: Dict[Tuple[str, str], str] = {}
        edge_key_to_idx: Dict[Tuple[str, str, str], int] = {}

        graphs_dir = src_path / "graphs"

        for gid_batch, node_rows_batch, edge_rows_batch, x_batch, ea_batch in tqdm(
            _iter_graph_batches(graph_ids, src_path, graphs_dir, batch_size),
            desc="Loading graphs",
            unit="batch",
        ):
            for gid, node_rows, edge_rows, x_tensor, edge_attr_tensor in zip(
                gid_batch, node_rows_batch, edge_rows_batch, x_batch, ea_batch
            ):
                local_to_global: Dict[str, str] = {}

                # ---------------------------------------------------------- #
                # 2a. Nodes                                                    #
                # ---------------------------------------------------------- #
                for row_idx, raw in enumerate(node_rows):
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

                    # Extract per-node embedding from .pt if available
                    node_emb: Optional[List[float]] = None
                    if x_tensor is not None and row_idx < x_tensor.shape[0]:
                        try:
                            node_emb = x_tensor[row_idx].tolist()
                        except Exception:
                            node_emb = None

                    if global_id in graph.nodes:
                        existing_meta = graph.nodes[global_id].metadata
                        _append_unique(existing_meta, "graph_ids", str(gid))
                        _append_unique(existing_meta, "original_node_ids", f"{gid}:{local_id}")
                        if node_emb is not None and graph.nodes[global_id].embedding is None:
                            graph.nodes[global_id].embedding = node_emb
                    else:
                        metadata = {
                            k: v
                            for k, v in raw.items()
                            if k not in {"node_id", "id", "node_attr", "text", "type"}
                        }
                        metadata["graph_ids"] = [str(gid)]
                        metadata["original_node_ids"] = [f"{gid}:{local_id}"]
                        if keep_source_reference:
                            metadata.setdefault("_source_path", str(src_path))
                            metadata.setdefault("_source_style", "subgraph_union_graph")

                        graph.add_node(
                            NodeRecord(
                                id=global_id,
                                type=node_type,
                                text=node_text,
                                embedding=node_emb,
                                metadata=metadata,
                            )
                        )
                        node_key_to_global[node_key] = global_id

                    # Buffer for ChromaDB upsert (dedup by global_id within batch)
                    if node_col is not None and node_emb is not None and global_id not in node_ids_seen:
                        node_ids_seen.add(global_id)
                        node_ids.append(global_id)
                        node_embs.append(node_emb)
                        node_docs.append(str(node_text or ""))
                        node_metas.append({"graph_id": str(gid), "local_id": local_id})
                        if len(node_ids) >= batch_size:
                            _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)
                            node_ids_seen.clear()

                # ---------------------------------------------------------- #
                # 2b. Edges                                                    #
                # ---------------------------------------------------------- #
                for row_idx, raw in enumerate(edge_rows):
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

                    # Extract per-edge embedding from .pt if available
                    edge_emb: Optional[List[float]] = None
                    if edge_attr_tensor is not None and row_idx < edge_attr_tensor.shape[0]:
                        try:
                            edge_emb = edge_attr_tensor[row_idx].tolist()
                        except Exception:
                            edge_emb = None

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
                        if k not in {"src", "source", "dst", "target", "edge_attr", "relation", "weight"}
                    }
                    metadata["graph_ids"] = [str(gid)]
                    metadata["original_edge_ids"] = [f"{gid}:{src_local}->{tgt_local}"]
                    if keep_source_reference:
                        metadata.setdefault("_source_path", str(src_path))
                        metadata.setdefault("_source_style", "subgraph_union_graph")

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

                    # Buffer for ChromaDB upsert
                    if edge_col is not None and edge_emb is not None:
                        edge_id = f"{gid}:{row_idx}:{src_local}->{tgt_local}"
                        edge_ids_buf.append(edge_id)
                        edge_embs_buf.append(edge_emb)
                        edge_docs_buf.append(f"{source_global} {relation} {target_global}")
                        edge_metas_buf.append({
                            "graph_id": str(gid),
                            "source": source_global,
                            "target": target_global,
                            "relation": relation,
                        })
                        if len(edge_ids_buf) >= batch_size:
                            _flush_upsert(edge_col, edge_ids_buf, edge_embs_buf, edge_docs_buf, edge_metas_buf)

            # Flush remaining buffers at end of each tqdm batch
            if node_col is not None and node_ids:
                _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)
                node_ids_seen.clear()
            if edge_col is not None and edge_ids_buf:
                _flush_upsert(edge_col, edge_ids_buf, edge_embs_buf, edge_docs_buf, edge_metas_buf)

        # 3. Final flush + attach indices                                      #
        # ------------------------------------------------------------------ #
        if node_col is not None:
            _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)
            node_ids_seen.clear()
            graph.attach_index(
                "node_vector",
                ChromaCollectionIndexer(
                    node_col,
                    persist_path=db_path,
                    distance_metric=str(distance_metric),
                ),
            )
        if edge_col is not None:
            _flush_upsert(edge_col, edge_ids_buf, edge_embs_buf, edge_docs_buf, edge_metas_buf)
            graph.attach_index(
                "edge_vector",
                ChromaCollectionIndexer(
                    edge_col,
                    persist_path=db_path,
                    distance_metric=str(distance_metric),
                ),
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
    attach_index: Optional[bool] = None,
    **kwargs: Any,
) -> SearchableGraphContainer:
    """Convenience wrapper around :class:`GRetrieverAdapter`.

    Args:
        source: Path to the dataset root with ``nodes/`` and ``edges/`` dirs.
        container: Optional existing container to populate.
        keep_source_reference: Store ``_source_path``/``_source_style`` on
            every node and edge.
        attach_index: Load ``.pt`` embeddings and attach ``node_vector`` /
            ``edge_vector`` ChromaDB indices.  Defaults to the
            ``GRETRIEVER_ATTACH_INDEX`` env-var (True when unset).
    """
    return GRetrieverAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        attach_index=attach_index,
        **kwargs,
    )
