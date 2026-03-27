# src/GraphContainer/adapters/hipporag.py
from __future__ import annotations

from ast import literal_eval
import json
import os
from glob import glob
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError
from ..indexers import ChromaCollectionIndexer


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


def _resolve_working_dir(source: Any) -> Path:
    """Resolve a dataset root to the nested HippoRAG working directory."""
    src = _normalize_source(source)

    if (src / "graph.pickle").exists():
        return src

    matches = sorted(path.parent for path in src.glob("*/graph.pickle"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise UnsupportedSourceError(
            f"Multiple HippoRAG working directories found under: {src}"
        )
    raise UnsupportedSourceError(
        f"HippoRAG working directory not found under: {src}"
    )


def _normalize_text_key(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _entity_hash_id(text: str) -> str:
    return f"entity-{md5(text.encode()).hexdigest()}"


def _parse_fact_triple(raw_content: Any) -> Optional[Tuple[str, str, str]]:
    if isinstance(raw_content, (list, tuple)) and len(raw_content) == 3:
        head, rel, tail = raw_content
        return str(head), str(rel), str(tail)

    if not isinstance(raw_content, str):
        return None

    content = raw_content.strip()
    if not content:
        return None

    try:
        parsed = literal_eval(content)
    except (SyntaxError, ValueError):
        return None

    if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
        head, rel, tail = parsed
        return str(head), str(rel), str(tail)

    return None


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


def _read_parquet(path: Path) -> Tuple[List[str], List[str], List[Any]]:
    """Read a HippoRAG EmbeddingStore parquet file.

    Returns (hash_ids, texts, embeddings).  If the file does not exist,
    returns three empty lists so callers can degrade gracefully.
    """
    if not path.exists():
        return [], [], []

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read HippoRAG parquet files") from exc

    df = pd.read_parquet(path)
    hash_ids = df["hash_id"].values.tolist()
    texts = df["content"].values.tolist()
    embeddings = df["embedding"].values.tolist()
    return hash_ids, texts, embeddings


def _load_igraph(pickle_path: Path) -> Any:
    """Load an igraph Graph from a pickle file."""
    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError("igraph (python-igraph) is required to load HippoRAG graphs") from exc

    return ig.Graph.Read_Pickle(str(pickle_path))


def _iter_openie_triples(openie_path: Path):
    """Yield (head, relation, tail) string-tuples from an OpenIE results file.

    The file has the structure::

        {"docs": [
            {"idx": ..., "passage": ...,
             "extracted_entities": [...],
             "extracted_triples": [[h, r, t], ...]},
            ...
        ], ...}
    """
    if not openie_path.exists():
        return

    with openie_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for doc in data.get("docs", []):
        passage_id = doc.get("idx", "")
        for triple in doc.get("extracted_triples", []):
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                yield str(triple[0]), str(triple[1]), str(triple[2]), passage_id


def _iter_openie_docs(openie_path: Path):
    """Yield OpenIE docs with chunk id, passage text, entities, and triples."""
    if not openie_path.exists():
        return

    with openie_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for doc in data.get("docs", []):
        entities = [
            str(entity)
            for entity in doc.get("extracted_entities", [])
            if entity is not None and str(entity).strip()
        ]
        triples = [
            (str(triple[0]), str(triple[1]), str(triple[2]))
            for triple in doc.get("extracted_triples", [])
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        ]
        yield {
            "idx": str(doc.get("idx", "")),
            "passage": str(doc.get("passage", "")),
            "entities": entities,
            "triples": triples,
        }


class HippoRAGAdapter(GraphAdapter):
    """Adapter for HippoRAG v2 working directories.

    HippoRAG v2 stores its graph in a *working directory* with the layout::

        <working_dir>/
            graph.pickle                         – igraph Graph (nodes + weighted edges)
            entity_embeddings/
                vdb_entity.parquet               – phrase/entity nodes
            chunk_embeddings/
                vdb_chunk.parquet                – passage/chunk nodes
            fact_embeddings/
                vdb_fact.parquet                 – triple-level facts

    The ``openie_results_ner_<llm>.json`` file lives one level up
    (in ``save_dir``) and is used to reconstruct edge *relations*.  The
    adapter discovers it automatically by globbing the parent directory; you
    can also pass ``openie_path`` as a kwarg to override this.

    The ``source`` argument can be either the dataset root
    (``{save_dir}``) or the nested working directory
    (``{save_dir}/{llm_name}_{embedding_model_name}``).
    """

    def __init__(self):
        super().__init__(name="topology_semantic_graph", version="2.0.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _resolve_working_dir(source)
            if not (src / "graph.pickle").exists():
                return False
            # Parquet files may sit directly in src/ or inside a model-named
            # subdirectory (e.g. gpt-4o-mini_text-embedding-3-small/).
            def _has_parquet(name: str) -> bool:
                return bool(list(src.glob(f"**/{name}")))
            if not _has_parquet("vdb_entity.parquet"):
                return False
            if not _has_parquet("vdb_chunk.parquet"):
                return False
            if not _has_parquet("vdb_fact.parquet"):
                return False
            # OpenIE file may live alongside graph.pickle (same dir)
            # OR one level up (top-level of extracted zip).
            from glob import glob as _glob
            openie_found = (
                _glob(str(src / "openie_results_ner_*.json"))
                or _glob(str(src.parent / "openie_results_ner_*.json"))
            )
            if not openie_found:
                return False
            return True
        except Exception:
            return False


    # ---------------------------------------------------------------------- #
    # import_graph                                                             #
    # ---------------------------------------------------------------------- #

    def import_graph(
        self,
        source: Any,
        container: Optional[SimpleGraphContainer] = None,
        *,
        keep_source_reference: bool = False,
        load_embeddings: bool = True,
        attach_index: Optional[bool] = None,
        openie_path: Optional[str] = None,
        **kwargs: Any,
    ) -> SearchableGraphContainer:
        """Import a HippoRAG v2 graph from *source* working directory.

        Args:
            source: Path to the HippoRAG dataset root or working directory
                that contains ``graph.pickle`` and the ``*_embeddings/``
                sub-directories.
            container: Optional pre-existing container to populate.
            keep_source_reference: If True, stores ``_source_path`` /
                ``_source_style`` in every node's metadata.
            load_embeddings: If True (default), loads stored embedding vectors
                into each :class:`NodeRecord`.
            attach_index: If True, upserts entity and chunk embeddings into a
                ChromaDB collection and attaches it as the ``node_vector``
                index so that FastInsight vector search works.  Defaults to
                the ``HIPPORAG_ATTACH_INDEX`` env-var (True when unset).
            openie_path: Explicit path to the
                ``openie_results_ner_<llm>.json`` file.  When omitted the
                adapter globs the *parent* of *source* for a matching file.
                Fact relations are primarily reconstructed from
                ``fact_embeddings/vdb_fact.parquet``.
        """
        src = _resolve_working_dir(source)

        graph_pickle = src / "graph.pickle"
        entity_parquet = src / "entity_embeddings" / "vdb_entity.parquet"
        chunk_parquet = src / "chunk_embeddings" / "vdb_chunk.parquet"
        fact_parquet = src / "fact_embeddings" / "vdb_fact.parquet"

        if not graph_pickle.exists():
            raise UnsupportedSourceError(
                f"HippoRAG graph.pickle not found under: {src}"
            )

        # ------------------------------------------------------------------ #
        # 0. Resolve index-attachment settings from env / kwargs              #
        # ------------------------------------------------------------------ #
        def _env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        if attach_index is None:
            attach_index = _env_bool("HIPPORAG_ATTACH_INDEX", True)
        batch_size = int(os.getenv("HIPPORAG_BATCH_SIZE", "5000"))
        db_path = os.getenv("VECTOR_STORE_PATH", "./data/database/chroma_db")
        distance_metric = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")

        # Set up ChromaDB collections if we will be attaching an index.
        node_col = None
        edge_col = None
        if attach_index and load_embeddings:
            import chromadb

            base_name = Path(source).resolve().name if isinstance(source, (str, Path)) else src.name
            node_col_name = f"{base_name}_nodes"
            edge_col_name = f"{base_name}_facts"
            db_dir = Path(db_path).resolve()
            db_dir.mkdir(parents=True, exist_ok=True)
            try:
                _chroma_client = chromadb.PersistentClient(path=str(db_dir))
            except Exception:
                _chroma_client = chromadb.EphemeralClient()
                db_path = None
            node_col = _chroma_client.get_or_create_collection(name=node_col_name)
            edge_col = _chroma_client.get_or_create_collection(name=edge_col_name)

        # Accumulation buffers for batched upserts.
        node_ids: List[str] = []
        node_embs: List[List[float]] = []
        node_docs: List[str] = []
        node_metas: List[Dict[str, Any]] = []
        edge_ids: List[str] = []
        edge_embs: List[List[float]] = []
        edge_docs: List[str] = []
        edge_metas: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------ #
        # 1. Load igraph → get edge topology + weights                        #
        # ------------------------------------------------------------------ #
        ig_graph = None
        try:
            ig_graph = _load_igraph(graph_pickle)
        except ImportError:
            # Fallback mode: still import nodes/fact edges from parquet files
            # even when python-igraph is unavailable.
            ig_graph = None

        # ------------------------------------------------------------------ #
        # 2. Load embedding parquet files                                      #
        # ------------------------------------------------------------------ #
        entity_ids, entity_texts, entity_embs = _read_parquet(entity_parquet)
        chunk_ids, chunk_texts, chunk_embs = _read_parquet(chunk_parquet)
        fact_ids, fact_texts, fact_embs = _read_parquet(fact_parquet)

        # ------------------------------------------------------------------ #
        # 3. Discover OpenIE file for relation labels                          #
        # ------------------------------------------------------------------ #
        resolved_openie: Optional[Path] = None
        if openie_path is not None:
            resolved_openie = Path(openie_path)
        else:
            # Glob parent directory for openie_results_ner_*.json
            parent = src.parent
            matches = sorted(glob(str(parent / "openie_results_ner_*.json")))
            if matches:
                resolved_openie = Path(matches[-1])

        # Build OpenIE lookup tables for relation enrichment and chunk-entity links.
        triple_relation_map: Dict[Tuple[str, str], str] = {}
        openie_chunk_entities: Dict[str, List[str]] = {}
        if resolved_openie is not None:
            for doc in _iter_openie_docs(resolved_openie):
                passage_id = doc["idx"]
                if passage_id:
                    openie_chunk_entities[passage_id] = doc["entities"]

                for head, rel, tail in doc["triples"]:
                    head_key = _normalize_text_key(head)
                    tail_key = _normalize_text_key(tail)
                    if head_key and tail_key:
                        triple_relation_map[(head_key, tail_key)] = rel

        # ------------------------------------------------------------------ #
        # 4. Populate the graph container                                      #
        # ------------------------------------------------------------------ #
        graph = container_or_new(container)
        graph.nodes = {}
        graph.edges = []
        graph._adj = {}
        entity_text_to_ids: Dict[str, List[str]] = {}

        # Entity nodes (phrase nodes) – processed in batches with tqdm
        entity_total = len(entity_ids)
        for batch_start in tqdm(
            range(0, max(entity_total, 1), batch_size),
            desc="Loading entity nodes",
            unit="batch",
        ):
            batch_end = min(batch_start + batch_size, entity_total)
            for hash_id, text, emb in zip(
                entity_ids[batch_start:batch_end],
                entity_texts[batch_start:batch_end],
                entity_embs[batch_start:batch_end],
            ):
                node_id = str(hash_id)
                node_text = None if text is None else str(text)
                metadata: Dict[str, Any] = {"hash_id": node_id, "node_type": "Entity"}
                if keep_source_reference:
                    metadata["_source_path"] = str(src)
                    metadata["_source_style"] = "topology_semantic_graph"
                emb_val = emb if load_embeddings else None
                graph.add_node(
                    NodeRecord(
                        id=node_id,
                        type="Entity",
                        text=node_text,
                        embedding=emb_val,
                        metadata=metadata,
                    )
                )
                text_key = _normalize_text_key(node_text)
                if text_key:
                    entity_text_to_ids.setdefault(text_key, []).append(node_id)
                # Buffer for ChromaDB upsert
                if node_col is not None and emb_val is not None:
                    emb_list = emb_val.tolist() if hasattr(emb_val, "tolist") else list(emb_val)
                    node_ids.append(node_id)
                    node_embs.append(emb_list)
                    node_docs.append(node_text or "")
                    node_metas.append(dict(metadata))
                    if len(node_ids) >= batch_size:
                        _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)

        # Chunk / passage nodes – processed in batches with tqdm
        chunk_total = len(chunk_ids)
        for batch_start in tqdm(
            range(0, max(chunk_total, 1), batch_size),
            desc="Loading chunk nodes",
            unit="batch",
        ):
            batch_end = min(batch_start + batch_size, chunk_total)
            for hash_id, text, emb in zip(
                chunk_ids[batch_start:batch_end],
                chunk_texts[batch_start:batch_end],
                chunk_embs[batch_start:batch_end],
            ):
                node_id = str(hash_id)
                node_text = None if text is None else str(text)
                metadata: Dict[str, Any] = {"hash_id": node_id, "node_type": "Chunk"}
                if keep_source_reference:
                    metadata["_source_path"] = str(src)
                    metadata["_source_style"] = "topology_semantic_graph"
                emb_val = emb if load_embeddings else None
                graph.add_node(
                    NodeRecord(
                        id=node_id,
                        type="Chunk",
                        text=node_text,
                        embedding=emb_val,
                        metadata=metadata,
                    )
                )
                # Buffer for ChromaDB upsert
                if node_col is not None and emb_val is not None:
                    emb_list = emb_val.tolist() if hasattr(emb_val, "tolist") else list(emb_val)
                    node_ids.append(node_id)
                    node_embs.append(emb_list)
                    node_docs.append(node_text or "")
                    node_metas.append(dict(metadata))
                    if len(node_ids) >= batch_size:
                        _flush_upsert(node_col, node_ids, node_embs, node_docs, node_metas)

        edge_key_to_index: Dict[Tuple[str, str, str], int] = {}
        fact_relation_by_node_ids: Dict[Tuple[str, str], str] = {}

        # Add chunk -> entity links from OpenIE extracted_entities.
        for chunk_id, entity_names in openie_chunk_entities.items():
            if chunk_id not in graph.nodes:
                continue

            for entity_name in entity_names:
                entity_key = _normalize_text_key(entity_name)
                if not entity_key:
                    continue

                entity_id = (
                    entity_text_to_ids[entity_key][0]
                    if entity_key in entity_text_to_ids
                    else _entity_hash_id(entity_key)
                )

                if entity_id not in graph.nodes:
                    metadata = {
                        "hash_id": entity_id,
                        "inferred_from_openie_entity": True,
                    }
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "topology_semantic_graph"
                    graph.add_node(
                        NodeRecord(
                            id=entity_id,
                            type="Entity",
                            text=entity_name,
                            embedding=None,
                            metadata=metadata,
                        )
                    )
                    entity_text_to_ids.setdefault(entity_key, []).append(entity_id)

                chunk_to_entity_key = (chunk_id, entity_id, "mentions")
                if chunk_to_entity_key not in edge_key_to_index:
                    graph.add_edge(
                        EdgeRecord(
                            source=chunk_id,
                            target=entity_id,
                            relation="mentions",
                            weight=1.0,
                            metadata={
                                "source": "openie",
                                "openie_passage_id": chunk_id,
                            },
                        )
                    )
                    edge_key_to_index[chunk_to_entity_key] = len(graph.edges) - 1

                entity_to_chunk_key = (entity_id, chunk_id, "mentioned_in")
                if entity_to_chunk_key in edge_key_to_index:
                    continue

                graph.add_edge(
                    EdgeRecord(
                        source=entity_id,
                        target=chunk_id,
                        relation="mentioned_in",
                        weight=1.0,
                        metadata={
                            "source": "openie",
                            "openie_passage_id": chunk_id,
                        },
                    )
                )
                edge_key_to_index[entity_to_chunk_key] = len(graph.edges) - 1

        # ------------------------------------------------------------------ #
        # 5. Add fact edges from vdb_fact.parquet (with edge embedding upsert)#
        # ------------------------------------------------------------------ #
        fact_total = len(fact_ids)
        for batch_start in tqdm(
            range(0, max(fact_total, 1), batch_size),
            desc="Loading fact edges",
            unit="batch",
        ):
            batch_end = min(batch_start + batch_size, fact_total)
            for fact_id, fact_content, fact_emb in zip(
                fact_ids[batch_start:batch_end],
                fact_texts[batch_start:batch_end],
                fact_embs[batch_start:batch_end],
            ):
                triple = _parse_fact_triple(fact_content)
                if triple is None:
                    continue

                head, relation, tail = triple
                head_key = _normalize_text_key(head)
                tail_key = _normalize_text_key(tail)
                if not head_key or not tail_key:
                    continue

                source_id = (
                    entity_text_to_ids[head_key][0]
                    if head_key in entity_text_to_ids
                    else _entity_hash_id(head_key)
                )
                target_id = (
                    entity_text_to_ids[tail_key][0]
                    if tail_key in entity_text_to_ids
                    else _entity_hash_id(tail_key)
                )

                if source_id not in graph.nodes:
                    metadata: Dict[str, Any] = {
                        "hash_id": source_id,
                        "inferred_from_fact": True,
                    }
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "topology_semantic_graph"
                    graph.add_node(
                        NodeRecord(
                            id=source_id,
                            type="Entity",
                            text=head_key,
                            embedding=None,
                            metadata=metadata,
                        )
                    )
                    entity_text_to_ids.setdefault(head_key, []).append(source_id)

                if target_id not in graph.nodes:
                    metadata = {
                        "hash_id": target_id,
                        "inferred_from_fact": True,
                    }
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "topology_semantic_graph"
                    graph.add_node(
                        NodeRecord(
                            id=target_id,
                            type="Entity",
                            text=tail_key,
                            embedding=None,
                            metadata=metadata,
                        )
                    )
                    entity_text_to_ids.setdefault(tail_key, []).append(target_id)

                relation_text = str(relation).strip() if relation is not None else ""

                edge_key = (source_id, target_id, relation_text)
                if edge_key in edge_key_to_index:
                    continue

                graph.add_edge(
                    EdgeRecord(
                        source=source_id,
                        target=target_id,
                        relation=relation_text,
                        weight=1.0,
                        metadata={
                            "source": "vdb_fact",
                            "fact_hash_id": str(fact_id),
                        },
                    )
                )
                edge_key_to_index[edge_key] = len(graph.edges) - 1
                fact_relation_by_node_ids.setdefault((source_id, target_id), relation_text)

                # Buffer fact embedding for edge_col upsert
                if edge_col is not None and load_embeddings and fact_emb is not None:
                    emb_list = fact_emb.tolist() if hasattr(fact_emb, "tolist") else list(fact_emb)
                    edge_ids.append(str(fact_id))
                    edge_embs.append(emb_list)
                    edge_docs.append(str(fact_content) if fact_content is not None else "")
                    edge_metas.append({
                        "source": source_id,
                        "target": target_id,
                        "relation": relation_text,
                        "fact_hash_id": str(fact_id),
                    })
                    if len(edge_ids) >= batch_size:
                        _flush_upsert(edge_col, edge_ids, edge_embs, edge_docs, edge_metas)


        # Guard: if no nodes were loaded from parquet, surface igraph vertices
        if ig_graph is not None and not graph.nodes and ig_graph.vcount() > 0:
            for v in ig_graph.vs:
                vid = v["name"] if "name" in v.attributes() else str(v.index)
                if vid not in graph.nodes:
                    metadata = {"hash_id": vid}
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "topology_semantic_graph"
                    graph.add_node(
                        NodeRecord(
                            id=vid,
                            type="Unknown",
                            text=vid,
                            embedding=None,
                            metadata=metadata,
                        )
                    )

        # ------------------------------------------------------------------ #
        # 6. Add edges from igraph (topology + weight)                         #
        # ------------------------------------------------------------------ #
        if ig_graph is not None:
            for edge in ig_graph.es:
                src_v = ig_graph.vs[edge.source]
                tgt_v = ig_graph.vs[edge.target]
                src_id = str(src_v["name"]) if "name" in src_v.attributes() else str(edge.source)
                tgt_id = str(tgt_v["name"]) if "name" in tgt_v.attributes() else str(edge.target)

                try:
                    weight = float(edge["weight"])
                except (KeyError, TypeError, ValueError):
                    weight = 1.0

                relation = fact_relation_by_node_ids.get((src_id, tgt_id))
                if relation is None:
                    src_text_key = _normalize_text_key(
                        graph.nodes[src_id].text if src_id in graph.nodes else ""
                    )
                    tgt_text_key = _normalize_text_key(
                        graph.nodes[tgt_id].text if tgt_id in graph.nodes else ""
                    )
                    relation = triple_relation_map.get((src_text_key, tgt_text_key), "related")

                edge_key = (src_id, tgt_id, relation)
                if edge_key in edge_key_to_index:
                    existing_idx = edge_key_to_index[edge_key]
                    if graph.edges[existing_idx].weight < weight:
                        graph.edges[existing_idx].weight = weight
                    continue

                graph.add_edge(
                    EdgeRecord(
                        source=src_id,
                        target=tgt_id,
                        relation=relation,
                        weight=weight,
                        metadata={"source": "igraph"},
                    )
                )
                edge_key_to_index[edge_key] = len(graph.edges) - 1

        # ------------------------------------------------------------------ #
        # 7. Flush remaining buffered upserts and attach vector indices        #
        # ------------------------------------------------------------------ #
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

        return graph

    def export_graph(
        self,
        container: SimpleGraphContainer,
        destination: Any,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Export *container* to ``nodes.jsonl`` + ``edges.jsonl``."""
        dest = Path(destination)
        if dest.exists() and not overwrite and any(dest.iterdir()):
            raise GraphAdapterError(f"Destination is not empty: {dest}")
        dest.mkdir(parents=True, exist_ok=True)

        nodes_path = dest / "nodes.jsonl"
        edges_path = dest / "edges.jsonl"

        with nodes_path.open("w", encoding="utf-8") as f:
            for node in container.nodes.values():
                payload: Dict[str, Any] = {
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
            (dest / "manifest.json").write_text(
                json.dumps({"vector_store": store_info}, indent=2), encoding="utf-8"
            )

        return {
            "destination": str(dest),
            "node_file": str(nodes_path),
            "edge_file": str(edges_path),
            "nodes": len(container.nodes),
            "edges": len(container.edges),
        }


# --------------------------------------------------------------------------- #
# Module-level convenience wrapper                                              #
# --------------------------------------------------------------------------- #

def import_graph_from_hipporag(
    source: Any,
    *,
    container: Optional[SimpleGraphContainer] = None,
    keep_source_reference: bool = False,
    load_embeddings: bool = True,
    attach_index: Optional[bool] = None,
    openie_path: Optional[str] = None,
    **kwargs: Any,
) -> SearchableGraphContainer:
    """Convenience wrapper around :class:`HippoRAGAdapter`.

    Args:
        source: Path to the HippoRAG *working directory*
            (``{save_dir}/{llm_name}_{embedding_model_name}``).
        container: Optional existing container to populate.
        keep_source_reference: Store ``_source_path``/``_source_style`` on
            every node.
        load_embeddings: Load stored embedding vectors into nodes.
        attach_index: If True, upserts embeddings into ChromaDB and attaches a
            ``node_vector`` index for FastInsight retrieval.  Defaults to the
            ``HIPPORAG_ATTACH_INDEX`` env-var (True when unset).
        openie_path: Explicit path to ``openie_results_ner_*.json``.
            Auto-detected from the parent directory when omitted.
    """
    return HippoRAGAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        load_embeddings=load_embeddings,
        attach_index=attach_index,
        openie_path=openie_path,
        **kwargs,
    )
