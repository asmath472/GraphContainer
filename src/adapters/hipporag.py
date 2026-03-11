# src/GraphContainer/adapters/hipporag.py
from __future__ import annotations

from ast import literal_eval
import json
from glob import glob
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core import SearchableGraphContainer, SimpleGraphContainer
from ..types import EdgeRecord, NodeRecord
from ..utils import container_or_new
from .base import GraphAdapter, GraphAdapterError, UnsupportedSourceError


def _normalize_source(source: Any) -> Path:
    if isinstance(source, (str, Path)):
        return Path(source)
    raise UnsupportedSourceError(f"Unsupported source type: {type(source)}")


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

    The ``source`` argument should be the *working directory*
    (``{save_dir}/{llm_name}_{embedding_model_name}``).
    """

    def __init__(self):
        super().__init__(name="hipporag", version="2.0.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            return (src / "graph.pickle").exists() and (
                (src / "entity_embeddings" / "vdb_entity.parquet").exists()
                or (src / "chunk_embeddings" / "vdb_chunk.parquet").exists()
            )
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
        openie_path: Optional[str] = None,
        **kwargs: Any,
    ) -> SearchableGraphContainer:
        """Import a HippoRAG v2 graph from *source* working directory.

        Args:
            source: Path to the HippoRAG working directory that contains
                ``graph.pickle`` and the ``*_embeddings/`` sub-directories.
            container: Optional pre-existing container to populate.
            keep_source_reference: If True, stores ``_source_path`` /
                ``_source_style`` in every node's metadata.
            load_embeddings: If True (default), loads stored embedding vectors
                into each :class:`NodeRecord`.
            openie_path: Explicit path to the
                ``openie_results_ner_<llm>.json`` file.  When omitted the
                adapter globs the *parent* of *source* for a matching file.
                Fact relations are primarily reconstructed from
                ``fact_embeddings/vdb_fact.parquet``.
        """
        src = _normalize_source(source)

        graph_pickle = src / "graph.pickle"
        entity_parquet = src / "entity_embeddings" / "vdb_entity.parquet"
        chunk_parquet = src / "chunk_embeddings" / "vdb_chunk.parquet"
        fact_parquet = src / "fact_embeddings" / "vdb_fact.parquet"

        if not graph_pickle.exists():
            raise UnsupportedSourceError(
                f"HippoRAG graph.pickle not found under: {src}"
            )

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
        fact_ids, fact_texts, _ = _read_parquet(fact_parquet)

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

        # Build a set of known (head, tail) → relation for edge enrichment
        triple_relation_map: Dict[Tuple[str, str], str] = {}
        if resolved_openie is not None:
            for head, rel, tail, _ in _iter_openie_triples(resolved_openie):
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

        # Entity nodes (phrase nodes)
        for hash_id, text, emb in zip(entity_ids, entity_texts, entity_embs):
            node_id = str(hash_id)
            node_text = None if text is None else str(text)
            metadata: Dict[str, Any] = {"hash_id": node_id}
            if keep_source_reference:
                metadata["_source_path"] = str(src)
                metadata["_source_style"] = "hipporag"
            graph.add_node(
                NodeRecord(
                    id=node_id,
                    type="Entity",
                    text=node_text,
                    embedding=emb if load_embeddings else None,
                    metadata=metadata,
                )
            )
            text_key = _normalize_text_key(node_text)
            if text_key:
                entity_text_to_ids.setdefault(text_key, []).append(node_id)

        # Chunk / passage nodes
        for hash_id, text, emb in zip(chunk_ids, chunk_texts, chunk_embs):
            node_id = str(hash_id)
            node_text = None if text is None else str(text)
            metadata: Dict[str, Any] = {"hash_id": node_id}
            if keep_source_reference:
                metadata["_source_path"] = str(src)
                metadata["_source_style"] = "hipporag"
            graph.add_node(
                NodeRecord(
                    id=node_id,
                    type="Chunk",
                    text=node_text,
                    embedding=emb if load_embeddings else None,
                    metadata=metadata,
                )
            )

        edge_key_to_index: Dict[Tuple[str, str, str], int] = {}
        fact_relation_by_node_ids: Dict[Tuple[str, str], str] = {}

        # ------------------------------------------------------------------ #
        # 5. Add fact edges from vdb_fact.parquet                             #
        # ------------------------------------------------------------------ #
        for fact_id, fact_content in zip(fact_ids, fact_texts):
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
                    metadata["_source_style"] = "hipporag"
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
                    metadata["_source_style"] = "hipporag"
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
            if not relation_text:
                relation_text = "related"

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

        # Guard: if no nodes were loaded from parquet, surface igraph vertices
        if ig_graph is not None and not graph.nodes and ig_graph.vcount() > 0:
            for v in ig_graph.vs:
                vid = v["name"] if "name" in v.attributes() else str(v.index)
                if vid not in graph.nodes:
                    metadata = {"hash_id": vid}
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "hipporag"
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
        openie_path: Explicit path to ``openie_results_ner_*.json``.
            Auto-detected from the parent directory when omitted.
    """
    return HippoRAGAdapter().import_graph(
        source=source,
        container=container,
        keep_source_reference=keep_source_reference,
        load_embeddings=load_embeddings,
        openie_path=openie_path,
        **kwargs,
    )
