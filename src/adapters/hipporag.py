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


def _load_igraph(pickle_path: Path) -> Any:
    """Load an igraph Graph from a pickle file."""
    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError("igraph (python-igraph) is required to load HippoRAG graphs") from exc

    return ig.Graph.Read_Pickle(str(pickle_path))


def _iter_openie_triples(openie_path: Path):
    """Yield (head, relation, tail, passage_id, passage_text) tuples from OpenIE results.

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
        passage_text = doc.get("passage", "")
        for triple in doc.get("extracted_triples", []):
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                yield str(triple[0]), str(triple[1]), str(triple[2]), passage_id, passage_text


class HippoRAGAdapter(GraphAdapter):
    """Adapter for HippoRAG v2 working directories.

    HippoRAG v2 stores its graph in a *working directory* with the layout::

        <working_dir>/
            graph.pickle                         – igraph Graph (nodes + weighted edges)
            openie_results_ner_<llm>.json        – OpenIE triples (used here)

    The ``openie_results_ner_<llm>.json`` file is used to reconstruct nodes
    and edge relations. The adapter discovers it automatically by globbing
    the working directory first, then the parent directory. You can also pass
    ``openie_path`` as a kwarg to override this.

    The ``source`` argument should be the *working directory*
    (``{save_dir}/{llm_name}_{embedding_model_name}``).
    """

    def __init__(self):
        super().__init__(name="hipporag", version="2.0.0")

    def can_import(self, source: Any) -> bool:
        try:
            src = _normalize_source(source)
            if not (src / "graph.pickle").exists():
                return False
            if (src / "openie_results_ner_gpt-4o-mini.json").exists():
                return True
            if any((src / f).exists() for f in ["openie_results_ner_gpt-4o.json"]):
                return True
            # Fallback: allow any openie_results_ner_*.json in src or parent
            if glob(str(src / "openie_results_ner_*.json")):
                return True
            if glob(str(src.parent / "openie_results_ner_*.json")):
                return True
            return False
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
        # 2. OpenIE-only mode (no parquet)                                    #
        # ------------------------------------------------------------------ #
        openie_only = True

        # ------------------------------------------------------------------ #
        # 3. Discover OpenIE file for relation labels                          #
        # ------------------------------------------------------------------ #
        resolved_openie: Optional[Path] = None
        if openie_path is not None:
            resolved_openie = Path(openie_path)
        else:
            # Prefer openie_results_ner_*.json under the working dir, then parent.
            matches = sorted(glob(str(src / "openie_results_ner_*.json")))
            if not matches:
                parent = src.parent
                matches = sorted(glob(str(parent / "openie_results_ner_*.json")))
            if matches:
                resolved_openie = Path(matches[-1])

        # Build a set of known (head, tail) → relation for edge enrichment
        triple_relation_map: Dict[Tuple[str, str], str] = {}
        entity_passage_map: Dict[str, str] = {}
        edge_passage_map: Dict[Tuple[str, str], str] = {}
        if resolved_openie is not None:
            for head, rel, tail, _, passage_text in _iter_openie_triples(resolved_openie):
                head_key = _normalize_text_key(head)
                tail_key = _normalize_text_key(tail)
                if head_key and tail_key:
                    triple_relation_map[(head_key, tail_key)] = rel
                    if passage_text:
                        entity_passage_map.setdefault(head_key, passage_text)
                        entity_passage_map.setdefault(tail_key, passage_text)
                        edge_passage_map.setdefault((head_key, tail_key), passage_text)

        # ------------------------------------------------------------------ #
        # 4. Populate the graph container                                      #
        # ------------------------------------------------------------------ #
        graph = container_or_new(container)
        graph.nodes = {}
        graph.edges = []
        graph._adj = {}
        entity_text_to_ids: Dict[str, List[str]] = {}
        openie_seeded = False

        edge_key_to_index: Dict[Tuple[str, str, str], int] = {}
        fact_relation_by_node_ids: Dict[Tuple[str, str], str] = {}

        # ------------------------------------------------------------------ #
        # 4.5 Build nodes/edges directly from OpenIE triples                   #
        # ------------------------------------------------------------------ #
        if resolved_openie is not None:
            for head, rel, tail, passage_id, passage_text in _iter_openie_triples(resolved_openie):
                head_key = _normalize_text_key(head)
                tail_key = _normalize_text_key(tail)
                if not head_key or not tail_key:
                    continue

                source_id = _entity_hash_id(head_key)
                target_id = _entity_hash_id(tail_key)

                if source_id not in graph.nodes:
                    metadata: Dict[str, Any] = {
                        "inferred_from_openie": True,
                        "original_label": source_id,
                    }
                    if passage_id:
                        metadata["openie_passage_id"] = str(passage_id)
                    if passage_text:
                        metadata["passage"] = passage_text
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "hipporag"
                    graph.add_node(
                        NodeRecord(
                            id=source_id,
                            type="Entity",
                            text=head,
                            embedding=None,
                            metadata=metadata,
                        )
                    )
                    entity_text_to_ids.setdefault(head_key, []).append(source_id)

                if target_id not in graph.nodes:
                    metadata = {
                        "inferred_from_openie": True,
                        "original_label": target_id,
                    }
                    if passage_id:
                        metadata["openie_passage_id"] = str(passage_id)
                    if passage_text:
                        metadata["passage"] = passage_text
                    if keep_source_reference:
                        metadata["_source_path"] = str(src)
                        metadata["_source_style"] = "hipporag"
                    graph.add_node(
                        NodeRecord(
                            id=target_id,
                            type="Entity",
                            text=tail,
                            embedding=None,
                            metadata=metadata,
                        )
                    )
                    entity_text_to_ids.setdefault(tail_key, []).append(target_id)

                relation_text = str(rel).strip() if rel is not None else ""
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
                            "source": "openie",
                            "passage": passage_text,
                            "openie_passage_id": str(passage_id) if passage_id else None,
                        },
                    )
                )
                edge_key_to_index[edge_key] = len(graph.edges) - 1
                openie_seeded = True

        # Guard: if OpenIE produced no nodes, surface igraph vertices
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
        if ig_graph is not None and graph.nodes and not openie_only:
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
                    src_label = ""
                    tgt_label = ""
                    if src_id in graph.nodes:
                        src_label = graph.nodes[src_id].metadata.get("entity_name") or graph.nodes[src_id].text or ""
                    if tgt_id in graph.nodes:
                        tgt_label = graph.nodes[tgt_id].metadata.get("entity_name") or graph.nodes[tgt_id].text or ""
                    src_text_key = _normalize_text_key(src_label)
                    tgt_text_key = _normalize_text_key(tgt_label)
                    relation = triple_relation_map.get((src_text_key, tgt_text_key), "related")
                passage_text = None
                if src_id in graph.nodes and tgt_id in graph.nodes:
                    src_label = graph.nodes[src_id].metadata.get("entity_name") or ""
                    tgt_label = graph.nodes[tgt_id].metadata.get("entity_name") or ""
                    src_key = _normalize_text_key(src_label)
                    tgt_key = _normalize_text_key(tgt_label)
                    passage_text = edge_passage_map.get((src_key, tgt_key))

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
                        metadata={
                            "source": "igraph",
                            "passage": passage_text,
                        },
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
