"""
oracle.py – Gold-node resolver for the GraphContainer live visualizer.

The style is NOT auto-detected; it is provided explicitly by the caller
based on the graph type the user selected during import:

  Graph type (import mode)        → Oracle style
  ─────────────────────────────────────────────────────────────────────
  Component Graph                 → STYLE_FASTINSIGHT  ("fastinsight")
  Attribute Bundle Graph          → STYLE_LIGHTRAG     ("lightrag")
  Topology-Semantic Graph         → STYLE_HIPPORAG     ("hipporag")
  Subgraph Union Graph            → STYLE_G_RETRIEVER  ("g_retriever")

---

FASTINSIGHT
  Required:
    dataset_dir/queries.jsonl  – must contain a ``gold_contents`` list per entry
  Resolution:  query text → gold_contents (direct node/entity IDs)

LIGHTRAG  (mirrors InfoRAG/src/evaluate.py::lightrag_style_retrieval_results)
  Required:
    dataset_dir/queries.jsonl
    dataset_dir/corpus.jsonl
    dataset_dir/qrels/train.tsv  (optional)
    dataset_dir/qrels/test.tsv   (optional)
    rag_storage_dir/kv_store_text_chunks.json
    rag_storage_dir/kv_store_full_entities.json
    rag_storage_dir/kv_store_full_relations.json
  Resolution:
    query text → query_id → corpus_ids (via qrels)
              → titles (via corpus.jsonl)
              → full_doc_ids (via kv_store_text_chunks)
              → entity names (via kv_store_full_entities)

HIPPORAG
  Required:
    dataset_dir/queries.jsonl         – ``_id`` field is the query_index
    hipporag_outputs_dir/gold_subgraphs.json
  Resolution:  query text → _id (query_index) → gold_entities

G_RETRIEVER  (Subgraph Union Graph / scene-graph datasets)
  Required:
    dataset_dir/questions.csv  – columns: q_id, image_id, question, answer, full_answer
    dataset_dir/nodes/{image_id}.csv  – columns: node_id, node_attr
    dataset_dir/edges/{image_id}.csv  – columns: src, edge_attr, dst
  Resolution:
    question text → image_id (via questions.csv)
                 → all node_attr values in nodes/{image_id}.csv (oracle nodes)
                 → all (src_attr, dst_attr) pairs in edges/{image_id}.csv (oracle edges)

  Note: Unlike other styles this resolver is lazy – it does NOT pre-load all
  node CSVs into memory at startup (100k questions × ~30 nodes each would be
  ~3M rows).  Instead it indexes question→image_id at startup and loads the
  per-image node/edge CSVs on demand, caching them for repeated queries.
"""

from __future__ import annotations

import re
import csv
import json
import uuid
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Increase CSV field size limit for large attributes (e.g. WebQSP, G-Retriever)
csv.field_size_limit(10 * 1024 * 1024)  # 10MB

# ---------------------------------------------------------------------------
# Style constants  (match the graph-type names used by the frontend)
# ---------------------------------------------------------------------------
STYLE_FASTINSIGHT = "fastinsight"   # Component Graph
STYLE_LIGHTRAG    = "lightrag"      # Attribute Bundle Graph
STYLE_HIPPORAG    = "hipporag"      # Topology-Semantic Graph
STYLE_G_RETRIEVER = "g_retriever"  # Subgraph Union Graph

VALID_STYLES = {STYLE_FASTINSIGHT, STYLE_LIGHTRAG, STYLE_HIPPORAG, STYLE_G_RETRIEVER}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)
    return records


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm(text: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return " ".join(str(text).strip().split())


# ---------------------------------------------------------------------------
# OracleResolver
# ---------------------------------------------------------------------------

class OracleResolver:
    """
    Resolves gold/oracle entity/node names for a given query string.

    Parameters
    ----------
    style:
        One of ``"fastinsight"``, ``"lightrag"``, ``"hipporag"``,
        ``"g_retriever"``.  Pass the value that corresponds to the
        graph type selected by the user during import.
    dataset_dir:
        Directory containing ``queries.jsonl`` (and for lightrag also
        ``corpus.jsonl`` + ``qrels/``).
    rag_storage_dir:
        *lightrag only* – directory with ``kv_store_text_chunks.json``,
        ``kv_store_full_entities.json``, ``kv_store_full_relations.json``.
    hipporag_outputs_dir:
        *hipporag only* – directory containing ``gold_subgraphs.json``.
    """

    def __init__(
        self,
        style: str,
        dataset_dir: str | Path,
        rag_storage_dir: Optional[str | Path] = None,
        hipporag_outputs_dir: Optional[str | Path] = None,
    ) -> None:
        if style not in VALID_STYLES:
            raise ValueError(
                f"Unknown oracle style '{style}'. "
                f"Must be one of: {sorted(VALID_STYLES)}"
            )
        self.style = style
        self.dataset_dir = Path(dataset_dir)
        self.rag_storage_dir = Path(rag_storage_dir) if rag_storage_dir else None
        self.hipporag_outputs_dir = Path(hipporag_outputs_dir) if hipporag_outputs_dir else None

        # G-Retriever paths (set during _load_g_retriever)
        self._gr_nodes_dir: Optional[Path] = None
        self._gr_edges_dir: Optional[Path] = None

        # Number of queries successfully indexed
        self.query_count: int = 0

        # ── Fastinsight ──────────────────────────────────────────────────────
        # norm_query_text → list[gold_node_id]
        self._fi_map: Dict[str, List[str]] = {}

        # ── LightRAG ─────────────────────────────────────────────────────────
        # norm_query_text → query_id (str)
        self._lr_text_to_qid: Dict[str, str] = {}
        # query_id → list[corpus_id]
        self._lr_qrel: Dict[str, List[Tuple[str, float]]] = {}
        # corpus_id → title string
        self._lr_corpus_title: Dict[str, str] = {}
        # title → full_doc_id
        self._lr_title_to_docid: Dict[str, str] = {}
        # full_doc_id → list[entity_name]
        self._lr_entities: Dict[str, List[str]] = {}
        # full_doc_id → list[tuple[src, tgt]]  (edges, for future use)
        self._lr_relations: Dict[str, List[Tuple[str, str]]] = {}
        # norm_query_text → list[full_doc_id] (fallback when qrels are missing)
        self._lr_fallback_q_to_docs: Dict[str, List[str]] = {}
        # norm_query_text → (nodes, edges)  (direct high-precision oracle)
        self._lr_direct_oracle: Dict[str, Tuple[List[str], List[Tuple[str, str]]]] = {}

        # ── HippoRAG ─────────────────────────────────────────────────────────
        # mapping norm(query_text) -> list of gold entities
        self._hp_text_to_gold_nodes: Dict[str, List[str]] = {}

        # ── G-Retriever ───────────────────────────────────────────────────────
        # norm_question_text → image_id (str, matches CSV filenames)
        self._gr_q_to_image: Dict[str, str] = {}
        # LRU-style cache: image_id → (list[node_attr], list[tuple[src_attr,dst_attr]])
        self._gr_cache: Dict[str, Tuple[List[str], List[Tuple[str, str]]]] = {}

        self._load()

    # =========================================================================
    # Loading
    # =========================================================================

    def _load(self) -> None:
        logger.info("OracleResolver: loading style=%s dataset_dir=%s", self.style, self.dataset_dir)
        if self.style == STYLE_FASTINSIGHT:
            self._load_fastinsight()
        elif self.style == STYLE_LIGHTRAG:
            self._load_lightrag()
        elif self.style == STYLE_HIPPORAG:
            self._load_hipporag()
        elif self.style == STYLE_G_RETRIEVER:
            self._load_g_retriever()

    # ── Fastinsight ──────────────────────────────────────────────────────────

    def _load_fastinsight(self) -> None:
        queries_path = self.dataset_dir / "queries.jsonl"
        if not queries_path.exists():
            raise FileNotFoundError(f"queries.jsonl not found: {queries_path}")

        text_to_qid = {}
        for rec in _load_jsonl(queries_path):
            text = _norm(rec.get("text", ""))
            gold = rec.get("gold_contents", [])
            if text and isinstance(gold, list) and gold:
                self._fi_map[text] = gold
            elif text:
                text_to_qid[text] = str(rec.get("_id", ""))

        if not self._fi_map and text_to_qid:
            # Fallback to qrels if gold_contents is missing (e.g., scifact)
            qrel_dir = self.dataset_dir / "qrels"
            print(f"[DEBUG] Oracle fallback: searching qrels in {qrel_dir}")
            qid_to_corpus = {}
            for tsv_name in ("train.tsv", "test.tsv"):
                tsv_path = qrel_dir / tsv_name
                if not tsv_path.exists():
                    print(f"[DEBUG] qrel file missing: {tsv_path}")
                    continue
                with tsv_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            qid, cid = parts[0], parts[1]
                            qid_to_corpus.setdefault(qid, []).append(cid)
            
            print(f"[DEBUG] Built qid_to_corpus map with {len(qid_to_corpus)} queries")
            for text, qid in text_to_qid.items():
                if qid in qid_to_corpus:
                    self._fi_map[text] = qid_to_corpus[qid]
        
        print(f"[DEBUG] FastInsight oracle loaded with {len(self._fi_map)} queries")

        self.query_count = len(self._fi_map)
        logger.info(
            "OracleResolver[fastinsight]: loaded %d queries from %s",
            self.query_count, queries_path,
        )

    # ── LightRAG ─────────────────────────────────────────────────────────────

    def _load_lightrag(self) -> None:
        if self.rag_storage_dir is None:
            raise ValueError("rag_storage_dir is required for lightrag style.")

        queries_path  = self.dataset_dir / "queries.jsonl"
        corpus_path   = self.dataset_dir / "corpus.jsonl"
        chunks_path   = self.rag_storage_dir / "kv_store_text_chunks.json"
        entities_path = self.rag_storage_dir / "kv_store_full_entities.json"
        relations_path = self.rag_storage_dir / "kv_store_full_relations.json"

        # 1. queries.jsonl → norm_text → query_id
        if not queries_path.exists():
            raise FileNotFoundError(f"queries.jsonl not found: {queries_path}")
        for rec in _load_jsonl(queries_path):
            qid  = str(rec.get("_id", ""))
            text = _norm(rec.get("text", ""))
            if qid and text:
                self._lr_text_to_qid[text] = qid
        self.query_count = len(self._lr_text_to_qid)

        # 2. qrels/ → query_id → list[corpus_id]
        #    Format (BEIR): query-id <TAB> corpus-id <TAB> score  (header on first line)
        qrel_dir = self.dataset_dir / "qrels"
        for tsv_name in ("train.tsv", "test.tsv"):
            tsv_path = qrel_dir / tsv_name
            if not tsv_path.exists():
                continue
            with tsv_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[1:]:          # skip header
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, cid, score = parts[0], parts[1], float(parts[2])
                    self._lr_qrel.setdefault(qid, []).append((cid, score))
                elif len(parts) >= 2:
                    qid, cid = parts[0], parts[1]
                    self._lr_qrel.setdefault(qid, []).append((cid, 1.0))
            
            # Sort by score descending
            for qid in self._lr_qrel:
                self._lr_qrel[qid].sort(key=lambda x: x[1], reverse=True)

        # 3. corpus.jsonl → corpus_id → title
        #    Mirrors evaluate.py: prefer data['title'], fallback to first line of 'text'
        if corpus_path.exists():
            for rec in _load_jsonl(corpus_path):
                cid = str(rec.get("_id", ""))
                if not cid:
                    continue
                if "title" in rec:
                    title = rec["title"]
                else:
                    title = rec.get("text", "").split("\n")[0]
                self._lr_corpus_title[cid] = title
        else:
            logger.warning("corpus.jsonl not found at %s", corpus_path)

        # 4. kv_store_text_chunks.json → title → full_doc_id
        #    Mirrors evaluate.py: first line of "content" field is the title
        if chunks_path.exists():
            chunks_data = _load_json(chunks_path)
            for _, doc in chunks_data.items():
                content = doc.get("content", "")
                title   = content.split("\n")[0] if content else ""
                doc_id  = doc.get("full_doc_id", "")
                if title and doc_id:
                    self._lr_title_to_docid[title] = doc_id
        else:
            logger.warning("kv_store_text_chunks.json not found at %s", chunks_path)

        # 5. kv_store_full_entities.json → full_doc_id → entity_names
        if entities_path.exists():
            for doc_id, info in _load_json(entities_path).items():
                self._lr_entities[doc_id] = info.get("entity_names", [])
        else:
            logger.warning("kv_store_full_entities.json not found at %s", entities_path)

        # 6. kv_store_full_relations.json → full_doc_id → relation_pairs
        if relations_path.exists():
            for doc_id, info in _load_json(relations_path).items():
                pairs = []
                for rel in info.get("relation_pairs", []):
                    if len(rel) == 2:
                        pairs.append((rel[0], rel[1]))
                self._lr_relations[doc_id] = pairs
        else:
            logger.warning("kv_store_full_relations.json not found at %s", relations_path)

        # 7. Fallback for datasets without qrels (e.g. 2wikimultihopqa, hotpotqa, musique)
        if not self._lr_qrel:
            dataset_name = self.dataset_dir.name
            dataset_json_path = Path(f"/mnt/disk2/cjhyun/HippoRAG/reproduce/dataset/{dataset_name}.json")
            if not dataset_json_path.exists():
                dataset_json_path = self.dataset_dir / f"{dataset_name}.json"
            if not dataset_json_path.exists():
                dataset_json_path = self.dataset_dir / f"{dataset_name}_test.json"
            if not dataset_json_path.exists():
                alt_path = Path(f"/mnt/disk2/cjhyun/InfoRAG/data/datasets/{dataset_name}/{dataset_name}.json")
                if alt_path.exists(): dataset_json_path = alt_path
            
            logger.debug("OracleResolver[lightrag]: Attempting fallback to %s", dataset_json_path)
            if dataset_json_path.exists():
                try:
                    samples = _load_json(dataset_json_path)
                    logger.debug("OracleResolver[lightrag]: Loaded %d samples from %s", len(samples), dataset_json_path)
                    for sample in samples:
                        q_raw = sample.get('question', '')
                        q = _norm(q_raw)
                        if not q: continue
                        
                        # 1. High-precision evidence triples + Answers
                        ev_nodes = set()
                        ev_edges = []
                        
                        def add_node(n: str):
                            if not n: return
                            n = n.strip()
                            # Filter out single-character noise (like "B") and fragments
                            if len(n) < 2 and not n.isdigit(): return
                            ev_nodes.add(n)

                        ans_raw = sample.get('answer', '')
                        if ans_raw:
                            add_node(ans_raw)
                        
                        for alt in sample.get('answer_aliases', []):
                            add_node(alt)

                        if 'evidences' in sample:
                            for triple in sample['evidences']:
                                if len(triple) >= 3:
                                    add_node(triple[0])
                                    add_node(triple[2])
                                    ev_edges.append((triple[0], triple[2]))
                        
                        # Musique/2Wiki: Chain the decomposition steps into a path
                        if 'question_decomposition' in sample:
                            steps = sample['question_decomposition']
                            for i in range(len(steps)):
                                step_ans = steps[i].get('answer')
                                if not step_ans: continue
                                add_node(step_ans)
                                # Link to next step answer to form a path
                                if i < len(steps) - 1:
                                    next_ans = steps[i+1].get('answer')
                                    if next_ans:
                                        ev_edges.append((step_ans, next_ans))
                                    
                        if ev_nodes:
                            self._lr_direct_oracle[q] = (list(ev_nodes), ev_edges)
                            self._lr_direct_oracle[q_raw.strip()] = (list(ev_nodes), ev_edges)

                        # 2. Document-level fallback
                        gold_doc_ids = set()
                        def find_doc_id(gt_title: str) -> Optional[str]:
                            if gt_title in self._lr_title_to_docid:
                                return self._lr_title_to_docid[gt_title]
                            for stored_title, doc_id in self._lr_title_to_docid.items():
                                if stored_title.startswith(gt_title) or gt_title in stored_title:
                                    return doc_id
                            return None

                        if 'supporting_facts' in sample:
                            for gt in set(item[0] for item in sample['supporting_facts']):
                                did = find_doc_id(gt)
                                if did: gold_doc_ids.add(did)
                        elif 'contexts' in sample:
                            for item in sample['contexts']:
                                if item.get('is_supporting'):
                                    did = find_doc_id(item['title'])
                                    if did: gold_doc_ids.add(did)
                        elif 'paragraphs' in sample:
                            for item in sample['paragraphs']:
                                if item.get('is_supporting'):
                                    did = find_doc_id(item['title'])
                                    if did: gold_doc_ids.add(did)
                        
                        if gold_doc_ids:
                            self._lr_fallback_q_to_docs[q] = list(gold_doc_ids)
                except Exception as e:
                    logger.error("OracleResolver[lightrag]: Failed to load fallback JSON %s: %s", dataset_json_path, e)

        print(f"[DEBUG] LightRAG oracle loaded with {len(self._lr_direct_oracle)} high-precision and {len(self._lr_fallback_q_to_docs)} document fallbacks")
        logger.info(
            "OracleResolver[lightrag]: %d queries | %d qrel entries | %d fallback | %d corpus | "
            "%d chunks | %d entity docs | %d relation docs",
            self.query_count,
            sum(len(v) for v in self._lr_qrel.values()),
            len(self._lr_fallback_q_to_docs),
            len(self._lr_corpus_title),
            len(self._lr_title_to_docid),
            len(self._lr_entities),
            len(self._lr_relations),
        )

    # ── HippoRAG ─────────────────────────────────────────────────────────────

    def _load_hipporag(self) -> None:
        if self.hipporag_outputs_dir is None:
            raise ValueError("hipporag_outputs_dir is required for hipporag style.")

        dataset_name = self.dataset_dir.name
        
        # Look for the dataset JSON file that contains the gold facts
        import glob
        # Try to find the dataset json in InfoRAG/data/datasets or HippoRAG/reproduce/dataset
        # The user's system stores the HippoRAG repo parallel to GraphContainer
        dataset_json_path = Path(f"/mnt/disk2/cjhyun/HippoRAG/reproduce/dataset/{dataset_name}.json")
        if not dataset_json_path.exists():
            # Fallback if not there
            dataset_json_path = self.dataset_dir / f"{dataset_name}.json"
        
        if not dataset_json_path.exists():
            raise FileNotFoundError(f"Could not find dataset json for {dataset_name} at {dataset_json_path}")

        openie_matches = glob.glob(str(self.hipporag_outputs_dir / "openie_results_ner_*.json"))
        if not openie_matches:
            openie_matches = glob.glob(str(self.hipporag_outputs_dir.parent / "openie_results_ner_*.json"))
        if not openie_matches:
            raise FileNotFoundError(f"openie_results_ner_*.json not found in or above {self.hipporag_outputs_dir}")
        openie_path = Path(openie_matches[-1])

        # 1. Load openie_results mapping norm(passage) -> list[entity]
        norm_passage_to_entities = {}
        for doc in _load_json(openie_path).get("docs", []):
            passage = doc.get("passage", "")
            if not passage: continue
            norm_p = " ".join(passage.split())
            
            entities = []
            for e in doc.get("extracted_entities", []):
                if e and str(e).strip():
                    entities.append(str(e))
            if norm_p:
                norm_passage_to_entities[norm_p] = entities

        # 2. Parse dataset json to extract gold passages per query, then map to entities
        samples = _load_json(dataset_json_path)
        for sample in samples:
            question = _norm(sample.get('question', ''))
            if not question:
                continue

            gold_doc = []
            if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
                gold_title = set([item[0] for item in sample['supporting_facts']])
                gold_title_and_content_list = [item for item in sample.get('context', []) if item[0] in gold_title]
                if dataset_name.startswith('hotpotqa'):
                    gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
                else:
                    gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
            elif 'contexts' in sample:
                gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item.get('is_supporting')]
            elif 'paragraphs' in sample:
                gold_paragraphs = []
                for item in sample['paragraphs']:
                    if item.get('is_supporting') is False:
                        continue
                    gold_paragraphs.append(item)
                gold_doc = [item['title'] + '\n' + (item.get('text', item.get('paragraph_text', ''))) for item in gold_paragraphs]

            gold_doc = list(set(gold_doc))
            
            gold_nodes = set()
            for doc_text in gold_doc:
                norm_doc = " ".join(doc_text.split())
                for ent in norm_passage_to_entities.get(norm_doc, []):
                    gold_nodes.add(ent)
                    
            if gold_nodes:
                self._hp_text_to_gold_nodes[question] = list(gold_nodes)

        self.query_count = len(self._hp_text_to_gold_nodes)
        logger.info(
            "OracleResolver[hipporag]: %d queries dynamically loaded from %s",
            self.query_count, dataset_json_path
        )

    # ── G-Retriever ──────────────────────────────────────────────────────────

    def _load_g_retriever(self) -> None:
        """
        Index questions.csv or train_dev.tsv at startup (question text → image_id).
        Per-image node/edge CSVs are loaded on-demand in ``_resolve_g_retriever``.
        """
        questions_path = self.dataset_dir / "questions.csv"
        train_dev_path = self.dataset_dir / "train_dev.tsv"

        nodes_dir = self.dataset_dir / "nodes"
        edges_dir = self.dataset_dir / "edges"
        if not nodes_dir.is_dir():
            raise FileNotFoundError(f"nodes/ directory not found under {self.dataset_dir}")
        if not edges_dir.is_dir():
            raise FileNotFoundError(f"edges/ directory not found under {self.dataset_dir}")

        self._gr_nodes_dir = nodes_dir
        self._gr_edges_dir = edges_dir

        if questions_path.exists():
            with questions_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = _norm(row.get("question", ""))
                    image_id = str(row.get("image_id", "")).strip()
                    if question and image_id:
                        if question not in self._gr_q_to_image:
                            self._gr_q_to_image[question] = image_id
        elif train_dev_path.exists():
            # Support ExplaGraphs which uses train_dev.tsv without a header
            with train_dev_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                for row_idx, row in enumerate(reader):
                    if len(row) >= 2:
                        q_full = _norm(f"{row[0]} {row[1]}")
                        image_id = str(row_idx)
                        if q_full and q_full not in self._gr_q_to_image:
                            self._gr_q_to_image[q_full] = image_id
        else:
            raise FileNotFoundError(f"Neither questions.csv nor train_dev.tsv found under {self.dataset_dir}")

        self.query_count = len(self._gr_q_to_image)
        logger.info(
            "OracleResolver[g_retriever]: indexed %d questions from %s",
            self.query_count, self.dataset_dir,
        )

    def _load_gr_image(self, image_id: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load and cache ``nodes/{image_id}.csv`` + ``edges/{image_id}.csv``.

        Returns (node_attr_list, [(src_attr, dst_attr), ...]).
        """
        if image_id in self._gr_cache:
            return self._gr_cache[image_id]

        # ─ nodes ────────────────────────────────────────────────────────────────
        nodes: List[str] = []
        node_csv = self._gr_nodes_dir / f"{image_id}.csv"  # type: ignore[operator]
        if node_csv.exists():
            with node_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    attr = (row.get("node_attr") or "").strip()
                    if attr:
                        nodes.append(attr)
        else:
            logger.warning("OracleResolver[g_retriever]: nodes/%s.csv not found", image_id)

        # ─ edges ───────────────────────────────────────────────────────────────
        # Edge endpoints are integer node_ids; map them back to node_attr strings
        # using the node list we just built (indexed by position == node_id).
        node_by_id: Dict[str, str] = {}
        if node_csv.exists():
            with node_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    nid  = str(row.get("node_id", "")).strip()
                    attr = (row.get("node_attr") or "").strip()
                    if nid and attr:
                        node_by_id[nid] = attr

        edges: List[Tuple[str, str]] = []
        edge_csv = self._gr_edges_dir / f"{image_id}.csv"  # type: ignore[operator]
        if edge_csv.exists():
            with edge_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    src_id = str(row.get("src", "")).strip()
                    dst_id = str(row.get("dst", "")).strip()
                    src_attr = node_by_id.get(src_id)
                    dst_attr = node_by_id.get(dst_id)
                    if src_attr and dst_attr:
                        edges.append((src_attr, dst_attr))
        else:
            logger.warning("OracleResolver[g_retriever]: edges/%s.csv not found", image_id)

        result = (nodes, edges)
        self._gr_cache[image_id] = result
        return result

    def resolve(self, query_text: str) -> List[str]:
        """
        Return the list of gold entity/node names for *query_text*.

        Lookup strategy: exact normalised-text match → substring containment
        fallback.  Returns ``[]`` when no match is found.
        """
        norm = _norm(query_text)
        if self.style == STYLE_FASTINSIGHT:
            return self._resolve_fastinsight(norm)
        if self.style == STYLE_LIGHTRAG:
            return self._resolve_lightrag(norm)
        if self.style == STYLE_HIPPORAG:
            return self._resolve_hipporag(norm)
        if self.style == STYLE_G_RETRIEVER:
            nodes, _ = self._resolve_g_retriever(norm)
            return nodes
        return []

    def resolve_with_edges(self, query_text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Like ``resolve`` but also returns gold edges as ``(src_attr, dst_attr)`` pairs.

        Edges are available for:
          - ``lightrag``   – from ``kv_store_full_relations.json``
          - ``g_retriever`` – from ``edges/{image_id}.csv``

        Other styles return ``(nodes, [])``.
        """
        norm = _norm(query_text)
        if self.style == STYLE_LIGHTRAG:
            return self._resolve_lightrag_with_edges(norm)
        if self.style == STYLE_G_RETRIEVER:
            return self._resolve_g_retriever(norm)
        nodes = self.resolve(query_text)
        return nodes, []

    def status(self) -> dict:
        """Return a JSON-serialisable status dict for the /api/oracle endpoint."""
        return {
            "loaded": bool(self.style),
            "style": self.style,
            "query_count": self.query_count,
            "dataset_dir": str(self.dataset_dir),
            "rag_storage_dir": str(self.rag_storage_dir) if self.rag_storage_dir else None,
            "hipporag_outputs_dir": str(self.hipporag_outputs_dir) if self.hipporag_outputs_dir else None,
        }

    # =========================================================================
    # Private resolution
    # =========================================================================

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fuzzy_match(norm_query: str, mapping: dict):
        """Return the value for an exact key match, then substring match, else None."""
        if norm_query in mapping:
            return mapping[norm_query]
        for key, val in mapping.items():
            if norm_query in key or key in norm_query:
                logger.debug("Oracle fuzzy match: '%s' → '%s'", norm_query, key)
                return val
        return None

    # ── Fastinsight ──────────────────────────────────────────────────────────

    def _resolve_fastinsight(self, norm: str) -> List[str]:
        result = self._fuzzy_match(norm, self._fi_map)
        if result is None:
            logger.debug("OracleResolver[fastinsight]: no match for '%s'", norm)
            return []
        return list(result)

    # ── LightRAG ─────────────────────────────────────────────────────────────

    def _corpus_ids_to_entities_and_edges(
        self, qid: str, query_text: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Mirrors evaluate.py::corpus_ids_to_graph but filters entities based on 
        lexical relevance to the query to avoid hallucinated/unrelated massive nodes.
        """
        seen_nodes: Set[str] = set()
        seen_edges: Set[Tuple[str, str]] = set()
        nodes: List[str] = []
        edges: List[Tuple[str, str]] = []

        # 1. Basic relevance filter function
        def is_relevant(entity_name: str, query: str) -> bool:
            # Rule 1: Reject massive hallucinated paragraphs (e.g., > 30 words)
            # This fixes the "Judge" paragraph bug.
            if len(entity_name.split()) > 30:
                return False
                
            # Rule 2: Require keyword overlap
            query_words = set(re.findall(r'\w+', query.lower()))
            entity_words = set(re.findall(r'\w+', entity_name.lower()))
            query_keywords = {w for w in query_words if len(w) > 3}
            if not query_keywords: return True
            return len(query_keywords.intersection(entity_words)) > 0

        # corpus_id → title → full_doc_id → entities / relations
        # Priority 1: Standard qrels (if available)
        # Cap at top 5 documents to avoid entity explosion in datasets like NFCorpus
        full_doc_ids: Set[str] = set()
        qrel_entries = self._lr_qrel.get(qid, [])
        for cid, score in qrel_entries[:5]:
            title = self._lr_corpus_title.get(cid)
            if title and title in self._lr_title_to_docid:
                doc_id = self._lr_title_to_docid[title]
                full_doc_ids.add(doc_id)

        for doc_id in full_doc_ids:
            # Filter Nodes
            for name in self._lr_entities.get(doc_id, []):
                # ★ APPLY FILTER HERE ★
                if name not in seen_nodes and is_relevant(name, query_text):
                    nodes.append(name)
                    seen_nodes.add(name)
                    
            # Filter Edges (Only keep edges where AT LEAST ONE node is relevant)
            for src, tgt in self._lr_relations.get(doc_id, []):
                if (src, tgt) not in seen_edges:
                    if src in seen_nodes or tgt in seen_nodes:
                        edges.append((src, tgt))
                        seen_edges.add((src, tgt))

        return nodes, edges

    def _doc_ids_to_entities_and_edges(
        self, doc_ids: List[str], query_text: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        seen_nodes: Set[str] = set()
        seen_edges: Set[Tuple[str, str]] = set()
        nodes: List[str] = []
        edges: List[Tuple[str, str]] = []

        def is_relevant(entity_name: str, query: str) -> bool:
            name_lower = entity_name.lower()
            if len(name_lower.split()) > 20:
                return False
            
            # For large document sets (like NFCorpus), only include entities 
            # that have some lexical overlap with the query to reduce noise.
            query_words = set(re.findall(r'\w+', query.lower()))
            entity_words = set(re.findall(r'\w+', name_lower))
            # Ignore stop words or very short words for overlap check
            query_keywords = {w for w in query_words if len(w) > 3}
            if not query_keywords: return True # Fallback if query is too short
            
            overlap = query_keywords.intersection(entity_words)
            return len(overlap) > 0

        for doc_id in set(doc_ids):
            for name in self._lr_entities.get(doc_id, []):
                if name not in seen_nodes and is_relevant(name, query_text):
                    nodes.append(name)
                    seen_nodes.add(name)
            for src, tgt in self._lr_relations.get(doc_id, []):
                if (src, tgt) not in seen_edges:
                    if src in seen_nodes or tgt in seen_nodes:
                        edges.append((src, tgt))
                        seen_edges.add((src, tgt))

        return nodes, edges
    
    # def _corpus_ids_to_entities_and_edges(
    #     self, corpus_ids: List[str]
    # ) -> Tuple[List[str], List[Tuple[str, str]]]:
    #     """
    #     Mirrors evaluate.py::corpus_ids_to_graph but returns lists instead of sets
    #     so that the caller can preserve insertion order if desired.
    #     """
    #     seen_nodes: Set[str] = set()
    #     seen_edges: Set[Tuple[str, str]] = set()
    #     nodes: List[str] = []
    #     edges: List[Tuple[str, str]] = []

    #     # corpus_id → title → full_doc_id → entities / relations
    #     full_doc_ids: Set[str] = set()
    #     for cid in corpus_ids:
    #         title = self._lr_corpus_title.get(cid)
    #         if title is None:
    #             logger.debug("OracleResolver[lightrag]: unknown corpus_id=%s", cid)
    #             continue
    #         doc_id = self._lr_title_to_docid.get(title)
    #         if doc_id is None:
    #             logger.debug(
    #                 "OracleResolver[lightrag]: no doc_id for title='%s' (corpus_id=%s)",
    #                 title, cid,
    #             )
    #             continue
    #         full_doc_ids.add(doc_id)

    #     for doc_id in full_doc_ids:
    #         for name in self._lr_entities.get(doc_id, []):
    #             if name not in seen_nodes:
    #                 nodes.append(name)
    #                 seen_nodes.add(name)
    #         for src, tgt in self._lr_relations.get(doc_id, []):
    #             if (src, tgt) not in seen_edges:
    #                 edges.append((src, tgt))
    #                 seen_edges.add((src, tgt))

    #     return nodes, edges

    def _resolve_lightrag(self, norm: str) -> List[str]:
        nodes, _ = self._resolve_lightrag_with_edges(norm)
        return nodes

    def _resolve_lightrag_with_edges(
        self, norm: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        # Priority 0: Direct high-precision oracle (from 'evidences' field)
        # Try exact, then normalized, then fuzzy
        direct = self._lr_direct_oracle.get(norm)
        if direct is None:
            norm_q = _norm(norm)
            direct = self._lr_direct_oracle.get(norm_q)
        
        if direct is None:
            # Last resort: fuzzy match query text
            best_q = self._fuzzy_match(norm, self._lr_direct_oracle)
            if best_q:
                direct = self._lr_direct_oracle[best_q]

        if direct is not None:
            return direct

        # Fallback for datasets lacking qrels (e.g. 2wikimultihopqa)
        if not self._lr_qrel and norm in self._lr_fallback_q_to_docs:
            return self._doc_ids_to_entities_and_edges(self._lr_fallback_q_to_docs[norm], norm)

        # 1. query text → query_id
        qid = self._fuzzy_match(norm, self._lr_text_to_qid)
        if qid is None:
            logger.debug("OracleResolver[lightrag]: no query_id for '%s'", norm)
            return [], []

        # 2. query_id → corpus_ids
        corpus_ids = self._lr_qrel.get(qid, [])
        if not corpus_ids:
            logger.debug("OracleResolver[lightrag]: no qrels for query_id=%s", qid)
            return [], []

        # 3. corpus_ids → entities + edges
        return self._corpus_ids_to_entities_and_edges(qid, norm)

    # ── HippoRAG ─────────────────────────────────────────────────────────────

    def _resolve_hipporag(self, norm: str) -> List[str]:
        gold_nodes = self._fuzzy_match(norm, self._hp_text_to_gold_nodes)
        if gold_nodes is None:
            logger.debug("OracleResolver[hipporag]: no gold nodes found for query '%s'", norm)
            return []
        
        return gold_nodes

    # ── G-Retriever ──────────────────────────────────────────────────────────

    def _resolve_g_retriever(self, norm: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        image_id = self._fuzzy_match(norm, self._gr_q_to_image)
        if image_id is None:
            logger.debug("OracleResolver[g_retriever]: no image_id for query '%s'", norm)
            return [], []
        
        return self._load_gr_image(image_id)
