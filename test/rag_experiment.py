from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.fastinsight import import_graph_from_fastinsight
from src.adapters.hipporag import import_graph_from_hipporag
from src.adapters.lightrag import import_graph_from_lightrag
from src.rag.embeddings import EmbeddingService
from src.rag.retrievers import FastInsightRetriever, OneHopRetriever


SYSTEM_PROMPT = """You are a graph RAG assistant. Answer strictly using the provided context."""
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
FASTINSIGHT_FINAL_TOP_K = 5


def resolve_path(raw_path: str, *base_roots: Path) -> Path:
    explicit = Path(raw_path)
    if explicit.is_absolute():
        if explicit.exists():
            return explicit
    else:
        for root in base_roots:
            candidate = (root / explicit).resolve()
            if candidate.exists():
                return candidate
        candidate = explicit.resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Path not found: {raw_path}")


def normalize_query(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None

    for key in ("query", "text", "question", "question_text", "prompt"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Fallbacks for unconventional formats.
    for value in item.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_queries(path: Path, limit: int = -1) -> List[str]:
    queries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            record = json.loads(raw)
            if not isinstance(record, dict):
                continue

            query = normalize_query(record)
            if query is None:
                continue
            queries.append(query)

            if limit > 0 and len(queries) >= limit:
                break

    if not queries:
        raise ValueError(f"No valid queries found in {path}")
    return queries


def build_prompt(query: str, context_chunks: Sequence[str], top_k: int = 10) -> str:
    chunks = [chunk for chunk in context_chunks if chunk][:top_k]
    context_text = "\n".join(chunks).strip()
    return (
        "### Instruction\n"
        "You are a highly precise analytical assistant. Answer the question strictly based ONLY on the provided Context below.\n\n"
        "### Strict Guidelines\n"
        "1. **Maximum Precision**: Be concise. Do not add conversational filler, polite phrases, or unnecessary explanations.\n"
        "2. **Keyword Preservation**: You MUST include specific keywords, proper nouns, numbers, and technical terms EXACTLY as they appear in the Context. Do not paraphrase key entities.\n"
        "3. **No External Knowledge**: Do not use any knowledge outside the provided Context.\n"
        "4. **Mandatory Citation**: You MUST cite the `title` of the source content used for the answer. Append the citation in the format `[Title]` immediately after the relevant information.\n\n"
        "### Context\n"
        f"{context_text}\n\n"
        f"### Question\n{query}\n\n"
        "### Answer"
    )


def cleanup_lightrag_chroma(graph_path: Path, graph: Any, *, verbose: bool = False) -> None:
    """Release LightRAG-related vector store artifacts when using lightrag graph."""
    index = graph.get_index("node_vector")
    persist_path = getattr(index, "persist_path", None)
    if not persist_path:
        persist_path = os.getenv("VECTOR_STORE_PATH", "./data/database/chroma_db")

    source_name = graph_path.name
    collection_names = [f"{source_name}_entities", f"{source_name}_relationships"]

    try:
        import chromadb
    except Exception:
        if verbose:
            print("[cleanup] chromadb import failed; skip cleanup.")
        return

    # Remove persisted collections first.
    try:
        client = chromadb.PersistentClient(path=str(Path(persist_path).resolve()))
        for collection_name in collection_names:
            try:
                client.delete_collection(collection_name)
            except Exception:
                # Collection may not exist if it was already removed.
                pass
    except Exception as exc:
        if verbose:
            print(f"[cleanup] Failed chromadb collection cleanup: {exc}")

    if verbose:
        collection_summary = ", ".join(collection_names)
        print(f"[cleanup] Deleted LightRAG collections: {collection_summary}")


def run_retrieval_and_generate(
    graph_label: str,
    graph: Any,
    retriever_name: str,
    retriever: Any,
    query: str,
    *,
    embedding_service: EmbeddingService,
    generator_model: str,
    generator_client: OpenAI,
    top_k: int,
    index_name: str,
    max_context_chunks: int,
    fastinsight_db_style: str = "fastinsight",
    fastinsight_extra: Optional[Dict[str, Any]] = None,
) -> str:
    retrieval_kwargs: Dict[str, Any] = {
        "top_k": top_k,
        "index_name": index_name,
        "embedding_service": embedding_service,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_error_policy": "raise",
    }

    if retriever_name == "fastinsight":
        retrieval_kwargs.update(
            {
                "seed_top_k": top_k,
                "diving_top_k": max(100, top_k),
                "final_top_k": FASTINSIGHT_FINAL_TOP_K,
                "database_construction_method": fastinsight_db_style,
            }
        )
        if fastinsight_extra:
            retrieval_kwargs.update(fastinsight_extra)

    result = retriever.retrieve(graph=graph, query=query, **retrieval_kwargs)
    context_chunks = [chunk for chunk in result.context_chunks if isinstance(chunk, str)]

    prompt = build_prompt(query, context_chunks=context_chunks, top_k=max_context_chunks)
    response = generator_client.chat.completions.create(
        model=generator_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return "No answer returned."
    return content


def get_graphs() -> List[Tuple[str, str, Any]]:
    graph_specs = [
        (
            "fastinsight",
            "./data/rag_storage/fastinsight/bsard-openai",
            import_graph_from_fastinsight,
        ),
        ("lightrag", "./data/rag_storage/lightrag/bsard", import_graph_from_lightrag),
        ("hipporag", "./data/rag_storage/hipporag/bsard", import_graph_from_hipporag),
    ]
    loaded_graphs: List[Tuple[str, str, Any]] = []
    for name, rel_path, loader in graph_specs:
        path = resolve_path(rel_path, PROJECT_ROOT, PROJECT_ROOT / "graph-rag-proj")
        loaded_graphs.append((name, str(path), loader(source=path)))

    return loaded_graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bsard")
    parser.add_argument("--query_path", default=None)
    parser.add_argument("--query_limit", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--index_name", default="node_vector")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--ollama_url", default="http://localhost:11434/v1")
    parser.add_argument("--ollama_model", default="gemma3:12b")
    parser.add_argument("--max_context_chunks", type=int, default=10)
    parser.add_argument(
        "--cleanup_lightrag_chroma",
        action="store_true",
        help="Delete LightRAG Chroma collection after experiment (and attempt to remove its folder).",
    )
    parser.add_argument(
        "--no_cleanup_lightrag_chroma",
        dest="cleanup_lightrag_chroma",
        action="store_false",
        help="Keep LightRAG Chroma data after experiment.",
    )
    parser.set_defaults(cleanup_lightrag_chroma=True)
    args = parser.parse_args()

    if args.query_path is None:
        dataset_query_path = f"./data/datasets/{args.dataset}/queries.jsonl"
    else:
        dataset_query_path = args.query_path

    query_path = resolve_path(
        dataset_query_path,
        PROJECT_ROOT,
        PROJECT_ROOT / "graph-rag-proj",
    )
    queries = load_queries(query_path, limit=args.query_limit)

    embedding_service = EmbeddingService(default_provider="openai", default_openai_model=EMBEDDING_MODEL)
    generator_client = OpenAI(base_url=args.ollama_url, api_key="ollama")

    output_root_arg = Path(args.output_dir)
    if output_root_arg.is_absolute():
        output_root = output_root_arg
    else:
        output_root = (PROJECT_ROOT / output_root_arg).resolve()
    output_root = output_root / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)

    graphs = get_graphs()
    for graph_name, graph_path, graph in graphs:
        graph_path_obj = Path(graph_path)
        try:
            for method in ("one-hop", "fastinsight"):
                output_path = output_root / f"{graph_name}_{method}.jsonl"
                query_results: List[Tuple[int, str, str]] = []
                db_style = "lightrag" if (method == "fastinsight" and graph_name == "lightrag") else "fastinsight"
                retriever = OneHopRetriever() if method == "one-hop" else FastInsightRetriever()

                for idx, query in enumerate(
                    tqdm(queries, desc=f"Processing queries for {graph_name}_{method}")
                ):
                    try:
                        answer = run_retrieval_and_generate(
                            graph_name,
                            graph,
                            method,
                            retriever,
                            query,
                            embedding_service=embedding_service,
                            generator_model=args.ollama_model,
                            generator_client=generator_client,
                            top_k=args.top_k,
                            index_name=args.index_name,
                            max_context_chunks=args.max_context_chunks,
                            fastinsight_db_style=db_style,
                        )
                    except Exception as exc:
                        answer = f"ERROR: {exc}"
                    query_results.append((idx, query, answer))

                with output_path.open("w", encoding="utf-8") as f:
                    for _, query, answer in query_results:
                        f.write(json.dumps({"query": query, "output": answer}, ensure_ascii=False) + "\n")
        finally:
            if graph_name == "lightrag" and args.cleanup_lightrag_chroma:
                cleanup_lightrag_chroma(graph_path_obj, graph, verbose=True)


if __name__ == "__main__":
    main()
