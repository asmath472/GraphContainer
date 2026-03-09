from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Sequence

from GraphContainer import SearchableGraphContainer
from GraphContainer.adapters import import_graph_from_fastinsight
from GraphContainer.rag.retrievers.one_hop import OneHopRetriever


def one_hop_search(
    graph: SearchableGraphContainer,
    query_vector: Sequence[float],
    *,
    index_name: str,
    k: int = 5,
) -> Dict[str, Any]:
    retriever = OneHopRetriever()
    result = retriever.retrieve(
        graph=graph,
        query="",
        query_vector=query_vector,
        index_name=index_name,
        top_k=k,
    )
    seed_set = set(result.seed_nodes)
    return {
        "index_name": index_name,
        "seed_nodes": result.seed_nodes,
        "one_hop_nodes": [node.id for node in result.nodes if node.id not in seed_set],
        "one_hop_edges": result.edges,
        "context_chunks": result.context_chunks,
        "metadata": result.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    graph = import_graph_from_fastinsight(args.source)

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    query_vector = model.encode(args.query, return_dense=True)["dense_vecs"].tolist()

    result = one_hop_search(
        graph,
        query_vector,
        index_name="node_vector",
        k=args.k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
