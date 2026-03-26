#!/usr/bin/env python
"""
Entrypoint script for GraphContainer live visualizer.
Use this instead of `python -m src.visualizer.live_visualizer`
to avoid the module double-import issue.

Usage examples:
  python serve.py --format attribute_bundle_graph --source ./data/rag_storage/lightrag/bsard
  python serve.py --format topology_semantic_graph --source ./data/rag_storage/hipporag/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2
  python serve.py --format subgraph_union_graph --source ./data/rag_storage/g_retriever/scene_graphs
  python serve.py --format expla_graphs --source ./data/rag_storage/expla_graphs
  python serve.py --format freebasekg --source rmanluo/RoG-webqsp
  python serve.py --format component_graph --source ./data/rag_storage/fastinsight/scifact-bge-m3

  # Multiple formats at once
  python serve.py \
    --graph component_graph:./data/rag_storage/fastinsight/scifact-bge-m3 \
    --graph attribute_bundle_graph:./data/rag_storage/lightrag/bsard \
    --graph topology_semantic_graph:./data/rag_storage/hipporag/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2 \
    --graph subgraph_union_graph:./data/rag_storage/g_retriever/scene_graphs 
  
  python serve.py --graph component_graph:./data/rag_storage/fastinsight/scifact-bge-m3  --graph attribute_bundle_graph:./data/rag_storage/lightrag/bsard  --graph topology_semantic_graph:./data/rag_storage/hipporag/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2  --graph subgraph_union_graph:./data/rag_storage/g_retriever/expla_graphs
  python serve.py --graph expla_graphs:./data/rag_storage/g_retriever/expla_graphs
"""
from src.visualizer.live_visualizer import _main

_main()
