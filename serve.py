#!/usr/bin/env python
"""
Entrypoint script for GraphContainer live visualizer.
Use this instead of `python -m src.visualizer.live_visualizer`
to avoid the module double-import issue.

Usage examples:
  python serve.py --format lightrag --source ./data/rag_storage/lightrag/bsard
  python serve.py --format hipporag --source ./data/rag_storage/hipporag/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2
  python serve.py --format g_retriever --source ./data/rag_storage/g_retriever/expla_graphs
  python serve.py --format tog --source ./data/rag_storage/tog/scifact
  python serve.py --format fastinsight --source ./data/rag_storage/fastinsight/scifact-bge-m3

  # Multiple formats at once
  python serve.py --graph fastinsight:./data/rag_storage/fastinsight/scifact-bge-m3  --graph lightrag:./data/rag_storage/lightrag/bsard  --graph hipporag:./data/rag_storage/hipporag/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2  --graph g_retriever:./data/rag_storage/g_retriever/expla_graphs
"""
from src.visualizer.live_visualizer import _main

_main()
