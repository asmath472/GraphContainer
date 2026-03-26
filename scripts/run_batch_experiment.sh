#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

python test/rag_experiment.py \
  --dataset bsard \
  --query_limit -1 \
  --top_k 10 \
  --index_name node_vector \
  --output_dir ./output/bsard \
  --ollama_url http://localhost:11434/v1 \
  --ollama_model gemma3:12b \
  --max_context_chunks 10
