# Data Licenses And Provenance

This file tracks known provenance for the public graph artifacts distributed at https://huggingface.co/datasets/hchaejeong/graphcontainer-graphs.

GraphContainer source code is MIT licensed. That license covers this repository's source code; it does not automatically cover upstream source datasets, upstream graph construction systems, or derived graph artifacts.

| Artifact | Source data | Source-data terms | Construction system | Notes |
|---|---|---|---|---|
| `fastinsight/scifact-bge-m3` | SciFact, https://huggingface.co/datasets/allenai/scifact and https://github.com/allenai/scifact | CC BY-NC 2.0, according to the SciFact dataset card and license file | FastInsight / Component Graph; public builder repository and commit not recorded | Embedding model recorded in local manifest as `BAAI/bge-m3`; verified stored graph counts: 5,183 nodes and 51,830 edges. |
| `g_retriever/scene_graphs` | GQA Scene Graphs, https://cs.stanford.edu/people/dorarad/gqa/download.html | Requires upstream-license verification. GQA states that images are from COCO and Flickr and scene graphs are based on Visual Genome; users must verify applicable upstream terms. | G-Retriever, https://github.com/XiaoxinHe/G-Retriever | G-Retriever code is listed as MIT on GitHub. This artifact is stored as `nodes.tar`, `edges.tar`, `graphs.tar`, `questions.csv`, and `q_embs.pt`; tar extraction is required before GraphContainer import. |
| `hipporag/2wikimultihopqa` | 2WikiMultiHopQA, https://huggingface.co/datasets/xanhho/2WikiMultihopQA and https://github.com/Alab-NII/2wikimultihop | Apache-2.0 according to the Hugging Face dataset card | HippoRAG, https://github.com/OSU-NLP-Group/HippoRAG | HippoRAG code is listed as MIT on GitHub. Builder commit not recorded. Artifact path records `gpt-4o-mini` and `nvidia/NV-Embed-v2`. |
| `lightrag/bsard` | BSARD, https://huggingface.co/datasets/maastrichtlawtech/bsard | CC BY-NC-SA 4.0 according to the BSARD dataset card | LightRAG, https://github.com/HKUDS/LightRAG | LightRAG code is listed as MIT on GitHub. Builder commit and embedding model not recorded. |

## Verification Sources

- GraphContainer paper: https://arxiv.org/abs/2607.19362
- GraphContainer Hugging Face paper page: https://huggingface.co/papers/2607.19362
- GraphContainer graph artifacts: https://huggingface.co/datasets/hchaejeong/graphcontainer-graphs
- SciFact dataset card: https://huggingface.co/datasets/allenai/scifact
- 2WikiMultiHopQA dataset card: https://huggingface.co/datasets/xanhho/2WikiMultihopQA
- GQA download page: https://cs.stanford.edu/people/dorarad/gqa/download.html
- GQA overview page: https://cs.stanford.edu/people/dorarad/gqa/index.html
- BSARD dataset card: https://huggingface.co/datasets/maastrichtlawtech/bsard
- G-Retriever repository: https://github.com/XiaoxinHe/G-Retriever
- HippoRAG repository: https://github.com/OSU-NLP-Group/HippoRAG
- LightRAG repository: https://github.com/HKUDS/LightRAG

## Open Items

- Record exact graph-builder commits for all four artifacts.
- Confirm GQA-derived SceneGraphs redistribution terms across GQA, COCO/Flickr, and Visual Genome.
- Record node and edge counts for `g_retriever/scene_graphs`, `hipporag/2wikimultihopqa`, and `lightrag/bsard` if a future release computes them with the importers.
- Record the LightRAG embedding model if it can be recovered from build logs or upstream artifact metadata.
