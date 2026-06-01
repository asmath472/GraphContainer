from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional

from ..core import SearchableGraphContainer
from .contracts import ChatRequest
from .embeddings import EmbeddingService
from .generator import OpenAIChatGenerator
from .pipeline import GraphRAGPipeline
from .retrievers import FastInsightRetriever, GARRetriever, HybridRetriever, OneHopRetriever, VectorRetriever


class GraphRAGService:
    def __init__(
        self,
        graph: SearchableGraphContainer,
        *,
        visualizer: Optional[Any] = None,
        pipeline: Optional[GraphRAGPipeline] = None,
        default_chat_model: str = "gpt-5-nano",
        default_embedding_provider: str = "hf",
        default_hf_embedding_model: str = "BAAI/bge-m3",
        default_openai_embedding_model: str = "text-embedding-3-small",
        hf_embedding_models: Optional[Iterable[str]] = None,
        openai_embedding_models: Optional[Iterable[str]] = None,
        embedding_model_catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.visualizer = visualizer

        if pipeline is None:
            embedding_service = EmbeddingService(
                default_provider=default_embedding_provider,
                default_hf_model=default_hf_embedding_model,
                default_openai_model=default_openai_embedding_model,
                hf_models=hf_embedding_models,
                openai_models=openai_embedding_models,
                model_catalog=embedding_model_catalog,
            )
            generator = OpenAIChatGenerator(
                default_model=default_chat_model,
            )
            pipeline = GraphRAGPipeline(
                embedding_service=embedding_service,
                generator=generator,
                retrievers={
                    "one-hop": OneHopRetriever(),
                    "vector": VectorRetriever(),
                    "hybrid": HybridRetriever(),
                    "fastinsight": FastInsightRetriever(),
                    "gar": GARRetriever(),
                },
                default_retrieval="one-hop",
                retrieval_aliases={
                    "graph-hop": "one-hop",
                    "graph-2-hop": "one-hop",
                    "graph": "one-hop",
                    "fi": "fastinsight",
                    "graph-adaptive-rerank": "gar",
                },
            )
        self.pipeline = pipeline

    def list_embedding_options(self) -> Dict[str, Any]:
        embedding_service = getattr(self.pipeline, "embedding_service", None)
        if embedding_service is not None and hasattr(embedding_service, "list_options"):
            payload = embedding_service.list_options()
            if isinstance(payload, dict):
                return payload
        return {
            "default_provider": "hf",
            "default_model": "BAAI/bge-m3",
            "default_value": "hf:BAAI/bge-m3",
            "providers": [],
            "options": [],
        }

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.graph, "search"):
            raise RuntimeError(
                "This graph does not support retrieval search. "
                "Load a SearchableGraphContainer with an attached index."
            )

        request = ChatRequest.from_payload(payload)

        session_id = request.session_id
        if self.visualizer is not None:
            if session_id is None:
                session_id = self.visualizer.create_session(
                        metadata={
                            "graph": request.graph,
                            "model": request.model,
                            "retrieval": request.retrieval,
                            "embedding_provider": request.embedding_provider,
                            "embedding_model": request.embedding_model,
                            "embedding_error_policy": request.embedding_error_policy,
                        }
                    )
            else:
                if not self.visualizer.has_session(session_id):
                    session_id = self.visualizer.create_session(
                        metadata={
                            "graph": request.graph,
                            "model": request.model,
                            "retrieval": request.retrieval,
                            "embedding_provider": request.embedding_provider,
                            "embedding_model": request.embedding_model,
                            "embedding_error_policy": request.embedding_error_policy,
                        }
                    )
                else:
                    self.visualizer.clear_session(session_id)

            self.visualizer.update_session(
                session_id,
                metadata={
                    "graph": request.graph,
                    "model": request.model,
                    "retrieval": request.retrieval,
                    "embedding_provider": request.embedding_provider,
                    "embedding_model": request.embedding_model,
                    "embedding_error_policy": request.embedding_error_policy,
                },
                progress={"message": "Starting graph retrieval"},
            )

        request = replace(request, session_id=session_id)
        response = self.pipeline.run(
            graph=self.graph,
            request=request,
            visualizer=self.visualizer,
        )
        if self.visualizer is not None and session_id:
            self.visualizer.update_session(
                session_id,
                metadata={
                    "graph": request.graph,
                    "model": request.model,
                    "retrieval": request.retrieval,
                    "embedding_provider": request.embedding_provider,
                    "embedding_model": request.embedding_model,
                    "embedding_error_policy": request.embedding_error_policy,
                    "llm_answer": response.answer,
                    "retrieval_elapsed_ms": response.metadata.get("retrieval_elapsed_ms"),
                    "generation_elapsed_ms": response.metadata.get("generation_elapsed_ms"),
                    "total_elapsed_ms": response.metadata.get("total_elapsed_ms"),
                },
                progress={"message": "Answer generated"},
            )

            # ── Oracle Evaluation & Visualization ─────────────────────────────
            oracle = getattr(self.visualizer, "oracle", None)
            if oracle is not None:
                query_text = request.message
                gold_nodes, gold_edges = oracle.resolve_with_edges(query_text)
                
                # File-based logging for robustness
                with open("oracle_debug.log", "a") as logf:
                    logf.write(f"Query: {query_text}\n")
                    logf.write(f"Resolved: {len(gold_nodes)} nodes, {len(gold_edges)} edges\n")
                    if gold_nodes:
                        logf.write(f"Sample gold: {gold_nodes[:5]}\n")
                
                print(f"[DEBUG] Oracle resolve for '{query_text}': {len(gold_nodes)} nodes, {len(gold_edges)} edges")

                
                # Use cached reverse mapping to map oracle labels to actual container node IDs
                oracle_to_node_id = self.visualizer.get_oracle_label_map()
                print(f"[DEBUG] Using oracle_to_node_id map with {len(oracle_to_node_id)} entries")

                mapped_gold_nodes = []
                for gn in gold_nodes:
                    sgn = str(gn).lower().strip()
                    if not sgn: continue
                    
                    # 1. Exact/Cached match
                    mid = oracle_to_node_id.get(sgn)
                    if mid:
                        mapped_gold_nodes.append(mid)
                        continue

                    # 2. Advanced Fuzzy match (token overlap & substring)
                    found_fuzzy = False
                    sgn_tokens = set(sgn.split())
                    
                    for label, nid in oracle_to_node_id.items():
                        l_lower = str(label).lower()
                        # Substring (either way)
                        if sgn in l_lower or l_lower in sgn:
                            mapped_gold_nodes.append(nid)
                            found_fuzzy = True
                            break
                        
                        # Token overlap (for names like "Mauro Scocco" vs "Scocco, Mauro")
                        label_tokens = set(l_lower.split())
                        if sgn_tokens and sgn_tokens.issubset(label_tokens):
                            mapped_gold_nodes.append(nid)
                            found_fuzzy = True
                            break
                        
                        # Simple date matching (e.g. "1962" in "11 September 1962")
                        if len(sgn) > 4 and sgn.isdigit() and sgn in l_lower:
                            mapped_gold_nodes.append(nid)
                            found_fuzzy = True
                            break
                    
                    if not found_fuzzy:
                        print(f"[DEBUG] Failed to map gold node: '{gn}' | Normal: '{sgn}'")
                
                mapped_gold_edges = []
                for s, t in gold_edges:
                    ms = oracle_to_node_id.get(str(s).lower())
                    mt = oracle_to_node_id.get(str(t).lower())
                    if ms and mt:
                        mapped_gold_edges.append((ms, mt))
                    else:
                        # Fallback for edges: sometimes oracle edges use IDs directly
                        ms = ms or str(s)
                        mt = mt or str(t)
                        mapped_gold_edges.append((ms, mt))



                retrieved_nids = {str(n.get("id")) for n in response.nodes}
                print(f"[DEBUG] Retrieved NIDs: {len(retrieved_nids)}")

                missed_nodes = [gn for gn in mapped_gold_nodes if str(gn) not in retrieved_nids]
                hit_nodes = [gn for gn in mapped_gold_nodes if str(gn) in retrieved_nids]

                retrieved_edge_pairs = set()
                for e in response.edges:
                    s = str(e.get("source"))
                    t = str(e.get("target"))
                    retrieved_edge_pairs.add((s, t))
                    retrieved_edge_pairs.add((t, s))

                missed_edges = [
                    (s, t) for s, t in mapped_gold_edges 
                    if (str(s), str(t)) not in retrieved_edge_pairs
                ]
                hit_edges = [
                    (s, t) for s, t in mapped_gold_edges 
                    if (str(s), str(t)) in retrieved_edge_pairs
                ]

                nodes_to_update = []
                edges_to_update = []

                purple_node = {"color": {"background": "#ce93d8", "border": "#8e24aa"}, "borderWidth": 2}
                purple_edge = {"color": "#ab47bc", "width": 3, "dashes": True}
                orange_node = {"color": {"background": "#ff9800", "border": "#e65100"}, "borderWidth": 4}
                orange_edge = {"color": "#ff9800", "width": 4}

                for gn in missed_nodes:
                    nodes_to_update.append({"id": str(gn), "style": purple_node})
                for gn in hit_nodes:
                    nodes_to_update.append({"id": str(gn), "style": orange_node})

                for s, t in missed_edges:
                    edges_to_update.append({"source": str(s), "target": str(t), "style": purple_edge})
                for s, t in hit_edges:
                    edges_to_update.append({"source": str(s), "target": str(t), "style": orange_edge})

                if nodes_to_update or edges_to_update:
                    msg = f"Oracle: {len(hit_nodes)} hit, {len(missed_nodes)} missed nodes"
                    self.visualizer.update_session(
                        session_id,
                        nodes=nodes_to_update,
                        edges=edges_to_update,
                        progress={"message": msg}
                    )

        return response.to_dict()
