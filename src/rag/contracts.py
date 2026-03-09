from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str

    @classmethod
    def from_payload(cls, item: Dict[str, Any]) -> "ChatMessage":
        role = str(item.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(item.get("content", item.get("text", ""))).strip()
        return cls(role=role, content=content)


@dataclass
class ChatRequest:
    message: str
    history: List[ChatMessage] = field(default_factory=list)
    graph: str = "default"
    model: str = "gpt-5-mini"
    retrieval: str = "one-hop"
    embedding_provider: str = "bge"
    embedding_model: str = "BAAI/bge-m3"
    top_k: int = 5
    index_name: str = "node_vector"
    session_id: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ChatRequest":
        if not isinstance(payload, dict):
            raise ValueError("Chat request payload must be a JSON object.")

        raw_message = payload.get("message", payload.get("query"))
        message = str(raw_message or "").strip()
        if not message:
            raise ValueError("`message` is required.")

        history_payload = payload.get("history", [])
        history: List[ChatMessage] = []
        if isinstance(history_payload, list):
            for item in history_payload:
                if isinstance(item, dict):
                    msg = ChatMessage.from_payload(item)
                    if msg.content:
                        history.append(msg)

        raw_top_k = payload.get("top_k", 5)
        try:
            top_k = max(1, int(raw_top_k))
        except (TypeError, ValueError):
            top_k = 5

        session_id = payload.get("session_id")
        if session_id is not None:
            session_id = str(session_id).strip() or None

        embedding_provider = str(payload.get("embedding_provider", payload.get("embedder", "bge"))).strip().lower()
        embedding_model = str(payload.get("embedding_model", "BAAI/bge-m3")).strip()

        embedding_payload = payload.get("embedding")
        if isinstance(embedding_payload, dict):
            embedding_provider = str(embedding_payload.get("provider", embedding_provider)).strip().lower()
            embedding_model = str(embedding_payload.get("model", embedding_model)).strip()
        elif isinstance(embedding_payload, str):
            raw = embedding_payload.strip()
            if ":" in raw:
                provider_part, model_part = raw.split(":", 1)
                embedding_provider = provider_part.strip().lower() or embedding_provider
                embedding_model = model_part.strip() or embedding_model
            elif raw:
                embedding_provider = raw.lower()

        return cls(
            message=message,
            history=history,
            graph=str(payload.get("graph", "default")),
            model=str(payload.get("model", "gpt-5-mini")),
            retrieval=str(payload.get("retrieval", "one-hop")),
            embedding_provider=embedding_provider or "bge",
            embedding_model=embedding_model or "BAAI/bge-m3",
            top_k=top_k,
            index_name=str(payload.get("index_name", "node_vector")),
            session_id=session_id,
        )


@dataclass
class RetrievedNode:
    id: str
    text: str = ""
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    method: str
    query: str
    seed_nodes: List[str] = field(default_factory=list)
    nodes: List[RetrievedNode] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    context_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "query": self.query,
            "seed_nodes": self.seed_nodes,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
            "context_chunks": self.context_chunks,
            "metadata": self.metadata,
        }


@dataclass
class ChatResponse:
    answer: str
    retrieval: str
    session_id: Optional[str]
    context_chunks: List[str] = field(default_factory=list)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieval": self.retrieval,
            "session_id": self.session_id,
            "context_chunks": self.context_chunks,
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
        }
