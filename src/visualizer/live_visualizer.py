from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import cgi
import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import parse_qs, urlparse

from ..core import SimpleGraphContainer


def _edge_overlay_key(source: str, target: str, relation: str = "RELATED") -> str:
    return f"{source}|||{target}|||{relation}"


def _default_node_style() -> Dict[str, Any]:
    return {
        "color": {"background": "#ffd54f", "border": "#ff6f00"},
        "borderWidth": 3,
    }


def _default_edge_style() -> Dict[str, Any]:
    # Keep edge color from base graph rendering; session overlay controls width only.
    return {"width": 3}


_CURRENT_STEP_EDGE_COLOR = {"color": "#1565c0", "highlight": "#1565c0"}
_PREVIOUS_STEP_EDGE_COLOR = {"color": "#4caf50", "highlight": "#4caf50"}


def _is_current_step_edge_style(style: Dict[str, Any]) -> bool:
    color = style.get("color")
    if not isinstance(color, dict):
        return False
    return str(color.get("color", "")).strip().lower() == "#1565c0"


def _node_retrieval_stage(style: Dict[str, Any]) -> str:
    if not isinstance(style, dict):
        return ""
    color = style.get("color")
    if not isinstance(color, dict):
        return ""
    border = str(color.get("border", "")).strip().lower()
    background = str(color.get("background", "")).strip().lower()
    if border == "#1565c0" or background == "#bbdefb":
        return "current"
    if border == "#4caf50" or background == "#c8e6c9":
        return "previous"
    return ""


@dataclass
class _SessionState:
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    progress: Dict[str, Any] = field(
        default_factory=lambda: {"current": 0, "total": 0, "percent": 0.0, "message": ""}
    )
    replay_snapshots: List[Dict[str, Any]] = field(default_factory=list)


class LiveGraphVisualizer:
    """
    Backend service for session-based graph visualization.

    Frontend static files are served from `visualizer/web`.
    """

    def __init__(
        self,
        container: Optional[SimpleGraphContainer] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        poll_interval_ms: int = 600,
        default_hops: int = 2,
    ):
        self.host = host
        self.port = int(port)
        self.poll_interval_ms = int(max(200, poll_interval_ms))
        self.default_hops = int(max(0, default_hops))

        # Named graph registry: name -> {"label": str, "container": SimpleGraphContainer}
        self._graphs: Dict[str, Dict[str, Any]] = {}
        self._active_graph_name: str = ""

        self._lock = threading.RLock()
        self._session_cv = threading.Condition(self._lock)
        self._sessions: Dict[str, _SessionState] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._chat_service: Optional[Any] = None

        self._adj_undirected: Dict[str, Set[str]] = {}
        self._incident_edges: Dict[str, List[int]] = {}

        if container is not None:
            self.register_graph("default", container, label="Default")

        self._web_dir = Path(__file__).resolve().parent / "web"

    # ------------------------------------------------------------------
    # Named-graph registry
    # ------------------------------------------------------------------

    def register_graph(
        self,
        name: str,
        container: "SimpleGraphContainer",
        *,
        label: str = "",
        graph_type: str = "",
    ) -> None:
        """Register a named graph. The first registered graph becomes active."""
        with self._lock:
            self._graphs[name] = {
                "label": label or name,
                "container": container,
                "graph_type": graph_type or "",
            }
            if not self._active_graph_name:
                self._active_graph_name = name
                self._build_topology_indexes()
            elif name == self._active_graph_name:
                # Replacing the active graph (e.g., lazy load) should refresh topology.
                self._build_topology_indexes()

    @property
    def container(self) -> "SimpleGraphContainer":
        """Return the currently active graph container."""
        if not self._active_graph_name or self._active_graph_name not in self._graphs:
            raise RuntimeError("No graph has been registered. Call register_graph() first.")
        return self._graphs[self._active_graph_name]["container"]

    def list_graphs(self) -> List[Dict[str, str]]:
        """Return [{name, label, active}] for all registered graphs."""
        with self._lock:
            return [
                {
                    "name": name,
                    "label": info["label"],
                    "active": name == self._active_graph_name,
                    "graph_type": info.get("graph_type", ""),
                    "indexes": (
                        info["container"].list_indexes()
                        if hasattr(info["container"], "list_indexes")
                        else []
                    ),
                    "has_node_vector_index": (
                        "node_vector" in info["container"].list_indexes()
                        if hasattr(info["container"], "list_indexes")
                        else False
                    ),
                }
                for name, info in self._graphs.items()
            ]

    def switch_graph(self, name: str) -> None:
        """Switch the active graph and rebuild topology indexes + reset chat service."""
        with self._lock:
            if name not in self._graphs:
                raise KeyError(f"Unknown graph: {name!r}. Available: {list(self._graphs.keys())}")
            self._active_graph_name = name
            self._chat_service = None  # force re-creation with new graph
            self._build_topology_indexes()

    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        with self._lock:
            if self._server is not None:
                return self.url
            handler = self._make_handler()
            self._server = ThreadingHTTPServer((self.host, self.port), handler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        return self.url

    def stop(self) -> None:
        with self._lock:
            if self._server is None:
                return
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    def _build_topology_indexes(self) -> None:
        self._adj_undirected = {node_id: set() for node_id in self.container.nodes.keys()}
        self._incident_edges = {node_id: [] for node_id in self.container.nodes.keys()}
        for i, edge in enumerate(self.container.edges):
            src, tgt = edge.source, edge.target
            self._adj_undirected.setdefault(src, set()).add(tgt)
            self._adj_undirected.setdefault(tgt, set()).add(src)
            self._incident_edges.setdefault(src, []).append(i)
            self._incident_edges.setdefault(tgt, []).append(i)

    @staticmethod
    def _json_clone(payload: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(payload, ensure_ascii=False))

    def _build_subgraph_view_locked(
        self,
        session_id: str,
        state: _SessionState,
        *,
        hops: int,
    ) -> Dict[str, Any]:
        hops = max(0, int(hops))
        seed_nodes: Set[str] = set(state.nodes.keys())
        if not seed_nodes:
            for edge_info in state.edges.values():
                seed_nodes.add(str(edge_info["source"]))
                seed_nodes.add(str(edge_info["target"]))

        if not seed_nodes:
            return {
                "session_id": session_id,
                "exists": True,
                "revision": state.revision,
                "nodes": [],
                "edges": [],
                "hops": hops,
                "highlighted": {"nodes": len(state.nodes), "edges": len(state.edges)},
                "progress": dict(state.progress),
                "updated_at": state.updated_at,
                "created_at": state.created_at,
            }

        selected: Set[str] = set(seed_nodes)
        frontier: Set[str] = set(seed_nodes)
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                next_frontier.update(self._adj_undirected.get(node_id, set()))
            next_frontier -= selected
            if not next_frontier:
                break
            selected.update(next_frontier)
            frontier = next_frontier

        for edge_info in state.edges.values():
            selected.add(str(edge_info["source"]))
            selected.add(str(edge_info["target"]))

        node_payloads: List[Dict[str, Any]] = []
        for node_id in sorted(selected):
            node = self.container.get_node(node_id)
            if node is None:
                continue
            is_lightrag = node.metadata.get("_source_style") == "lightrag"
            lightrag_label = node.metadata.get("entity_name") if is_lightrag else None
            if lightrag_label:
                label_text = str(lightrag_label)
            else:
                show_text_label = "original_label" in node.metadata and bool(node.text)
                label_text = node.text if show_text_label else node.id
            payload: Dict[str, Any] = {
                "id": node.id,
                "label": label_text,
                "group": node.type,
                "title": node.id,
                "node_type": node.type,
                "node_text": node.text or "",
                "node_meta": json.dumps(node.metadata, ensure_ascii=False),
            }
            overlay_node = state.nodes.get(node.id)
            if overlay_node is not None:
                payload.update(dict(overlay_node.get("style", {})))
            node_payloads.append(payload)

        candidate_edges: Set[int] = set()
        for node_id in selected:
            candidate_edges.update(self._incident_edges.get(node_id, []))

        edge_payloads: List[Dict[str, Any]] = []
        for edge_idx in sorted(candidate_edges):
            edge = self.container.edges[edge_idx]
            if edge.source not in selected or edge.target not in selected:
                continue
            relation = "" if edge.relation is None else str(edge.relation)
            payload = {
                "id": f"g:{edge_idx}",
                "from": edge.source,
                "to": edge.target,
                "label": relation,
                "title": (
                    f"relation={relation}, weight={edge.weight}"
                    if relation
                    else f"weight={edge.weight}"
                ),
                "arrows": "to",
                "relation": relation,
            }
            overlay_key = _edge_overlay_key(edge.source, edge.target, relation)
            overlay_edge = state.edges.get(overlay_key)
            if overlay_edge is not None:
                payload.update(dict(overlay_edge.get("style", {})))
            else:
                source_overlay = state.nodes.get(edge.source, {})
                target_overlay = state.nodes.get(edge.target, {})
                source_stage = _node_retrieval_stage(source_overlay.get("style", {}))
                target_stage = _node_retrieval_stage(target_overlay.get("style", {}))
                if source_stage and target_stage and (
                    source_stage == "current" or target_stage == "current"
                ):
                    payload.update({"color": dict(_CURRENT_STEP_EDGE_COLOR), "width": 3})
                elif source_stage and target_stage:
                    payload.update({"color": dict(_PREVIOUS_STEP_EDGE_COLOR), "width": 3})
            edge_payloads.append(payload)

        return {
            "session_id": session_id,
            "exists": True,
            "revision": state.revision,
            "hops": hops,
            "nodes": node_payloads,
            "edges": edge_payloads,
            "highlighted": {"nodes": len(state.nodes), "edges": len(state.edges)},
            "progress": dict(state.progress),
            "updated_at": state.updated_at,
            "created_at": state.created_at,
        }

    def _capture_replay_snapshot_locked(self, session_id: str, state: _SessionState) -> None:
        snapshot = self._build_subgraph_view_locked(session_id, state, hops=self.default_hops)
        state.replay_snapshots.append(self._json_clone(snapshot))
        max_snapshots = 360
        if len(state.replay_snapshots) > max_snapshots:
            del state.replay_snapshots[: len(state.replay_snapshots) - max_snapshots]

    def _bump_revision_locked(self, state: _SessionState) -> None:
        state.revision += 1
        state.updated_at = time.time()

    def _notify_session_event_locked(self) -> None:
        self._session_cv.notify_all()

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._sessions[session_id] = _SessionState(metadata=dict(metadata or {}))
            self._notify_session_event_locked()
        return session_id

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def wait_for_session_event(
        self,
        session_id: str,
        *,
        since_revision: int,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        with self._session_cv:
            deadline = time.time() + max(0.1, float(timeout))
            while True:
                state = self._sessions.get(session_id)
                if state is None:
                    return {"exists": False, "revision": -1}
                if state.revision != since_revision:
                    return {
                        "exists": True,
                        "revision": state.revision,
                        "updated_at": state.updated_at,
                    }

                remain = deadline - time.time()
                if remain <= 0:
                    return {
                        "exists": True,
                        "revision": state.revision,
                        "updated_at": state.updated_at,
                        "timeout": True,
                    }
                self._session_cv.wait(timeout=remain)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                state = self._sessions[session_id]
                state.nodes.clear()
                state.edges.clear()
                state.replay_snapshots.clear()
                self._bump_revision_locked(state)
                self._notify_session_event_locked()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
            if existed:
                self._notify_session_event_locked()

    def list_sessions(self) -> List[str]:
        with self._lock:
            return sorted(self._sessions.keys())

    def set_chat_service(self, service: Any) -> None:
        with self._lock:
            self._chat_service = service

    def _ensure_chat_service(self) -> Any:
        with self._lock:
            if self._chat_service is None:
                from ..rag import GraphRAGService

                self._chat_service = GraphRAGService(self.container, visualizer=self)
            return self._chat_service

    def _reset_chat_service(self) -> None:
        """Force chat service re-creation (called after graph switch)."""
        with self._lock:
            self._chat_service = None

    def update_session(
        self,
        session_id: str,
        *,
        nodes: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        edges: Optional[Sequence[Union[Tuple[str, str], Tuple[str, str, str], List[Any], Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"Unknown session: {session_id}")
            prev_percent = float(state.progress.get("percent", 0.0))

            if metadata:
                state.metadata.update(metadata)

            if progress:
                current = int(progress.get("current", state.progress.get("current", 0)) or 0)
                total = int(progress.get("total", state.progress.get("total", 0)) or 0)
                message = str(progress.get("message", state.progress.get("message", "")) or "")
                if total > 0:
                    percent = max(0.0, min(100.0, (current / total) * 100.0))
                else:
                    percent = float(state.progress.get("percent", 0.0))
                state.progress = {
                    "current": current,
                    "total": total,
                    "percent": percent,
                    "message": message,
                }

            for node in nodes or []:
                if isinstance(node, str):
                    node_id, style = node, _default_node_style()
                elif isinstance(node, dict):
                    node_id = str(node["id"])
                    style = dict(_default_node_style())
                    style.update(dict(node.get("style", {})))
                else:
                    raise TypeError("nodes must contain str or dict items.")
                state.nodes[node_id] = {"id": node_id, "style": style}

            if edges is not None:
                for edge_info in state.edges.values():
                    edge_style = edge_info.get("style", {})
                    if isinstance(edge_style, dict) and _is_current_step_edge_style(edge_style):
                        edge_style["color"] = dict(_PREVIOUS_STEP_EDGE_COLOR)

            for edge in edges or []:
                source: str
                target: str
                relation = "RELATED"
                style = dict(_default_edge_style())
                if isinstance(edge, (tuple, list)):
                    if len(edge) == 2:
                        source, target = str(edge[0]), str(edge[1])
                    elif len(edge) == 3:
                        source, target, relation = str(edge[0]), str(edge[1]), str(edge[2])
                    else:
                        raise ValueError("edge sequence must be (source,target) or (source,target,relation)")
                elif isinstance(edge, dict):
                    source = str(edge["source"])
                    target = str(edge["target"])
                    relation = str(edge.get("relation", "RELATED"))
                    edge_style = dict(edge.get("style", {}))
                    style.update(edge_style)
                else:
                    raise TypeError("edges must contain tuple/list or dict items.")

                key = _edge_overlay_key(source, target, relation)
                state.edges[key] = {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "style": style,
                }

            self._bump_revision_locked(state)
            if nodes is not None or edges is not None or progress is not None:
                next_percent = float(state.progress.get("percent", prev_percent))
                percent_changed = abs(next_percent - prev_percent) > 1e-9
                if percent_changed or nodes is not None or edges is not None or not state.replay_snapshots:
                    self._capture_replay_snapshot_locked(session_id, state)
            self._notify_session_event_locked()

    def record(
        self,
        session_id: str,
        node_ids: Optional[Any],
        *,
        style: Optional[Dict[str, Any]] = None,
        edges: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        if node_ids is None:
            nodes = None
        else:
            if isinstance(node_ids, (list, tuple, set)):
                ids = list(node_ids)
            else:
                ids = [node_ids]
            node_style = dict(_default_node_style())
            if style:
                node_style.update(style)
            nodes = [{"id": str(node_id), "style": node_style} for node_id in ids]

        progress = {"message": message} if message else None
        self.update_session(
            session_id,
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            progress=progress,
        )

    def set_progress(self, session_id: str, current: int, total: int, message: str = "") -> None:
        self.update_session(
            session_id,
            progress={
                "current": int(current),
                "total": int(total),
                "message": message,
            },
        )

    def highlight_search_result(self, session_id: str, result: Dict[str, Any]) -> None:
        raw_seed_nodes = result.get("seed_nodes", [])
        raw_edges = result.get("one_hop_edges", [])

        nodes: List[Dict[str, Any]] = []
        for item in raw_seed_nodes:
            if isinstance(item, str):
                nodes.append({"id": item})
            elif isinstance(item, dict) and item.get("id") is not None:
                nodes.append({"id": str(item["id"])})

        edges: List[Dict[str, Any]] = []
        for item in raw_edges:
            if isinstance(item, dict) and item.get("source") is not None and item.get("target") is not None:
                edges.append(
                    {
                        "source": str(item["source"]),
                        "target": str(item["target"]),
                        "relation": str(item.get("relation", "RELATED")),
                    }
                )
        self.update_session(session_id, nodes=nodes, edges=edges)

    def get_session_snapshot(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return {
                    "session_id": session_id,
                    "exists": False,
                    "revision": -1,
                    "nodes": [],
                    "edges": [],
                    "metadata": {},
                    "progress": {"current": 0, "total": 0, "percent": 0.0, "message": ""},
                    "updated_at": None,
                }
            return {
                "session_id": session_id,
                "exists": True,
                "revision": state.revision,
                "nodes": list(state.nodes.values()),
                "edges": list(state.edges.values()),
                "metadata": dict(state.metadata),
                "progress": dict(state.progress),
                "updated_at": state.updated_at,
                "created_at": state.created_at,
            }

    def get_session_subgraph(self, session_id: str, hops: int = 2) -> Dict[str, Any]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return {
                    "session_id": session_id,
                    "exists": False,
                    "revision": -1,
                    "nodes": [],
                    "edges": [],
                    "hops": hops,
                    "highlighted": {"nodes": 0, "edges": 0},
                }
            return self._build_subgraph_view_locked(session_id, state, hops=hops)

    def get_session_replay(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return {
                    "session_id": session_id,
                    "exists": False,
                    "revision": -1,
                    "snapshots": [],
                    "count": 0,
                }
            return {
                "session_id": session_id,
                "exists": True,
                "revision": state.revision,
                "hops": self.default_hops,
                "count": len(state.replay_snapshots),
                "snapshots": [self._json_clone(snapshot) for snapshot in state.replay_snapshots],
            }

    def _safe_static_path(self, rel_path: str) -> Optional[Path]:
        rel = rel_path.lstrip("/")
        base = self._web_dir.resolve()
        candidate = (base / rel).resolve()
        if base not in candidate.parents and candidate != base:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    @staticmethod
    def _make_unique_name(existing: Dict[str, Any], base: str) -> str:
        if base not in existing:
            return base
        suffix = 2
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"

    def _make_handler(self):
        visualizer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _write_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_text(self, text: str, content_type: str, status: int = 200) -> None:
                body = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_file(self, path: Path, content_type: str, status: int = 200) -> None:
                body = path.read_bytes()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _begin_sse(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

            def _send_sse(self, payload: Dict[str, Any], *, event: Optional[str] = None) -> None:
                if event:
                    self.wfile.write(f"event: {event}\n".encode("utf-8"))
                data = json.dumps(payload, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

            def _read_json_body(self) -> Dict[str, Any]:
                length_raw = self.headers.get("Content-Length", "0")
                try:
                    length = int(length_raw)
                except ValueError:
                    raise ValueError("Invalid Content-Length")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON body: {exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload

            def _parse_multipart(self) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    raise ValueError("Expected multipart/form-data")
                length_raw = self.headers.get("Content-Length", "0")
                try:
                    int(length_raw)
                except ValueError:
                    raise ValueError("Invalid Content-Length")

                env = {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": length_raw,
                }
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ=env,
                    keep_blank_values=True,
                )
                # Keep a reference so uploaded temp files remain open while we stream them to disk.
                self._multipart_form = form
                fields: Dict[str, str] = {}
                files: List[Dict[str, Any]] = []
                if not form.list:
                    return fields, files

                for item in form.list:
                    if item.filename:
                        files.append(
                            {
                                "name": item.name,
                                "filename": item.filename,
                                "file": item.file,
                                "size": getattr(item, "length", None),
                            }
                        )
                    else:
                        fields[item.name] = str(item.value)
                return fields, files

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/":
                    index_path = visualizer._safe_static_path("index.html")
                    if index_path is None:
                        self._write_json({"error": "index.html not found"}, status=500)
                        return
                    self._write_file(index_path, "text/html; charset=utf-8")
                    return

                if path.startswith("/static/"):
                    rel_path = path[len("/static/") :]
                    static_path = visualizer._safe_static_path(rel_path)
                    if static_path is None:
                        self._write_json({"error": f"static file not found: {rel_path}"}, status=404)
                        return
                    content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
                    if content_type.startswith("text/") or content_type in {"application/javascript"}:
                        content_type = f"{content_type}; charset=utf-8"
                    self._write_file(static_path, content_type)
                    return

                if path == "/api/health":
                    self._write_json({"ok": True, "url": visualizer.url})
                    return

                if path == "/api/config":
                    self._write_json(
                        {
                            "poll_interval_ms": visualizer.poll_interval_ms,
                            "default_hops": visualizer.default_hops,
                            "chat_enabled": True,
                            "default_chat_retrieval": "one-hop",
                            "graphs": visualizer.list_graphs(),
                        }
                    )
                    return

                if path == "/api/graphs":
                    self._write_json({"graphs": visualizer.list_graphs()})
                    return

                if path == "/api/sessions":
                    self._write_json({"sessions": visualizer.list_sessions()})
                    return

                prefix = "/api/session/"
                if path.startswith(prefix):
                    rest = path[len(prefix) :].strip("/")
                    if not rest:
                        self._write_json({"error": "session id is required"}, status=400)
                        return

                    if rest.endswith("/events"):
                        session_id = rest[: -len("/events")].strip("/")
                        if not session_id:
                            self._write_json({"error": "session id is required"}, status=400)
                            return
                        if not visualizer.has_session(session_id):
                            self._write_json(
                                {"error": f"session not found: {session_id}", "session_id": session_id},
                                status=404,
                            )
                            return

                        try:
                            since = int((query.get("since") or ["-1"])[0])
                        except (TypeError, ValueError):
                            since = -1

                        try:
                            self._begin_sse()
                            while True:
                                evt = visualizer.wait_for_session_event(
                                    session_id,
                                    since_revision=since,
                                    timeout=20.0,
                                )
                                if not evt.get("exists"):
                                    self._send_sse(
                                        {
                                            "session_id": session_id,
                                            "exists": False,
                                            "revision": -1,
                                        },
                                        event="session_deleted",
                                    )
                                    return

                                rev = int(evt.get("revision", since))
                                if rev != since:
                                    since = rev
                                    self._send_sse(
                                        {
                                            "session_id": session_id,
                                            "exists": True,
                                            "revision": rev,
                                            "updated_at": evt.get("updated_at"),
                                        },
                                        event="session_update",
                                    )
                                else:
                                    self.wfile.write(b": keep-alive\n\n")
                                    self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return

                    if rest.endswith("/subgraph"):
                        session_id = rest[: -len("/subgraph")].strip("/")
                        if not session_id:
                            self._write_json({"error": "session id is required"}, status=400)
                            return
                        try:
                            hops = int((query.get("hops") or [str(visualizer.default_hops)])[0])
                        except (TypeError, ValueError):
                            hops = visualizer.default_hops
                        view = visualizer.get_session_subgraph(session_id, hops=hops)
                        if not view.get("exists"):
                            self._write_json(view, status=404)
                            return
                        self._write_json(view)
                        return

                    if rest.endswith("/replay"):
                        session_id = rest[: -len("/replay")].strip("/")
                        if not session_id:
                            self._write_json({"error": "session id is required"}, status=400)
                            return
                        replay = visualizer.get_session_replay(session_id)
                        if not replay.get("exists"):
                            self._write_json(replay, status=404)
                            return
                        self._write_json(replay)
                        return

                    session_id = rest
                    snap = visualizer.get_session_snapshot(session_id)
                    if not snap.get("exists"):
                        self._write_json(snap, status=404)
                        return
                    self._write_json(snap)
                    return

                self._write_json({"error": "not found"}, status=404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/api/import":
                    try:
                        fields, files = self._parse_multipart()
                    except ValueError as exc:
                        self._write_json({"error": str(exc)}, status=400)
                        return

                    adapter_key = (fields.get("adapter") or "").strip()
                    if not adapter_key:
                        self._write_json({"error": "adapter is required"}, status=400)
                        return

                    label_input = (fields.get("label") or "").strip()
                    dataset_name = (fields.get("dataset_name") or "").strip()
                    if not dataset_name:
                        self._write_json({"error": "dataset_name is required"}, status=400)
                        return
                    importer_path = _ADAPTER_IMPORTERS.get(adapter_key)
                    if importer_path is None:
                        self._write_json(
                            {"error": f"Unknown adapter key {adapter_key!r}.", "valid": list(_ADAPTER_IMPORTERS)},
                            status=400,
                        )
                        return

                    source_path: Union[str, Path]
                    temp_dir: Optional[Path] = None
                    try:
                        if adapter_key == "freebasekg":
                            source_path = dataset_name
                        else:
                            if not files:
                                self._write_json(
                                    {"error": "No files uploaded for import.", "adapter": adapter_key},
                                    status=400,
                                )
                                return

                            temp_dir = Path(mkdtemp(prefix="graph_import_"))
                            for file_item in files:
                                filename = str(file_item.get("filename") or "upload.bin")
                                rel = Path(filename)
                                safe_parts = [p for p in rel.parts if p not in {"", ".", ".."}]
                                safe_rel = Path(*safe_parts) if safe_parts else Path(rel.name)
                                dest = (temp_dir / safe_rel).resolve()
                                if temp_dir not in dest.parents and dest != temp_dir:
                                    self._write_json({"error": "Invalid upload path"}, status=400)
                                    return
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                file_handle = file_item.get("file")
                                if not file_handle or getattr(file_handle, "closed", False):
                                    self._write_json(
                                        {"error": f"Upload stream closed for {filename}."}, status=400
                                    )
                                    return
                                try:
                                    file_handle.seek(0)
                                except Exception:
                                    pass
                                with dest.open("wb") as fh:
                                    shutil.copyfileobj(file_handle, fh, length=1024 * 1024)
                                expected_size = file_item.get("size")
                                if isinstance(expected_size, int) and expected_size > 0:
                                    actual_size = dest.stat().st_size
                                    if actual_size != expected_size:
                                        self._write_json(
                                            {
                                                "error": (
                                                    f"Upload truncated for {filename}. "
                                                    f"Expected {expected_size} bytes, got {actual_size}."
                                                )
                                            },
                                            status=400,
                                        )
                                        return

                            source_path = temp_dir
                            zip_files = [p for p in temp_dir.glob("**/*.zip") if p.is_file()]
                            if len(zip_files) == 1 and zip_files[0].parent == temp_dir:
                                zip_path = zip_files[0]
                                with zipfile.ZipFile(zip_path, "r") as zf:
                                    bad = zf.testzip()
                                    if bad is not None:
                                        self._write_json(
                                            {"error": f"Zip file is corrupted (bad entry: {bad})."},
                                            status=400,
                                        )
                                        return
                                with zipfile.ZipFile(zip_path, "r") as zf:
                                    zf.extractall(temp_dir)
                                zip_path.unlink(missing_ok=True)

                            while True:
                                children = [p for p in source_path.iterdir() if p.name not in (".DS_Store", "__MACOSX")]
                                if len(children) == 1 and children[0].is_dir():
                                    source_path = children[0]
                                else:
                                    break

                        if adapter_key == "hipporag":
                            resolved_root, error = _resolve_hipporag_root(source_path)
                            if error:
                                self._write_json({"error": error}, status=400)
                                return
                            source_path = resolved_root

                        validation_error = _validate_import_requirements(
                            adapter_key,
                            source_path,
                            dataset_name=dataset_name,
                        )
                        if validation_error:
                            self._write_json({"error": validation_error}, status=400)
                            return

                        label = label_input or _build_graph_label(
                            adapter_key,
                            source_path,
                            dataset_name=dataset_name,
                        )

                        module_path, fn_name = importer_path.rsplit(".", 1)
                        mod = importlib.import_module(
                            module_path, package=__name__.rsplit(".", 1)[0]
                        )
                        import_fn = getattr(mod, fn_name)
                        graph = import_fn(source_path)

                        with visualizer._lock:
                            name = LiveGraphVisualizer._make_unique_name(visualizer._graphs, label or adapter_key)
                            visualizer.register_graph(
                                name, graph, label=label, graph_type=adapter_key
                            )
                            visualizer.switch_graph(name)

                        self._write_json(
                            {
                                "ok": True,
                                "name": name,
                                "label": label,
                                "active": name,
                                "graphs": visualizer.list_graphs(),
                            }
                        )
                        return
                    except Exception as exc:
                        if hasattr(self, "_multipart_form"):
                            self._multipart_form = None
                        if temp_dir is not None and temp_dir.exists():
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        self._write_json({"error": str(exc)}, status=400)
                        return
                    finally:
                        if hasattr(self, "_multipart_form"):
                            self._multipart_form = None

                try:
                    payload = self._read_json_body()
                except ValueError as exc:
                    self._write_json({"error": str(exc)}, status=400)
                    return

                if path == "/api/session":
                    metadata = payload.get("metadata")
                    if metadata is not None and not isinstance(metadata, dict):
                        self._write_json({"error": "metadata must be object"}, status=400)
                        return
                    session_id = visualizer.create_session(metadata=metadata)
                    self._write_json(
                        {
                            "ok": True,
                            "session_id": session_id,
                            "snapshot": visualizer.get_session_snapshot(session_id),
                        },
                        status=201,
                    )
                    return

                if path == "/api/chat":
                    try:
                        service = visualizer._ensure_chat_service()
                        result = service.chat(payload)
                    except Exception as exc:
                        self._write_json({"error": str(exc)}, status=400)
                        return

                    self._write_json({"ok": True, **result})
                    return

                if path == "/api/graph/switch":
                    name = payload.get("name")
                    if not name or not isinstance(name, str):
                        self._write_json({"error": "'name' is required"}, status=400)
                        return
                    try:
                        visualizer.switch_graph(name)
                    except KeyError as exc:
                        self._write_json({"error": str(exc)}, status=404)
                        return
                    self._write_json(
                        {
                            "ok": True,
                            "active": name,
                            "graphs": visualizer.list_graphs(),
                        }
                    )
                    return

                prefix = "/api/session/"
                if path.startswith(prefix):
                    rest = path[len(prefix) :].strip("/")
                    if not rest:
                        self._write_json({"error": "session id is required"}, status=400)
                        return

                    if rest.endswith("/update"):
                        session_id = rest[: -len("/update")].strip("/")
                        if not session_id:
                            self._write_json({"error": "session id is required"}, status=400)
                            return
                        if not visualizer.has_session(session_id):
                            self._write_json(
                                {"error": f"session not found: {session_id}", "session_id": session_id},
                                status=404,
                            )
                            return

                        nodes = payload.get("nodes")
                        edges = payload.get("edges")
                        metadata = payload.get("metadata")
                        progress = payload.get("progress")
                        if nodes is not None and not isinstance(nodes, list):
                            self._write_json({"error": "nodes must be array"}, status=400)
                            return
                        if edges is not None and not isinstance(edges, list):
                            self._write_json({"error": "edges must be array"}, status=400)
                            return
                        if metadata is not None and not isinstance(metadata, dict):
                            self._write_json({"error": "metadata must be object"}, status=400)
                            return
                        if progress is not None and not isinstance(progress, dict):
                            self._write_json({"error": "progress must be object"}, status=400)
                            return

                        try:
                            visualizer.update_session(
                                session_id,
                                nodes=nodes,
                                edges=edges,
                                metadata=metadata,
                                progress=progress,
                            )
                        except Exception as exc:
                            self._write_json({"error": str(exc)}, status=400)
                            return

                        self._write_json(
                            {
                                "ok": True,
                                "session_id": session_id,
                                "snapshot": visualizer.get_session_snapshot(session_id),
                            }
                        )
                        return

                    if rest.endswith("/clear"):
                        session_id = rest[: -len("/clear")].strip("/")
                        if not session_id:
                            self._write_json({"error": "session id is required"}, status=400)
                            return
                        if not visualizer.has_session(session_id):
                            self._write_json(
                                {"error": f"session not found: {session_id}", "session_id": session_id},
                                status=404,
                            )
                            return
                        visualizer.clear_session(session_id)
                        self._write_json(
                            {
                                "ok": True,
                                "session_id": session_id,
                                "snapshot": visualizer.get_session_snapshot(session_id),
                            }
                        )
                        return

                self._write_json({"error": "not found"}, status=404)

            def do_DELETE(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                prefix = "/api/session/"
                if path.startswith(prefix):
                    session_id = path[len(prefix) :].strip("/")
                    if not session_id:
                        self._write_json({"error": "session id is required"}, status=400)
                        return
                    if not visualizer.has_session(session_id):
                        self._write_json(
                            {"error": f"session not found: {session_id}", "session_id": session_id},
                            status=404,
                        )
                        return
                    visualizer.delete_session(session_id)
                    self._write_json({"ok": True, "session_id": session_id})
                    return

                self._write_json({"error": "not found"}, status=404)

        return _Handler


def serve_graph(
    container: SimpleGraphContainer,
    *,
    name: str = "default",
    label: str = "",
    graph_type: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
    _visualizer: Optional["LiveGraphVisualizer"] = None,
) -> LiveGraphVisualizer:
    """Serve a single graph container and return the visualizer.

    If *_visualizer* is provided (used internally by :func:`serve_multi`),
    the graph is registered on that existing instance instead of creating a new one.
    """
    if _visualizer is None:
        visualizer = LiveGraphVisualizer(
            host=host,
            port=port,
            poll_interval_ms=poll_interval_ms,
            default_hops=default_hops,
        )
        visualizer.register_graph(
            name, container, label=label or name, graph_type=graph_type or name
        )
        visualizer.start()
    else:
        _visualizer.register_graph(
            name, container, label=label or name, graph_type=graph_type or name
        )
        visualizer = _visualizer
    return visualizer


def serve_fastinsight(
    source_path: Union[str, Path],
    *,
    name: str = "fastinsight",
    label: str = "FastInsight",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    from ..adapters.fastinsight import import_graph_from_fastinsight

    graph = import_graph_from_fastinsight(source_path)
    return serve_graph(
        graph,
        name=name,
        label=label,
        graph_type="fastinsight",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

def serve_lightrag(
    source_path: Union[str, Path],
    *,
    name: str = "lightrag",
    label: str = "LightRAG",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
    attach_index: bool = True,
    load_embeddings: bool = True,
    lazy_load: bool = False,
) -> LiveGraphVisualizer:
    from ..adapters.lightrag import import_graph_from_lightrag

    def _set_env() -> None:
        os.environ["LIGHTRAG_ATTACH_INDEX"] = "1" if attach_index else "0"
        os.environ["LIGHTRAG_LOAD_EMBEDDINGS"] = "1" if load_embeddings else "0"

    if not lazy_load:
        _set_env()
        graph = import_graph_from_lightrag(source_path)
        return serve_graph(
            graph,
            name=name,
            label=label,
            graph_type="lightrag",
            host=host,
            port=port,
            poll_interval_ms=poll_interval_ms,
            default_hops=default_hops,
        )

    # Lazy-load: start server with an empty graph, then replace when loaded.
    empty = SimpleGraphContainer()
    visualizer = serve_graph(
        empty,
        name=name,
        label=label,
        graph_type="lightrag",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

    def _load_graph() -> None:
        _set_env()
        graph = import_graph_from_lightrag(source_path)
        visualizer.register_graph(name, graph, label=label, graph_type="lightrag")

    threading.Thread(target=_load_graph, daemon=True).start()
    return visualizer

def serve_hipporag(
    source_path: Union[str, Path],
    *,
    name: str = "hipporag",
    label: str = "HippoRAG",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    from ..adapters.hipporag import import_graph_from_hipporag

    graph = import_graph_from_hipporag(source_path)
    return serve_graph(
        graph,
        name=name,
        label=label,
        graph_type="hipporag",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

def serve_g_retriever(
    source_path: Union[str, Path],
    *,
    name: str = "g_retriever",
    label: str = "G-Retriever",
    graph_id: Optional[str] = "all",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    """Serve a G-Retriever graph.

    *graph_id* selects a single CSV pair (e.g. ``"0"``), or ``"all"`` (default)
    to merge every CSV pair in the source directory into one big graph.
    """
    from ..adapters.g_retriever import import_graph_from_g_retriever

    graph = import_graph_from_g_retriever(source_path, graph_id=graph_id)
    return serve_graph(
        graph,
        name=name,
        label=label,
        graph_type="g_retriever",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

def serve_expla_graphs(
    source_path: Union[str, Path],
    *,
    name: str = "expla_graphs",
    label: str = "ExplaGraphs",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    """Serve an expla_graphs TSV (train_dev.tsv) graph."""
    from ..adapters.expla_graphs import import_graph_from_expla_graphs

    graph = import_graph_from_expla_graphs(source_path)
    return serve_graph(
        graph,
        name=name,
        label=label,
        graph_type="expla_graphs",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

def serve_freebasekg(
    source_path: Union[str, Path],
    *,
    name: str = "freebasekg",
    label: str = "FreebaseKG",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    """Serve a Freebase KG graph from the RoG-webqsp dataset on Hugging Face."""
    from ..adapters.freebasekg import import_graph_from_freebasekg

    graph = import_graph_from_freebasekg(source_path)
    return serve_graph(
        graph,
        name=name,
        label=label,
        graph_type="freebasekg",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

def serve_tog(
    source_path: Union[str, Path],
    *,
    name: str = "tog",
    label: str = "Think-on-Graph",
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    from ..adapters.tog import import_graph_from_tog

    graph = import_graph_from_tog(source_path)
    return serve_graph(
        graph,
        name=name,
        label=label,
        graph_type="tog",
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )


_ADAPTER_IMPORTERS: Dict[str, str] = {
    "fastinsight": "..adapters.fastinsight.import_graph_from_fastinsight",
    "lightrag": "..adapters.lightrag.import_graph_from_lightrag",
    "hipporag": "..adapters.hipporag.import_graph_from_hipporag",
    "g_retriever": "..adapters.g_retriever.import_graph_from_g_retriever",
    "expla_graphs": "..adapters.expla_graphs.import_graph_from_expla_graphs",
    "freebasekg": "..adapters.freebasekg.import_graph_from_freebasekg",
}

_ADAPTER_DEFAULT_LABELS: Dict[str, str] = {
    "fastinsight": "Component",
    "lightrag": "Attribute Bundle",
    "hipporag": "Topology-Semantic",
    "g_retriever": "Subgraph Union",
    "expla_graphs": "Triplet Sequence",
}


def _pretty_dataset_name(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    # Use last path segment if it looks like a path or HF repo id.
    if "/" in raw:
        raw = raw.split("/")[-1]
    raw = raw.replace("_", " ").replace("-", " ")
    parts = [p for p in raw.split() if p]
    pretty: List[str] = []
    for part in parts:
        if part.isupper():
            pretty.append(part)
        elif any(ch.isdigit() for ch in part):
            pretty.append(part)
        else:
            pretty.append(part[:1].upper() + part[1:])
    return " ".join(pretty)


def _dataset_name_from_source(
    adapter_key: str,
    source_path: Union[str, Path, None],
    *,
    graph_id: Optional[str] = None,
    dataset_name: str = "",
) -> str:
    if dataset_name.strip():
        return _pretty_dataset_name(dataset_name)
    if graph_id and graph_id != "all":
        return _pretty_dataset_name(str(graph_id))
    if adapter_key == "freebasekg":
        return _pretty_dataset_name(str(source_path or ""))
    if source_path is None:
        return ""
    try:
        path = Path(source_path)
        if path.name:
            return _pretty_dataset_name(path.name)
        if path.parent and path.parent.name:
            return _pretty_dataset_name(path.parent.name)
    except Exception:
        return _pretty_dataset_name(str(source_path))
    return ""


def _build_graph_label(
    adapter_key: str,
    source_path: Union[str, Path, None],
    *,
    graph_id: Optional[str] = None,
    dataset_name: str = "",
) -> str:
    format_label = _ADAPTER_DEFAULT_LABELS.get(str(adapter_key), str(adapter_key))
    dataset_label = _dataset_name_from_source(
        adapter_key,
        source_path,
        graph_id=graph_id,
        dataset_name=dataset_name,
    )
    if dataset_label:
        return f"{dataset_label} ({format_label})"
    return format_label


def _validate_import_requirements(
    adapter_key: str,
    source_path: Union[str, Path],
    *,
    dataset_name: str = "",
) -> Optional[str]:
    importer_path = _ADAPTER_IMPORTERS.get(adapter_key)
    if importer_path is None:
        return f"Unknown adapter key: {adapter_key}"

    # Standard requirements strings for error messages
    requirements_msg = {
        "fastinsight": "Requires nodes.jsonl and edges.jsonl.",
        "lightrag": "Requires vdb_entities.json and vdb_relationships.json.",
        "hipporag": "Requires graph.pickle, entity_embeddings, chunk_embeddings, fact_embeddings, and openie_results_ner_*.json.",
        "g_retriever": "Requires nodes/{i}.csv, edges/{i}.csv, and graphs/{i}.pt files for each query i.",
        "expla_graphs": "Requires a .tsv file.",
        "freebasekg": "Requires a Hugging Face dataset name (e.g. 'rmanluo/RoG-webqsp').",
    }

    try:
        # Import the adapter class
        module_path, _ = importer_path.rsplit(".", 1)
        mod = importlib.import_module(module_path, package=__name__.rsplit(".", 1)[0])

        adapter_class_map = {
            "fastinsight": "FastInsightAdapter",
            "lightrag": "LightRAGAdapter",
            "hipporag": "HippoRAGAdapter",
            "g_retriever": "GRetrieverAdapter",
            "expla_graphs": "ExplaGraphsAdapter",
            "freebasekg": "FreebaseKGAdapter",
        }
        classname = adapter_class_map.get(adapter_key)

        adapter_cls = getattr(mod, classname, None) if classname else None
        if adapter_cls:
            adapter = adapter_cls()
            if not adapter.can_import(source_path):
                msg = requirements_msg.get(adapter_key, "File requirements not met.")
                return f"{msg}"
    except Exception as exc:
        return f"Validation error for {adapter_key}: {exc}"

    return None


def _resolve_hipporag_root(source_path: Union[str, Path]) -> Tuple[Path, Optional[str]]:
    src = Path(source_path)
    has_graph = (src / "graph.pickle").exists()
    has_openie = any(p.is_file() for p in src.glob("openie_results_ner_*.json")) or any(p.is_file() for p in src.parent.glob("openie_results_ner_*.json"))
    if has_graph and has_openie:
        return src, None

    candidates: List[Path] = []
    try:
        for graph_path in src.rglob("graph.pickle"):
            parent = graph_path.parent
            # Check for openie in the same dir OR the parent of the graph.pickle dir.
            if any(p.is_file() for p in parent.glob("openie_results_ner_*.json")) or any(p.is_file() for p in parent.parent.glob("openie_results_ner_*.json")):
                candidates.append(parent)
    except Exception:
        return src, None

    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        preview = ", ".join(str(p) for p in candidates[:3])
        return src, f"Multiple Topology-Semantic graph roots found. Please zip a single root. Examples: {preview}"
    return src, None


def serve_multi(
    graphs: Dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    """Start a visualizer server with multiple named graphs pre-loaded.

    ``graphs`` is a mapping from a human-readable graph *name* to either:

    * A ``(adapter_key, source_path)`` tuple, where *adapter_key* is one of
      ``"fastinsight"``, ``"lightrag"``, ``"hipporag"``, ``"g_retriever"``, ``"expla_graphs"``, ``"freebasekg"``, ``"tog"``.
    * A ``(adapter_key, source_path, label)`` tuple for a custom display label.
    * An already-loaded :class:`~GraphContainer.core.SimpleGraphContainer` instance.

    Example::

        visualizer = serve_multi(
            {
                "LightRAG":    ("lightrag",    "/data/lightrag"),
                "HippoRAG":    ("hipporag",    "/data/hipporag"),
                "G-Retriever": ("g_retriever", "/data/g_retriever"),
                "Expla Graphs": ("expla_graphs", "/data/expla_graphs"),
                "Freebase KG": ("freebasekg", "/data/freebasekg"),
                "ToG":         ("tog",          "/data/tog"),
            },
            port=8765,
        )
    """
    import importlib

    visualizer = LiveGraphVisualizer(
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )

    for name, spec in graphs.items():
        if isinstance(spec, tuple):
            extra_kwargs: Dict[str, Any] = {}
            if len(spec) == 2:
                adapter_key, source_path = spec
                label = _build_graph_label(str(adapter_key), source_path)
            elif len(spec) == 3:
                adapter_key, source_path, label = spec
            elif len(spec) == 4:
                adapter_key, source_path, label, extra_kwargs = spec
            else:
                raise ValueError(
                    f"Graph spec for {name!r} must be (adapter_key, path), "
                    f"(adapter_key, path, label), or (adapter_key, path, label, kwargs); "
                    f"got tuple of length {len(spec)}"
                )
            importer_path = _ADAPTER_IMPORTERS.get(str(adapter_key))
            if importer_path is None:
                raise ValueError(
                    f"Unknown adapter key {adapter_key!r}. "
                    f"Valid keys: {list(_ADAPTER_IMPORTERS.keys())}"
                )
            # e.g. "..adapters.lightrag.import_graph_from_lightrag"
            module_path, fn_name = importer_path.rsplit(".", 1)
            mod = importlib.import_module(module_path, package=__name__.rsplit(".", 1)[0])
            import_fn = getattr(mod, fn_name)
            container = import_fn(source_path, **(extra_kwargs or {}))
        else:
            # Assume it's already a SimpleGraphContainer
            container = spec
            label = name

        visualizer.register_graph(name, container, label=label)

    if not visualizer._graphs:
        raise ValueError("No graphs were registered. Pass at least one entry in 'graphs'.")

    visualizer.start()
    return visualizer


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve GraphContainer live visualizer on localhost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Single format
    python -m GraphContainer.visualizer.live_visualizer --format lightrag --source /data/lightrag
    python -m GraphContainer.visualizer.live_visualizer --format hipporag --source /data/hipporag
    python -m GraphContainer.visualizer.live_visualizer --format g_retriever --source /data/g_retriever
    python -m GraphContainer.visualizer.live_visualizer --format expla_graphs --source /data/expla_graphs
    python -m GraphContainer.visualizer.live_visualizer --format freebasekg --source rmanluo/RoG-webqsp
    python -m GraphContainer.visualizer.live_visualizer --format tog --source /data/tog

    # Multiple formats at once (specify --graph format:path pairs)
    python -m GraphContainer.visualizer.live_visualizer \\
        --graph lightrag:/data/lightrag \\
        --graph hipporag:/data/hipporag \\
        --graph g_retriever:/data/g_retriever \\
        --graph expla_graphs:/data/expla_graphs \\
        --graph freebasekg:rmanluo/RoG-webqsp \\
        --graph tog:/data/tog
    """,
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["lightrag", "hipporag", "g_retriever", "expla_graphs", "freebasekg", "tog", "fastinsight"],
        help="Graph adapter format (use with --source for a single graph)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source directory when using --format (single graph mode)",
    )
    parser.add_argument(
        "--graph",
        metavar="FORMAT:PATH",
        action="append",
        dest="graphs",
        default=[],
        help=(
            "Register a graph as FORMAT:PATH (repeatable). "
            "FORMAT is one of: lightrag, hipporag, g_retriever, expla_graphs, freebasekg, tog, fastinsight. "
            "Example: --graph lightrag:/data/lr --graph hipporag:/data/hr"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-ms", type=int, default=600)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument(
        "--graph-id",
        default=None,
        dest="graph_id",
        help=(
            "Graph ID for formats that store multiple graphs (e.g. g_retriever). "
            "Use 'all' to merge all CSV pairs into one graph (default when omitted). "
            "Use a specific id like '0' to load a single graph."
        ),
    )
    args = parser.parse_args()

    _serve_fn_map = {
        "lightrag": serve_lightrag,
        "hipporag": serve_hipporag,
        "g_retriever": serve_g_retriever,
        "expla_graphs": serve_expla_graphs,
        "freebasekg": serve_freebasekg,
        "tog": serve_tog,
        "fastinsight": serve_fastinsight,
    }

    common_kwargs: Dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "poll_interval_ms": args.poll_ms,
        "default_hops": args.hops,
    }

    # ── Multi-graph mode ────────────────────────────────────────────────
    if args.graphs:
        multi_spec: Dict[str, Any] = {}
        for entry in args.graphs:
            sep = entry.find(":")
            if sep == -1:
                parser.error(f"--graph value must be FORMAT:PATH, got: {entry!r}")
            fmt = entry[:sep].strip()
            raw_path = entry[sep + 1:].strip()
            if fmt not in _ADAPTER_IMPORTERS:
                parser.error(
                    f"Unknown format {fmt!r}. Valid: {list(_ADAPTER_IMPORTERS.keys())}"
                )
            path = raw_path
            extra_kwargs: Dict[str, Any] = {}
            if "?graph_id=" in raw_path or "#graph_id=" in raw_path:
                if "?" in raw_path:
                    path, query = raw_path.split("?", 1)
                else:
                    path, query = raw_path.split("#", 1)
                query_params = parse_qs(query)
                if "graph_id" in query_params:
                    extra_kwargs["graph_id"] = query_params["graph_id"][0]
            elif raw_path.endswith("@all"):
                path = raw_path[:-4]
                extra_kwargs["graph_id"] = "all"
            # Use dataset + format label as the graph name (unique per entry)
            label = _build_graph_label(fmt, path, graph_id=extra_kwargs.get("graph_id"))
            name = LiveGraphVisualizer._make_unique_name(multi_spec, label)
            if extra_kwargs:
                multi_spec[name] = (fmt, path, label, extra_kwargs)
            else:
                multi_spec[name] = (fmt, path, label)

        # Also include --source/--format if provided alongside --graph
        if args.format and args.source:
            fmt, path = args.format, args.source
            label = _build_graph_label(fmt, path, graph_id=args.graph_id)
            name = LiveGraphVisualizer._make_unique_name(multi_spec, label)
            if fmt == "g_retriever":
                multi_spec[name] = (fmt, path, label, {"graph_id": args.graph_id or "all"})
            else:
                multi_spec[name] = (fmt, path, label)

        visualizer = serve_multi(multi_spec, **common_kwargs)
        active = visualizer._active_graph_name
        print(f"Visualizer URL: {visualizer.url}")
        print(f"  Loaded graphs: {[g['name'] for g in visualizer.list_graphs()]}")
        print(f"  Active graph:  {active}")

    # ── Single-graph mode ───────────────────────────────────────────────
    else:
        if not args.source:
            parser.error("--source is required when not using --graph")
        fmt = args.format or "lightrag"
        serve_fn = _serve_fn_map[fmt]
        extra_kwargs: Dict[str, Any] = {}
        if fmt == "g_retriever":
            # default to "all" so multi-CSV datasets just work out of the box
            extra_kwargs["graph_id"] = args.graph_id if args.graph_id is not None else "all"
        label = _build_graph_label(fmt, args.source, graph_id=extra_kwargs.get("graph_id"))
        name = label
        visualizer = serve_fn(args.source, name=name, label=label, **common_kwargs, **extra_kwargs)
        print(f"Visualizer URL: {visualizer.url}  (format={fmt})")

    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        visualizer.stop()


if __name__ == "__main__":
    _main()
