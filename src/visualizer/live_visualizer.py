from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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


class LiveGraphVisualizer:
    """
    Backend service for session-based graph visualization.

    Frontend static files are served from `visualizer/web`.
    """

    def __init__(
        self,
        container: SimpleGraphContainer,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        poll_interval_ms: int = 600,
        default_hops: int = 2,
    ):
        self.container = container
        self.host = host
        self.port = int(port)
        self.poll_interval_ms = int(max(200, poll_interval_ms))
        self.default_hops = int(max(0, default_hops))

        self._lock = threading.RLock()
        self._session_cv = threading.Condition(self._lock)
        self._sessions: Dict[str, _SessionState] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._chat_service: Optional[Any] = None

        self._adj_undirected: Dict[str, Set[str]] = {}
        self._incident_edges: Dict[str, List[int]] = {}
        self._build_topology_indexes()

        self._web_dir = Path(__file__).resolve().parent / "web"

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
                    # Do not allow session overlay to override edge color.
                    edge_style.pop("color", None)
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
            self._notify_session_event_locked()

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
                payload: Dict[str, Any] = {
                    "id": node.id,
                    "label": node.id,
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
                relation = edge.relation or "RELATED"
                payload: Dict[str, Any] = {
                    "id": f"g:{edge_idx}",
                    "from": edge.source,
                    "to": edge.target,
                    "label": relation,
                    "title": f"relation={relation}, weight={edge.weight}",
                    "arrows": "to",
                    "relation": relation,
                }
                overlay_key = _edge_overlay_key(edge.source, edge.target, relation)
                overlay_edge = state.edges.get(overlay_key)
                if overlay_edge is not None:
                    payload.update(dict(overlay_edge.get("style", {})))
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

                if path == "/static/app.js":
                    js_path = visualizer._safe_static_path("app.js")
                    if js_path is None:
                        self._write_json({"error": "app.js not found"}, status=500)
                        return
                    self._write_file(js_path, "application/javascript; charset=utf-8")
                    return

                if path == "/static/style.css":
                    css_path = visualizer._safe_static_path("style.css")
                    if css_path is None:
                        self._write_json({"error": "style.css not found"}, status=500)
                        return
                    self._write_file(css_path, "text/css; charset=utf-8")
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
                        }
                    )
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
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    visualizer = LiveGraphVisualizer(
        container,
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )
    visualizer.start()
    return visualizer


def serve_fastinsight(
    source_path: Union[str, Path],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 600,
    default_hops: int = 1,
) -> LiveGraphVisualizer:
    from ..adapters.fastinsight import import_graph_from_fastinsight

    graph = import_graph_from_fastinsight(source_path)
    return serve_graph(
        graph,
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms,
        default_hops=default_hops,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Serve GraphContainer live visualizer on localhost.")
    parser.add_argument("--source", required=True, help="FastInsight source directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-ms", type=int, default=600)
    parser.add_argument("--hops", type=int, default=1)
    args = parser.parse_args()

    visualizer = serve_fastinsight(
        args.source,
        host=args.host,
        port=args.port,
        poll_interval_ms=args.poll_ms,
        default_hops=args.hops,
    )
    print(f"Visualizer URL: {visualizer.url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        visualizer.stop()


if __name__ == "__main__":
    _main()
