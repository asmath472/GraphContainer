from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib import error, parse, request


class VisualizerHTTPError(RuntimeError):
    def __init__(self, message: str, *, status: Optional[int] = None, url: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.url = url


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}{path}"
    data = None
    headers: Dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as exc:
        detail = ""
        try:
            raw = exc.read()
            if raw:
                detail = raw.decode("utf-8")
        except Exception:
            detail = ""
        msg = f"HTTP {exc.code} {method} {url}"
        if detail:
            msg = f"{msg}: {detail}"
        raise VisualizerHTTPError(msg, status=exc.code, url=url) from exc
    except error.URLError as exc:
        raise VisualizerHTTPError(f"Request failed for {url}: {exc}", url=url) from exc


def health(base_url: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    return _request_json(base_url, "/api/health", timeout=timeout)


def get_config(base_url: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    return _request_json(base_url, "/api/config", timeout=timeout)


def list_sessions(base_url: str, *, timeout: float = 5.0) -> List[str]:
    resp = _request_json(base_url, "/api/sessions", timeout=timeout)
    sessions = resp.get("sessions", [])
    if not isinstance(sessions, list):
        return []
    return [str(s) for s in sessions]


def create_session(
    base_url: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> str:
    resp = _request_json(
        base_url,
        "/api/session",
        method="POST",
        payload={"metadata": metadata or {}},
        timeout=timeout,
    )
    sid = resp.get("session_id")
    if not sid:
        raise VisualizerHTTPError("session_id not returned by visualizer server")
    return str(sid)


def get_session_snapshot(base_url: str, session_id: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    session_id = parse.quote(str(session_id), safe="")
    return _request_json(base_url, f"/api/session/{session_id}", timeout=timeout)


def get_session_subgraph(
    base_url: str,
    session_id: str,
    *,
    hops: int = 2,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    session_id = parse.quote(str(session_id), safe="")
    return _request_json(base_url, f"/api/session/{session_id}/subgraph?hops={int(hops)}", timeout=timeout)


def update_session(
    base_url: str,
    session_id: str,
    *,
    nodes: Optional[Sequence[Any]] = None,
    edges: Optional[Sequence[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    progress: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if nodes is not None:
        payload["nodes"] = list(nodes)
    if edges is not None:
        payload["edges"] = list(edges)
    if metadata is not None:
        payload["metadata"] = metadata
    if progress is not None:
        payload["progress"] = progress

    session_id = parse.quote(str(session_id), safe="")
    return _request_json(
        base_url,
        f"/api/session/{session_id}/update",
        method="POST",
        payload=payload,
        timeout=timeout,
    )


def set_progress(
    base_url: str,
    session_id: str,
    *,
    current: int,
    total: int,
    message: str = "",
    timeout: float = 5.0,
) -> Dict[str, Any]:
    return update_session(
        base_url,
        session_id,
        progress={
            "current": int(current),
            "total": int(total),
            "message": message,
        },
        timeout=timeout,
    )


def clear_session(base_url: str, session_id: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    session_id = parse.quote(str(session_id), safe="")
    return _request_json(
        base_url,
        f"/api/session/{session_id}/clear",
        method="POST",
        payload={},
        timeout=timeout,
    )


def delete_session(base_url: str, session_id: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    session_id = parse.quote(str(session_id), safe="")
    return _request_json(base_url, f"/api/session/{session_id}", method="DELETE", timeout=timeout)


@dataclass
class LiveVisualizerClient:

    def __init__(self, base_url: str = "http://0.0.0.0:8765", timeout: float = 5.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        return health(self.base_url, timeout=self.timeout)

    def get_config(self) -> Dict[str, Any]:
        return get_config(self.base_url, timeout=self.timeout)

    def list_sessions(self) -> List[str]:
        return list_sessions(self.base_url, timeout=self.timeout)

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        return create_session(self.base_url, metadata=metadata, timeout=self.timeout)

    def get_session_snapshot(self, session_id: str) -> Dict[str, Any]:
        return get_session_snapshot(self.base_url, session_id, timeout=self.timeout)

    def get_session_subgraph(self, session_id: str, *, hops: int = 2) -> Dict[str, Any]:
        return get_session_subgraph(self.base_url, session_id, hops=hops, timeout=self.timeout)

    def update_session(
        self,
        session_id: str,
        *,
        nodes: Optional[Sequence[Any]] = None,
        edges: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return update_session(
            self.base_url,
            session_id,
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            progress=progress,
            timeout=self.timeout,
        )

    def set_progress(self, session_id: str, *, current: int, total: int, message: str = "") -> Dict[str, Any]:
        return set_progress(
            self.base_url,
            session_id,
            current=current,
            total=total,
            message=message,
            timeout=self.timeout,
        )

    def clear_session(self, session_id: str) -> Dict[str, Any]:
        return clear_session(self.base_url, session_id, timeout=self.timeout)

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        return delete_session(self.base_url, session_id, timeout=self.timeout)
