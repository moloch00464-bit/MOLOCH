"""HTTP-Bridge zu Pi-Side Tool-Dispatcher (Welle 21 Phase 1 — Pi-Opus).

Pi-Endpoints (von Pi-Phase-1 erwartet, siehe task_welle20a_folgeissues_und_welle21_phase1_start):
  GET  http://192.168.178.30:9100/api/agent/tools     -> {tools: [...]}
  POST http://192.168.178.30:9100/api/agent/dispatch  -> {result, error}

Solange Pi-Phase-1 nicht live: MockBridge fuer lokale Tests.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import requests

logger = logging.getLogger("pi-bridge")

PI_BASE = "http://192.168.178.30:9100"
TOOLS_ENDPOINT = f"{PI_BASE}/api/agent/tools"
DISPATCH_ENDPOINT = f"{PI_BASE}/api/agent/dispatch"
DISPATCH_TIMEOUT = 30  # NEVER 5
CATALOG_TIMEOUT = 10


class ToolBridge(Protocol):
    def dispatch(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_catalog(self) -> List[Dict[str, Any]]: ...


class HttpBridge:
    """Echte Bridge zu Pi-Side. Wartet auf Pi-Phase-1-Endpoints."""

    def dispatch(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = requests.post(
                DISPATCH_ENDPOINT,
                json={"tool_name": tool_name, "params": params},
                timeout=DISPATCH_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
            return {"result": None, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"result": None, "error": str(e)[:200]}

    def get_catalog(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(TOOLS_ENDPOINT, timeout=CATALOG_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data.get("tools", [])
        except Exception as e:
            logger.warning(f"[catalog] Pi-Endpoint nicht erreichbar: {e}")
            return []


class MockBridge:
    """Lokal-Test-Bridge mit fake-Daten. Aktiv wenn Pi-Phase-1 noch nicht live."""

    MOCK_CATALOG: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "DuckDuckGo-Suche, returnt Top-Results mit URL/Title/Snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Such-Query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Holt URL, returnt plain-text-Inhalt der Seite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 8000},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spotify_top_artists",
                "description": "Markus' Top-Artists aus Spotify-Stats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "default": 20},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spotify_play",
                "description": "Spielt Song/Album/Playlist auf Spotify.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_mood",
                "description": "Aktuelle Tension/Zone/letzte-Reflektion aus audit_state.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    def __init__(self) -> None:
        # Lokaler search_proxy auf PC nutzen falls verfuegbar (Welle-19+20a live)
        self._sp_base = "http://localhost:11650"
        self._stats_path = Path("C:/Users/49179/moloch_repo/musik/spotify_stats.json")

    def dispatch(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "web_search":
                r = requests.post(
                    f"{self._sp_base}/search",
                    json={
                        "query": params.get("query", ""),
                        "max_results": int(params.get("max_results", 5)),
                    },
                    timeout=DISPATCH_TIMEOUT,
                )
                if r.status_code == 200:
                    return {"result": r.json(), "error": None}
                return {"result": None, "error": f"HTTP {r.status_code}"}

            if tool_name == "web_fetch":
                r = requests.post(
                    f"{self._sp_base}/fetch",
                    json={
                        "url": params.get("url", ""),
                        "max_chars": int(params.get("max_chars", 8000)),
                    },
                    timeout=DISPATCH_TIMEOUT,
                )
                if r.status_code == 200:
                    return {"result": r.json(), "error": None}
                return {"result": None, "error": f"HTTP {r.status_code}"}

            if tool_name == "spotify_top_artists":
                if not self._stats_path.exists():
                    return {"result": None, "error": "spotify_stats.json fehlt"}
                data = json.loads(self._stats_path.read_text(encoding="utf-8"))
                n = int(params.get("n", 20))
                return {"result": data.get("top_artists", [])[:n], "error": None}

            if tool_name == "spotify_play":
                # Mock — echter Call ginge ueber Pi-IPC. PC-Side hat das nicht.
                return {
                    "result": {"action": "spotify_play", "query": params.get("query"), "mock": True},
                    "error": "MockBridge: spotify_play braucht Pi-IPC (Phase 1 Pi-Opus)",
                }

            if tool_name == "get_mood":
                # Mock — echter Call ginge ueber Pi-Side audit_state.json
                return {
                    "result": {"tension": 0.3, "zone": "calm", "mock": True},
                    "error": "MockBridge: get_mood braucht Pi-Side state",
                }

            return {"result": None, "error": f"unknown tool: {tool_name}"}
        except Exception as e:
            return {"result": None, "error": str(e)[:200]}

    def get_catalog(self) -> List[Dict[str, Any]]:
        return list(self.MOCK_CATALOG)


def get_bridge(prefer_http: bool = True) -> ToolBridge:
    """Liefert HttpBridge wenn Pi-Endpoints erreichbar, sonst MockBridge."""
    if prefer_http:
        try:
            r = requests.get(TOOLS_ENDPOINT, timeout=3)
            if r.status_code == 200:
                logger.info("[bridge] Pi-Endpoints erreichbar -> HttpBridge")
                return HttpBridge()
        except Exception:
            pass
    logger.info("[bridge] Pi-Endpoints nicht erreichbar -> MockBridge")
    return MockBridge()
