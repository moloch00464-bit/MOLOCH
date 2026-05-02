"""W21 Tools — Web (Search + Fetch via PC Search-Proxy)."""
from __future__ import annotations
import logging
import requests
from typing import Any, Dict

logger = logging.getLogger("agent.tools.web")
SEARCH_PROXY_BASE = "http://192.168.178.20:11650"


def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    try:
        r = requests.post(
            f"{SEARCH_PROXY_BASE}/search",
            json={"query": query, "max_results": max_results},
            timeout=15,
        )
        if r.ok:
            return r.json()
        return {"error": f"http_{r.status_code}", "results": []}
    except Exception as e:
        return {"error": str(e)[:200], "results": []}


def web_fetch(url: str, max_chars: int = 8000) -> Dict[str, Any]:
    try:
        r = requests.post(
            f"{SEARCH_PROXY_BASE}/fetch",
            json={"url": url, "max_chars": max_chars},
            timeout=25,
        )
        if r.ok:
            return r.json()
        return {"error": f"http_{r.status_code}", "text": ""}
    except Exception as e:
        return {"error": str(e)[:200], "text": ""}
