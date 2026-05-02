"""W22 Browser-Tools (Welle 22) — wraps PC-Side browser_proxy auf :11680."""
from __future__ import annotations
import logging
import requests
from typing import Any, Dict

logger = logging.getLogger("agent.tools.browser")
BROWSER_PROXY_BASE = "http://192.168.178.20:11680"


def browser_open(url: str) -> Dict[str, Any]:
    """Oeffne URL im PC-Browser (Playwright). Returnt {ok, url, title} oder {error}."""
    try:
        r = requests.post(f"{BROWSER_PROXY_BASE}/open",
                          json={"url": url},
                          timeout=20)
        if r.ok:
            return r.json()
        return {"error": f"http_{r.status_code}", "url": url}
    except Exception as e:
        return {"error": str(e)[:200], "url": url}


def browser_click(selector: str) -> Dict[str, Any]:
    """Click auf CSS/XPath-Selector im aktuellen Browser-Tab."""
    try:
        r = requests.post(f"{BROWSER_PROXY_BASE}/click",
                          json={"selector": selector},
                          timeout=10)
        if r.ok:
            return r.json()
        return {"error": f"http_{r.status_code}", "selector": selector}
    except Exception as e:
        return {"error": str(e)[:200], "selector": selector}


def browser_screenshot(full_page: bool = False) -> Dict[str, Any]:
    """Screenshot der aktuellen Seite. Returnt {path, size_kb, ...} oder {error}."""
    try:
        r = requests.post(f"{BROWSER_PROXY_BASE}/screenshot",
                          json={"full_page": bool(full_page)},
                          timeout=15)
        if r.ok:
            return r.json()
        return {"error": f"http_{r.status_code}"}
    except Exception as e:
        return {"error": str(e)[:200]}
