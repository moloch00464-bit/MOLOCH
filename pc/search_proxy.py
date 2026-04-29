"""MOLOCH Search-Proxy (PC-Side, Welle-5 Web-Recherche-Pfad).

FastAPI on :11650. DuckDuckGo HTML-Scrape, kein API-Key.

Endpoint:
  POST /search  body={"query": str, "max_results": int=5}
                resp={"query": str, "results": [{title, snippet, url}], "duration_ms": int}
  GET  /health  resp={"status": "ok", "service": "moloch-search-proxy"}

Gedacht als Tool-Output-Lieferant fuer Pi-Bridge bei prompt_type=web_research.
Pi ruft via http://192.168.178.20:11650/search, kriegt JSON-Liste, prepended
das in den System-Prompt fuer Tentakel (dolphin-llama3:8b kann tool-calls).

Reboot-persistent via pc/install_search_proxy_task.bat (Scheduled Task AtLogOn).
Run-Wrapper (silent): pc/run_search_proxy_hidden.vbs.

NEVER commit api_keys, NEVER ohne timeout=15 zu DuckDuckGo. RateLimit clientseitig:
180s Cooldown pro identischer query (kein Hammer auf DDG).
"""
import logging
import os
import re
import time
import urllib.parse
from collections import OrderedDict
from typing import Optional

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("search-proxy")

HOST = os.environ.get("MOLOCH_SEARCH_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_SEARCH_PORT", "11650"))
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MOLOCH-SearchProxy/1.0"
TIMEOUT_SEC = 15
DEFAULT_MAX_RESULTS = 5
HARD_MAX_RESULTS = 10
COOLDOWN_SEC = 180
CACHE_SIZE = 64

app = FastAPI(title="MOLOCH Search-Proxy", version="1.0")

_cache: "OrderedDict[str, tuple[float, list[dict]]]" = OrderedDict()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=400)
    max_results: int = Field(DEFAULT_MAX_RESULTS, ge=1, le=HARD_MAX_RESULTS)


class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    duration_ms: int
    cached: bool


def _scrape_ddg(query: str, max_results: int) -> list[dict]:
    """Scrape DuckDuckGo HTML — top results als list of dicts."""
    payload = {"q": query, "kl": "de-de"}
    headers = {"User-Agent": USER_AGENT}
    r = requests.post(DDG_HTML_URL, data=payload, headers=headers, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results: list[dict] = []
    for hit in soup.select("div.result")[: max_results * 2]:
        title_tag = hit.select_one("a.result__a")
        snippet_tag = hit.select_one("a.result__snippet") or hit.select_one(
            "div.result__snippet"
        )
        if not title_tag:
            continue
        href = title_tag.get("href", "")
        # DDG redirect-URLs entpacken: /l/?uddg=<encoded-real-url>
        m = re.search(r"uddg=([^&]+)", href)
        real_url = urllib.parse.unquote(m.group(1)) if m else href
        title = title_tag.get_text(strip=True)
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        if not title or not real_url:
            continue
        results.append({"title": title, "snippet": snippet, "url": real_url})
        if len(results) >= max_results:
            break
    return results


def _cache_get(key: str) -> Optional[list[dict]]:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, results = entry
    if time.time() - ts > COOLDOWN_SEC:
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return results


def _cache_put(key: str, results: list[dict]) -> None:
    _cache[key] = (time.time(), results)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


@app.get("/health")
def health():
    return {"status": "ok", "service": "moloch-search-proxy", "cache_size": len(_cache)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    key = f"{req.query.lower().strip()}|{req.max_results}"
    cached = _cache_get(key)
    if cached is not None:
        logger.info(f"[search] cache-hit query={req.query!r}")
        return SearchResponse(
            query=req.query,
            results=cached,
            duration_ms=int((time.time() - t0) * 1000),
            cached=True,
        )
    try:
        results = _scrape_ddg(req.query, req.max_results)
    except requests.Timeout:
        raise HTTPException(504, "DuckDuckGo timeout")
    except requests.RequestException as e:
        raise HTTPException(502, f"DuckDuckGo unreachable: {e}")
    if not results:
        raise HTTPException(404, "Keine Ergebnisse")
    _cache_put(key, results)
    duration_ms = int((time.time() - t0) * 1000)
    logger.info(f"[search] {len(results)} results in {duration_ms}ms query={req.query!r}")
    return SearchResponse(
        query=req.query, results=results, duration_ms=duration_ms, cached=False
    )


def main():
    logger.info(f"MOLOCH Search-Proxy startet auf {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
