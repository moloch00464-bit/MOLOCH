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
FETCH_TIMEOUT_SEC = 20
FETCH_HARD_MAX_CHARS = 50000
FETCH_DEFAULT_MAX_CHARS = 8000
FETCH_CACHE_SIZE = 32

app = FastAPI(title="MOLOCH Search-Proxy", version="1.2")

_cache: "OrderedDict[str, tuple[float, list[dict]]]" = OrderedDict()
_fetch_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()

# Stats-State (in-memory, reset on service-restart). Audit-relevant:
# Beweist ob Pi-Routing den Search-Proxy fuer prompt_type=web wirklich anruft.
_stats = {
    "started_at": time.time(),
    # /search
    "request_count": 0,        # Anzahl /search-Aufrufe (cache-hit + miss)
    "cache_hit_count": 0,
    "cache_miss_count": 0,
    "error_count": 0,
    "last_call_ts": None,      # epoch-sec
    "last_query": None,        # max 200 chars
    "last_result_count": None,
    # /fetch (Welle 20a)
    "fetch_count": 0,
    "fetch_cache_hit": 0,
    "fetch_cache_miss": 0,
    "fetch_error_count": 0,
    "last_fetch_ts": None,
    "last_fetch_url": None,
    "last_fetch_chars": None,
}


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


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    max_chars: int = Field(FETCH_DEFAULT_MAX_CHARS, ge=200, le=FETCH_HARD_MAX_CHARS)


class FetchResponse(BaseModel):
    url: str
    final_url: str
    title: str
    text: str
    chars: int
    truncated: bool
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


def _fetch_cache_get(url: str) -> Optional[dict]:
    entry = _fetch_cache.get(url)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > COOLDOWN_SEC:
        _fetch_cache.pop(url, None)
        return None
    _fetch_cache.move_to_end(url)
    return data


def _fetch_cache_put(url: str, data: dict) -> None:
    _fetch_cache[url] = (time.time(), data)
    while len(_fetch_cache) > FETCH_CACHE_SIZE:
        _fetch_cache.popitem(last=False)


def _extract_text(html: str, max_chars: int) -> tuple[str, str, bool]:
    """HTML -> (title, plaintext, truncated)."""
    soup = BeautifulSoup(html, "html.parser")
    # script/style etc. raus
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    # Heuristik: bevorzuge <main>, <article>, sonst body
    main = soup.find("main") or soup.find("article") or soup.body or soup
    raw = main.get_text(separator="\n", strip=True)
    # Mehrfache Newlines kollabieren
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[ \t]+", " ", text)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n... [truncated]"
    return title, text, truncated


def _do_fetch(url: str, max_chars: int) -> dict:
    """Fetch URL, parse, return dict (final_url, title, text, chars, truncated)."""
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "de-DE,de;q=0.9,en;q=0.6"}
    r = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT_SEC, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
        raise ValueError(f"unsupported content-type: {ctype}")
    title, text, truncated = _extract_text(r.text, max_chars)
    return {
        "final_url": str(r.url),
        "title": title,
        "text": text,
        "chars": len(text),
        "truncated": truncated,
    }


def _stats_record_call(query: str, result_count: Optional[int], cached: bool, error: bool) -> None:
    _stats["request_count"] += 1
    _stats["last_call_ts"] = time.time()
    _stats["last_query"] = query[:200]
    _stats["last_result_count"] = result_count
    if error:
        _stats["error_count"] += 1
    elif cached:
        _stats["cache_hit_count"] += 1
    else:
        _stats["cache_miss_count"] += 1


@app.get("/health")
def health():
    return {"status": "ok", "service": "moloch-search-proxy", "cache_size": len(_cache)}


@app.get("/stats")
def stats():
    """Audit-Endpoint. Zeigt ob Search-Proxy aktiv genutzt wird."""
    last_ts = _stats["last_call_ts"]
    last_fetch = _stats["last_fetch_ts"]
    return {
        **_stats,
        "uptime_sec": int(time.time() - _stats["started_at"]),
        "seconds_since_last_call": int(time.time() - last_ts) if last_ts else None,
        "seconds_since_last_fetch": int(time.time() - last_fetch) if last_fetch else None,
        "cache_size": len(_cache),
        "fetch_cache_size": len(_fetch_cache),
    }


@app.post("/fetch", response_model=FetchResponse)
def fetch(req: FetchRequest):
    """URL -> Plain-Text. Browser-Like-Behavior fuer Welle 20a.

    Kein Click/Navigate, aber HTTP-GET mit Redirect-Follow + HTML->Text.
    Pi-Specialist-Router ruft das wenn user_query eine URL enthaelt.
    """
    t0 = time.time()
    url = req.url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    cache_key = f"{url}|{req.max_chars}"
    cached_data = _fetch_cache_get(cache_key)
    if cached_data is not None:
        logger.info(f"[fetch] cache-hit url={url!r}")
        _stats["fetch_count"] += 1
        _stats["fetch_cache_hit"] += 1
        _stats["last_fetch_ts"] = time.time()
        _stats["last_fetch_url"] = url[:200]
        _stats["last_fetch_chars"] = cached_data.get("chars")
        return FetchResponse(
            url=url, **cached_data,
            duration_ms=int((time.time() - t0) * 1000),
            cached=True,
        )
    try:
        data = _do_fetch(url, req.max_chars)
    except requests.Timeout:
        _stats["fetch_count"] += 1
        _stats["fetch_error_count"] += 1
        _stats["last_fetch_ts"] = time.time()
        _stats["last_fetch_url"] = url[:200]
        raise HTTPException(504, f"fetch timeout: {url}")
    except (requests.RequestException, ValueError) as e:
        _stats["fetch_count"] += 1
        _stats["fetch_error_count"] += 1
        _stats["last_fetch_ts"] = time.time()
        _stats["last_fetch_url"] = url[:200]
        raise HTTPException(502, f"fetch failed: {e}")
    _fetch_cache_put(cache_key, data)
    duration_ms = int((time.time() - t0) * 1000)
    logger.info(f"[fetch] {data['chars']} chars in {duration_ms}ms url={url!r}")
    _stats["fetch_count"] += 1
    _stats["fetch_cache_miss"] += 1
    _stats["last_fetch_ts"] = time.time()
    _stats["last_fetch_url"] = url[:200]
    _stats["last_fetch_chars"] = data["chars"]
    return FetchResponse(
        url=url, **data,
        duration_ms=duration_ms,
        cached=False,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    key = f"{req.query.lower().strip()}|{req.max_results}"
    cached = _cache_get(key)
    if cached is not None:
        logger.info(f"[search] cache-hit query={req.query!r}")
        _stats_record_call(req.query, len(cached), cached=True, error=False)
        return SearchResponse(
            query=req.query,
            results=cached,
            duration_ms=int((time.time() - t0) * 1000),
            cached=True,
        )
    try:
        results = _scrape_ddg(req.query, req.max_results)
    except requests.Timeout:
        _stats_record_call(req.query, None, cached=False, error=True)
        raise HTTPException(504, "DuckDuckGo timeout")
    except requests.RequestException as e:
        _stats_record_call(req.query, None, cached=False, error=True)
        raise HTTPException(502, f"DuckDuckGo unreachable: {e}")
    if not results:
        _stats_record_call(req.query, 0, cached=False, error=True)
        raise HTTPException(404, "Keine Ergebnisse")
    _cache_put(key, results)
    duration_ms = int((time.time() - t0) * 1000)
    logger.info(f"[search] {len(results)} results in {duration_ms}ms query={req.query!r}")
    _stats_record_call(req.query, len(results), cached=False, error=False)
    return SearchResponse(
        query=req.query, results=results, duration_ms=duration_ms, cached=False
    )


def main():
    logger.info(f"MOLOCH Search-Proxy startet auf {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
