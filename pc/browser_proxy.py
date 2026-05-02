"""MOLOCH Browser-Proxy (PC-Side, Welle 22 #8).

Headless-Chromium via Playwright auf :11680. Pi-Cloud-LLM (DeepSeek) ruft via
http://192.168.178.20:11680/* fuer JS-rendered Web-Recherche, Click/Scroll/Type
und Screenshots — echtes Browser-Verhalten, kein Lynx-Niveau wie /fetch.

Endpoints:
  GET  /health
  GET  /stats
  POST /open       body {url, wait_until} -> {tab_id, title, url}
  POST /click      body {tab_id, selector}
  POST /scroll     body {tab_id, delta_y}
  POST /type       body {tab_id, selector, text}
  POST /screenshot body {tab_id, full_page} -> PNG bytes
  POST /text       body {tab_id} -> page-text
  POST /close      body {tab_id}

Browser-Pool: max 5 Tabs gleichzeitig (LRU eviction).
NEVER 5: HTTP-Timeouts via HTTPException. NEVER 8: kein shell=True.
NEVER 7: kein Runtime-State committed.
Reboot-persistent via pc/run_browser_proxy_hidden.vbs.

Sicherheit: Headless, kein Cookie-Persistenz default. KEIN Login auf Markus' Behalf.
"""
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from playwright.async_api import Browser, Page, Playwright, async_playwright
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("browser-proxy")

HOST = os.environ.get("MOLOCH_BROWSER_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_BROWSER_PORT", "11680"))
MAX_TABS = int(os.environ.get("MOLOCH_BROWSER_MAX_TABS", "5"))
DEFAULT_TIMEOUT_MS = 30000

_pages: Dict[str, Page] = {}
_pages_last_used: Dict[str, float] = {}
_playwright: Optional[Playwright] = None
_browser: Optional[Browser] = None

_stats = {
    "started_at": time.time(),
    "open_count": 0,
    "click_count": 0,
    "scroll_count": 0,
    "type_count": 0,
    "screenshot_count": 0,
    "text_count": 0,
    "close_count": 0,
    "error_count": 0,
    "last_call_ts": None,
    "last_url": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _playwright, _browser
    try:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
        logger.info("Chromium gestartet (headless)")
    except Exception as e:
        logger.error(f"Chromium-Start failed: {e}")
        _browser = None
    yield
    for tid, page in list(_pages.items()):
        try:
            await page.close()
        except Exception:
            pass
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
    if _playwright:
        try:
            await _playwright.stop()
        except Exception:
            pass


app = FastAPI(title="MOLOCH Browser-Proxy", version="1.0", lifespan=lifespan)


class OpenRequest(BaseModel):
    url: str = Field(..., max_length=2000)
    wait_until: str = Field("domcontentloaded", max_length=20)


class TabRequest(BaseModel):
    tab_id: str = Field(..., min_length=4, max_length=64)


class ClickRequest(TabRequest):
    selector: str = Field(..., max_length=500)


class ScrollRequest(TabRequest):
    delta_y: int = Field(500, ge=-50000, le=50000)


class TypeRequest(TabRequest):
    selector: str = Field(..., max_length=500)
    text: str = Field(..., max_length=5000)


class ScreenshotRequest(TabRequest):
    full_page: bool = False


def _stats_inc(key: str, url: Optional[str] = None):
    _stats[key] += 1
    _stats["last_call_ts"] = time.time()
    if url:
        _stats["last_url"] = url[:200]


async def _evict_lru():
    while len(_pages) > MAX_TABS:
        oldest = min(_pages_last_used.items(), key=lambda kv: kv[1])[0]
        page = _pages.pop(oldest, None)
        _pages_last_used.pop(oldest, None)
        if page:
            try:
                await page.close()
            except Exception:
                pass
            logger.info(f"[evict] tab={oldest} (lru)")


def _touch(tab_id: str):
    _pages_last_used[tab_id] = time.time()


@app.get("/health")
async def health():
    return {
        "status": "ok" if _browser else "error",
        "service": "moloch-browser-proxy",
        "engine": "playwright-chromium",
        "headless": True,
        "open_tabs": len(_pages),
        "max_tabs": MAX_TABS,
        "browser_ready": _browser is not None,
    }


@app.get("/stats")
async def stats():
    last = _stats["last_call_ts"]
    return {
        **_stats,
        "uptime_sec": int(time.time() - _stats["started_at"]),
        "seconds_since_last_call": int(time.time() - last) if last else None,
        "open_tabs": len(_pages),
    }


@app.post("/open")
async def open_url(req: OpenRequest):
    if _browser is None:
        raise HTTPException(503, "browser not initialized")
    if not (req.url.startswith("http://") or req.url.startswith("https://")):
        raise HTTPException(400, "url must start with http:// or https://")
    page = await _browser.new_page()
    tab_id = uuid.uuid4().hex[:12]
    try:
        await page.goto(req.url, wait_until=req.wait_until, timeout=DEFAULT_TIMEOUT_MS)
    except Exception as e:
        try:
            await page.close()
        except Exception:
            pass
        _stats["error_count"] += 1
        raise HTTPException(502, f"goto failed: {str(e)[:200]}")
    _pages[tab_id] = page
    _touch(tab_id)
    await _evict_lru()
    _stats_inc("open_count", url=req.url)
    title = await page.title()
    return {"tab_id": tab_id, "title": title, "url": page.url}


@app.post("/click")
async def click(req: ClickRequest):
    page = _pages.get(req.tab_id)
    if page is None:
        raise HTTPException(404, "tab not found")
    _touch(req.tab_id)
    try:
        await page.click(req.selector, timeout=DEFAULT_TIMEOUT_MS)
    except Exception as e:
        _stats["error_count"] += 1
        raise HTTPException(502, f"click failed: {str(e)[:200]}")
    _stats_inc("click_count")
    return {"ok": True, "url": page.url}


@app.post("/scroll")
async def scroll(req: ScrollRequest):
    page = _pages.get(req.tab_id)
    if page is None:
        raise HTTPException(404, "tab not found")
    _touch(req.tab_id)
    try:
        await page.evaluate(f"window.scrollBy(0, {int(req.delta_y)})")
    except Exception as e:
        _stats["error_count"] += 1
        raise HTTPException(502, f"scroll failed: {str(e)[:200]}")
    _stats_inc("scroll_count")
    return {"ok": True}


@app.post("/type")
async def type_text(req: TypeRequest):
    page = _pages.get(req.tab_id)
    if page is None:
        raise HTTPException(404, "tab not found")
    _touch(req.tab_id)
    try:
        await page.fill(req.selector, req.text, timeout=DEFAULT_TIMEOUT_MS)
    except Exception as e:
        _stats["error_count"] += 1
        raise HTTPException(502, f"type failed: {str(e)[:200]}")
    _stats_inc("type_count")
    return {"ok": True}


@app.post("/screenshot")
async def screenshot(req: ScreenshotRequest):
    page = _pages.get(req.tab_id)
    if page is None:
        raise HTTPException(404, "tab not found")
    _touch(req.tab_id)
    try:
        png = await page.screenshot(full_page=req.full_page)
    except Exception as e:
        _stats["error_count"] += 1
        raise HTTPException(502, f"screenshot failed: {str(e)[:200]}")
    _stats_inc("screenshot_count")
    return Response(
        content=png, media_type="image/png",
        headers={"X-MOLOCH-Url": page.url, "X-MOLOCH-Bytes": str(len(png))},
    )


@app.post("/text")
async def get_text(req: TabRequest):
    page = _pages.get(req.tab_id)
    if page is None:
        raise HTTPException(404, "tab not found")
    _touch(req.tab_id)
    try:
        text = await page.inner_text("body")
    except Exception as e:
        _stats["error_count"] += 1
        raise HTTPException(502, f"text failed: {str(e)[:200]}")
    _stats_inc("text_count")
    return {"text": text[:50000], "url": page.url, "title": await page.title()}


@app.post("/close")
async def close(req: TabRequest):
    page = _pages.pop(req.tab_id, None)
    _pages_last_used.pop(req.tab_id, None)
    if page is None:
        raise HTTPException(404, "tab not found")
    try:
        await page.close()
    except Exception:
        pass
    _stats_inc("close_count")
    return {"ok": True}


def main():
    logger.info(f"MOLOCH Browser-Proxy startet auf {HOST}:{PORT} (max {MAX_TABS} tabs)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
