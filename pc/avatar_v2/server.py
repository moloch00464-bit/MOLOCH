"""MOLOCH Avatar 2.0 Server — FastAPI auf :11801.

Liefert:
  GET  /            HTML-Loader (Three.js + WebGL)
  GET  /static/...  Static Assets (JS, CSS, SVG, Modelle)
  GET  /api/state   Aggregierter State von Pi + PC
  GET  /health      Service-Status

Quelle: Pi `/api/state/current` (kommt mit Phase 1 Pi-Side) ODER Fallback `/state_full`.
Polling: Browser pollt /api/state alle 200ms via state_polling.js.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("avatar-v2")

HOST = os.environ.get("MOLOCH_AVATAR_V2_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_AVATAR_V2_PORT", "11801"))
PI_BASE = os.environ.get("MOLOCH_PI_BASE", "http://192.168.178.30:9100")
PI_TUNNEL_URL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000")
PROXY_URL = os.environ.get("MOLOCH_PROXY_URL", "http://localhost:11600")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"MOLOCH Avatar 2.0 startet auf {HOST}:{PORT}")
    logger.info(f"Pi-Source: {PI_BASE} (Fallback: {PI_TUNNEL_URL})")
    yield


app = FastAPI(title="MOLOCH Avatar 2.0", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9000",
        "http://192.168.178.20:9000",
        "http://192.168.178.30:9100",
        "https://192.168.178.30:9443",
        "http://localhost:11800",  # Legacy-Avatar-Compatibility
    ],
    allow_methods=["GET"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _safe_get(client: httpx.AsyncClient, url: str, timeout: float = 2.0):
    """NEVER 5: timeout=Pflicht. Returnt (data, dt_ms)."""
    t0 = time.time()
    try:
        r = await client.get(url, timeout=timeout)
        dt = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return None, dt
        try:
            return r.json(), dt
        except Exception:
            return None, dt
    except Exception:
        return None, int((time.time() - t0) * 1000)


@app.get("/", response_class=HTMLResponse)
def root() -> FileResponse:
    """HTML-Loader fuer Three.js Avatar."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/state")
async def api_state():
    """Aggregierter State.

    Versucht zuerst Pi /api/state/current (Phase 1 Endpoint, kommt von Pi-Opus).
    Fallback auf /state_full (existiert schon). Plus PC-Adapter-Health.

    IMPORTANT FIX (Code-Review #3): Default-Timeout auf Client-Level.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
        # Phase-1-Endpoint (kommt mit Pi-Side state_engine.py)
        pi_state, pi_dt = await _safe_get(
            client, f"{PI_BASE}/api/state/current", timeout=2.0
        )
        # Fallback auf bestehenden Endpoint
        if pi_state is None:
            pi_state, pi_dt = await _safe_get(
                client, f"{PI_BASE}/state_full", timeout=2.0
            )
        pc_health, pc_dt = await _safe_get(client, f"{PROXY_URL}/health")

    return {
        "ts": time.time(),
        "pi": {
            "online": pi_state is not None,
            "latency_ms": pi_dt,
            "state": pi_state,
        },
        "pc": {
            "online": pc_health is not None,
            "latency_ms": pc_dt,
            "health": pc_health,
        },
    }


@app.get("/health")
def health():
    """Service-Health-Check."""
    return {
        "status": "ok",
        "service": "moloch-avatar-v2",
        "version": "2.0.0",
        "port": PORT,
        "pi_base": PI_BASE,
    }


def main() -> None:
    logger.info(f"MOLOCH Avatar 2.0 startet auf {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
