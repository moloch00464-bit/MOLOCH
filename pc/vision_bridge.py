"""MOLOCH Vision-Bridge (PC-Side, Welle 22 Punkt #10).

FastAPI :9003 — VLM-Schnittstelle fuer Browser-Screenshot-Analyse + Bildbeschreibung.
Stub-Status 2026-05-02: kein Modell installiert. Pi kann /describe rufen,
kriegt 503 mit Setup-Hinweis. Vision-Stack-Wahl steht aus:

| Option           | Disk    | Latenz CPU | Cost      |
|------------------|---------|------------|-----------|
| moondream2 lokal | ~2.5 GB | 5-15s      | $0        |
| Claude-Vision    | 0       | 2-5s       | $$$ (Pro) |
| OpenRouter VL2   | 0       | 3-8s       | $ (paygo) |

Default: keiner — Stub-Modus. Markus entscheidet via env MOLOCH_VISION_BACKEND.

Endpoints:
  GET  /health   - {status, backend, model_ready}
  GET  /stats    - request_count etc.
  POST /describe body {image_path or image_url, prompt} -> {text, model, duration_ms}

NEVER: keine API-Keys loggen, kein shell=True, Timeouts via HTTPException.
Reboot-persistent via run_vision_bridge_hidden.vbs (Startup-Folder).
"""
import logging
import os
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vision-bridge")

HOST = os.environ.get("MOLOCH_VISION_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_VISION_PORT", "9003"))
BACKEND = os.environ.get("MOLOCH_VISION_BACKEND", "stub")  # stub|moondream2|claude|openrouter

app = FastAPI(title="MOLOCH Vision-Bridge", version="0.1-stub")

_stats = {
    "started_at": time.time(),
    "request_count": 0,
    "error_count": 0,
    "last_call_ts": None,
    "last_image_source": None,
    "total_describe_calls": 0,
    "backend": BACKEND,
}

_model_loaded = False
_model_load_error: Optional[str] = None


class DescribeRequest(BaseModel):
    image_path: Optional[str] = Field(None, max_length=2000)
    image_url: Optional[str] = Field(None, max_length=2000)
    prompt: str = Field("Beschreibe das Bild auf Deutsch in 2-3 Saetzen.", max_length=500)


@app.get("/health")
async def health():
    return {
        "status": "stub" if BACKEND == "stub" else ("ok" if _model_loaded else "loading"),
        "service": "moloch-vision-bridge",
        "backend": BACKEND,
        "model_ready": _model_loaded,
        "version": "0.1-stub",
        "available_backends": ["stub", "moondream2", "claude", "openrouter"],
        "switch_via": "env MOLOCH_VISION_BACKEND",
    }


@app.get("/stats")
async def stats():
    last = _stats["last_call_ts"]
    return {
        **_stats,
        "uptime_sec": int(time.time() - _stats["started_at"]),
        "seconds_since_last_call": int(time.time() - last) if last else None,
        "model_ready": _model_loaded,
    }


@app.post("/describe")
async def describe(req: DescribeRequest):
    """Bild beschreiben. Stub-Modus liefert 503 bis Backend gewaehlt + Modell geladen."""
    _stats["request_count"] += 1
    _stats["last_call_ts"] = time.time()
    src = req.image_path or req.image_url or "(none)"
    _stats["last_image_source"] = src[:200]

    if not (req.image_path or req.image_url):
        _stats["error_count"] += 1
        raise HTTPException(400, "Need image_path or image_url")

    if BACKEND == "stub":
        _stats["error_count"] += 1
        raise HTTPException(
            503,
            "Vision-Backend not configured. Set env MOLOCH_VISION_BACKEND="
            "moondream2|claude|openrouter and install model. See README.",
        )

    # Backend-Implementations als Stubs — Pi-Cooperation noetig fuer Pi-Side-Tools
    if BACKEND == "moondream2":
        raise HTTPException(503, "moondream2 not yet wired up — install via 'pip install transformers torch' + download ~/moloch_pc_state/moondream2/")
    if BACKEND == "claude":
        raise HTTPException(503, "claude-vision not wired — needs ANTHROPIC_API_KEY in api_keys.json")
    if BACKEND == "openrouter":
        raise HTTPException(503, "openrouter-vl2 not wired — needs OPENROUTER_API_KEY")

    raise HTTPException(503, f"unknown backend: {BACKEND}")


def main():
    logger.info(
        f"MOLOCH Vision-Bridge startet auf {HOST}:{PORT} (backend: {BACKEND}, "
        f"stub-mode: {BACKEND == 'stub'})"
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
