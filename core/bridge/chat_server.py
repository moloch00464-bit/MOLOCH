#!/usr/bin/env python3
"""
MOLOCH Chat-Server (Pi-Side)
=============================
FastAPI Port 9100. Wrapper um local_llm_bridge fuer PC-Chat-UI.

Endpoints:
  GET  /health
  GET  /status     -> Bridge-Stats
  POST /chat       -> {text, force_local?, use_reason?} -> {text, provider, duration_ms}
"""
import logging
import os
import sys
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.expanduser("~/moloch"))
from core.autonomy.local_llm_bridge import get_llm_bridge, _load_tentacle_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chat-server")

HOST = os.environ.get("MOLOCH_CHAT_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_CHAT_PORT", "9100"))

app = FastAPI(title="MOLOCH Chat-Server", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    force_local: bool = False
    use_reason: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "service": "moloch-chat-server"}


@app.get("/status")
def status():
    b = get_llm_bridge()
    cfg = _load_tentacle_cfg()
    return {
        "llm_mode": b._llm_mode,
        "ollama_available": b._ollama_available,
        "last_provider": b._last_provider,
        "request_count": b._request_count,
        "tentacle": {
            "enabled": cfg.get("enabled"),
            "host": cfg.get("host"),
            "fail_count": b._tentacle_fail_count,
            "backoff_until": b._tentacle_backoff_until,
            "model_cached": b._tentacle_model_cached,
        },
    }


@app.post("/chat")
def chat(req: ChatRequest):
    b = get_llm_bridge()
    t0 = time.monotonic()
    if req.use_reason:
        out = b.reason_internal(req.text)
    else:
        out = b.ask_external(req.text, force_local=req.force_local)
    dur_ms = int((time.monotonic() - t0) * 1000)
    if out is None:
        raise HTTPException(503, "Bridge gibt None (Stille)")
    return {"text": out, "provider": b._last_provider, "duration_ms": dur_ms}


def main():
    logger.info(f"MOLOCH Chat-Server startet auf {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
