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
from core.longterm_memory import get_memory
from core.moloch_event_bus import get_event_bus

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
    # User-Input ins gemeinsame Memory + EventBus (Browser-Chat synchron mit Voice)
    try:
        get_memory().save_message("user", req.text, source="chat_server")
        get_event_bus().publish(
            "conversation.user_said",
            {"text": req.text, "source": "chat_server"},
            source="chat_server", priority=5,
        )
    except Exception as e:
        logger.warning(f"Memory/Bus user-write Fehler: {e}")

    b = get_llm_bridge()
    t0 = time.monotonic()
    # Browser-Chat-UI: PC=Hauptgehirn -> force_tentacle=True (KEIN qwen-Fallback fuers Reden).
    # Markus' Direktive: NPU-qwen ist nur fuer Befehle, nicht fuer Konversation.
    # Wenn force_local=True (User-Override aus UI): NPU wird genommen.
    if req.use_reason:
        out = b.reason_internal(req.text)
    else:
        out = b.ask_external(req.text, force_local=req.force_local,
                             force_tentacle=not req.force_local)
    dur_ms = int((time.monotonic() - t0) * 1000)
    if out is None:
        # Tentakel-offline-Fall (force_tentacle ohne PC erreichbar): ehrliche Meldung
        if b._last_provider == "tentacle_offline":
            raise HTTPException(
                503,
                "Rechner aus oder Tentakel nicht erreichbar — Moloch kann gerade nicht reden. "
                "Wenn der Rechner laeuft: Ollama-Service pruefen."
            )
        raise HTTPException(503, "Bridge gibt None (Stille)")

    # Moloch-Antwort ins gemeinsame Memory + EventBus
    try:
        get_memory().save_message("moloch", out, source="chat_server")
        get_event_bus().publish(
            "conversation.moloch_said",
            {"text": out, "source": "chat_server", "provider": b._last_provider},
            source="chat_server", priority=5,
        )
    except Exception as e:
        logger.warning(f"Memory/Bus moloch-write Fehler: {e}")

    return {"text": out, "provider": b._last_provider, "duration_ms": dur_ms}




@app.get("/history")
def history(n: int = 20):
    """Letzte N Konversations-Turns aus persistentem Memory (Cross-Channel:
    Browser-Chats UND Pi-Voice UND Test-Calls — alles zusammen)."""
    try:
        from core.longterm_memory import get_memory
        msgs = get_memory().get_recent_messages(n=n) or []
        return {"count": len(msgs), "messages": msgs}
    except Exception as e:
        raise HTTPException(500, f"Memory-Lesefehler: {e}")

def main():
    logger.info(f"MOLOCH Chat-Server startet auf {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
