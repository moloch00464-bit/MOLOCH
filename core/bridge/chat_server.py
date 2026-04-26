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
from fastapi.responses import HTMLResponse
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


_CHAT_UI_HTML = """<!doctype html>
<html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOLOCH</title>
<style>
  :root{--bg:#0d0d0f;--fg:#e6e6e6;--accent:#9b3030;--mute:#6e6e7a;--card:#16161b;--border:#26262e;}
  *{box-sizing:border-box}html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;height:100%}
  .wrap{display:flex;flex-direction:column;height:100vh;max-width:760px;margin:0 auto;padding:14px;gap:10px}
  .head{display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid var(--border)}
  .head h1{font:600 16px/1 system-ui;margin:0;letter-spacing:.5px}
  .head .meta{color:var(--mute);font-size:12px}
  .chat{flex:1;overflow-y:auto;padding:8px 4px;display:flex;flex-direction:column;gap:8px}
  .msg{padding:9px 12px;border-radius:10px;max-width:85%;white-space:pre-wrap;word-wrap:break-word}
  .me{align-self:flex-end;background:#1c2233;border:1px solid #2a3550}
  .moloch{align-self:flex-start;background:var(--card);border:1px solid var(--border)}
  .meta-line{font-size:11px;color:var(--mute);margin-top:3px}
  .form{display:flex;gap:8px;border-top:1px solid var(--border);padding-top:10px}
  textarea{flex:1;background:var(--card);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:10px;resize:none;font:14px system-ui;min-height:60px;max-height:160px}
  button{background:var(--accent);color:white;border:0;border-radius:8px;padding:0 16px;cursor:pointer;font:600 14px system-ui}
  button:disabled{opacity:.5;cursor:not-allowed}
  .row{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--mute);margin-top:6px}
  .row label{cursor:pointer}
  .err{color:#ff7676}
</style></head><body><div class="wrap">
<div class="head">
  <h1>MOLOCH — PIGH0ST</h1>
  <div class="meta" id="status">…</div>
</div>
<div class="chat" id="chat"></div>
<div class="form">
  <textarea id="inp" placeholder="Schreib was. Enter = senden, Shift+Enter = Zeile." autofocus></textarea>
  <button id="send">Senden</button>
</div>
<div class="row">
  <label><input type="checkbox" id="local"> NPU lokal (qwen2.5)</label>
  <label><input type="checkbox" id="reason"> reason_internal</label>
  <span id="err" class="err"></span>
</div>
</div>
<script>
const chat=document.getElementById("chat"),inp=document.getElementById("inp"),btn=document.getElementById("send"),
      st=document.getElementById("status"),err=document.getElementById("err"),
      cbLocal=document.getElementById("local"),cbReason=document.getElementById("reason");
function add(role,text,meta){const d=document.createElement("div");d.className="msg "+(role==="me"?"me":"moloch");
  d.textContent=text;chat.appendChild(d);
  if(meta){const m=document.createElement("div");m.className="meta-line "+(role==="me"?"me":"moloch");m.textContent=meta;chat.appendChild(m);}
  chat.scrollTop=chat.scrollHeight;}
async function refreshStatus(){try{const r=await fetch("/status");const j=await r.json();
  st.textContent=`mode: ${j.llm_mode} · last: ${j.last_provider} · ${j.request_count} req`;
}catch(e){st.textContent="status fetch failed";}}
async function loadHistory(){try{const r=await fetch("/history?n=10");const j=await r.json();
  for(const m of (j.messages||[])){const sender=m.sender==="user"?"me":"moloch";add(sender,m.text||"",`${m.ts||""} · ${m.source||""}`);}
}catch(e){}}
async function send(){const t=inp.value.trim();if(!t)return;err.textContent="";btn.disabled=true;
  add("me",t);inp.value="";const t0=Date.now();
  try{const r=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:t,force_local:cbLocal.checked,use_reason:cbReason.checked})});
    if(!r.ok){const txt=await r.text();throw new Error(`HTTP ${r.status}: ${txt}`);}
    const j=await r.json();
    add("moloch",j.text,`${j.provider} · ${j.duration_ms}ms`);
  }catch(e){err.textContent=e.message;add("moloch","[Fehler] "+e.message);}
  finally{btn.disabled=false;refreshStatus();inp.focus();}}
btn.onclick=send;
inp.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
refreshStatus();loadHistory();setInterval(refreshStatus,15000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    """Pi-lokales Browser-Chat-Fenster — keine externen Abhaengigkeiten."""
    return _CHAT_UI_HTML


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

    # Character Journal: Konversation als charakter-formenden Event protokollieren
    try:
        from core.memory.character_journal import get_journal
        get_journal().write_event(
            type="chat",
            interpretation=f"Markus: {req.text[:80]}",
            context="src=chat_server",
        )
    except Exception as e:
        logger.debug(f"Journal user-hook Fehler: {e}")

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

    # Character Journal: Eigene Antwort als charakter-formenden Event protokollieren
    try:
        from core.memory.character_journal import get_journal
        get_journal().write_event(
            type="chat",
            interpretation=f"Moloch: {out[:80]}",
            context=f"provider={b._last_provider}",
        )
    except Exception as e:
        logger.debug(f"Journal moloch-hook Fehler: {e}")

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
