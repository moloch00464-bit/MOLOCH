"""MOLOCH TTS-Bridge (PC-Side, Punkt #26).

FastAPI :9002 mit Edge-TTS (Microsoft Edge, kostenlos, hochwertige deutsche Stimmen).
Pi ruft via http://192.168.178.20:9002/speak fuer bessere TTS als Piper.

Stimmen (deutsch):
  de-DE-KillianNeural   maennlich, neutral (default)
  de-DE-AmalaNeural     weiblich, warm
  de-DE-ConradNeural    maennlich, dunkel
  de-DE-KatjaNeural     weiblich, freundlich
  de-DE-FlorianMultilingualNeural  multilingual
  ... (siehe GET /voices fuer komplette Liste)

NEVER 5: HTTP timeouts ueber HTTPException + max_length im Request.
NEVER 8: kein shell=True, edge-tts ist Python-only.
API-Keys: keine — Edge-TTS nutzt kostenlose Microsoft-Edge-Backend-API.

Reboot-persistent via pc/run_tts_bridge_hidden.vbs (Startup-Folder-Shortcut).
"""
import asyncio
import logging
import os
import time
from typing import Optional

import edge_tts
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tts-bridge")

HOST = os.environ.get("MOLOCH_TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_TTS_PORT", "9002"))
DEFAULT_VOICE = os.environ.get("MOLOCH_TTS_VOICE", "de-DE-ConradNeural")

app = FastAPI(title="MOLOCH TTS-Bridge", version="1.0")

# Stats — Audit-relevant
_stats = {
    "started_at": time.time(),
    "request_count": 0,
    "error_count": 0,
    "last_call_ts": None,
    "last_text_preview": None,
    "last_voice": None,
    "last_audio_bytes": None,
    "total_chars": 0,
    "total_audio_bytes": 0,
}


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(DEFAULT_VOICE, max_length=80)
    rate: str = Field("+0%", max_length=10)
    pitch: str = Field("+0Hz", max_length=10)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "moloch-tts-bridge",
        "engine": "edge-tts",
        "default_voice": DEFAULT_VOICE,
        "format": "audio/mpeg",
    }


@app.get("/stats")
def stats():
    last = _stats["last_call_ts"]
    return {
        **_stats,
        "uptime_sec": int(time.time() - _stats["started_at"]),
        "seconds_since_last_call": int(time.time() - last) if last else None,
    }


@app.get("/voices")
def voices():
    """Liste deutsche + multilingual Stimmen."""
    async def _list():
        all_voices = await edge_tts.list_voices()
        de = [
            {
                "name": v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
                "friendly_name": v.get("FriendlyName", ""),
            }
            for v in all_voices
            if v["Locale"].startswith("de-") or "Multilingual" in v.get("ShortName", "")
        ]
        return de

    try:
        voices_list = asyncio.run(_list())
    except Exception as e:
        raise HTTPException(502, f"edge-tts list failed: {str(e)[:200]}")
    return {"count": len(voices_list), "voices": voices_list}


async def _synthesize(text: str, voice: str, rate: str, pitch: str) -> bytes:
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    chunks = []
    async for ch in comm.stream():
        if ch["type"] == "audio":
            chunks.append(ch["data"])
    return b"".join(chunks)


@app.post("/speak")
def speak(req: SpeakRequest):
    """Text -> MP3-Audio (audio/mpeg). Synchroner Wrapper um async edge-tts."""
    t0 = time.time()
    _stats["request_count"] += 1
    _stats["last_call_ts"] = t0
    _stats["last_text_preview"] = req.text[:200]
    _stats["last_voice"] = req.voice
    _stats["total_chars"] += len(req.text)

    try:
        audio_bytes = asyncio.run(
            _synthesize(req.text, req.voice, req.rate, req.pitch)
        )
    except Exception as e:
        _stats["error_count"] += 1
        logger.warning(f"[speak] edge-tts fail: {str(e)[:200]}")
        raise HTTPException(502, f"edge-tts failed: {str(e)[:200]}")

    if not audio_bytes:
        _stats["error_count"] += 1
        raise HTTPException(502, "edge-tts returned empty audio")

    duration_ms = int((time.time() - t0) * 1000)
    _stats["last_audio_bytes"] = len(audio_bytes)
    _stats["total_audio_bytes"] += len(audio_bytes)
    logger.info(
        f"[speak] {len(req.text)} chars -> {len(audio_bytes)} bytes "
        f"in {duration_ms}ms voice={req.voice}"
    )

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-MOLOCH-Duration-MS": str(duration_ms),
            "X-MOLOCH-Voice": req.voice,
            "X-MOLOCH-Audio-Bytes": str(len(audio_bytes)),
        },
    )


def main():
    logger.info(f"MOLOCH TTS-Bridge startet auf {HOST}:{PORT} (default voice: {DEFAULT_VOICE})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
