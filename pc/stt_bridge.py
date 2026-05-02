"""MOLOCH STT-Bridge (PC-Side, Punkt #25).

FastAPI :9001 mit faster-whisper (CPU-only, int8-quantized).
Pi-WiFi-Mic ODER PC-Mikrofon -> Audio-Stream -> Transcript.

Default-Modell: small (244MB, ~3x realtime auf Ryzen 9 3900X CPU).
Modell wird lazy beim ersten /transcribe-Call geladen (HuggingFace-Cache).

Endpoints:
  GET  /health     - Service-Status + Model-Info
  GET  /stats      - request_count, last_call_ts (Audit)
  POST /transcribe - body: audio bytes (multipart/form-data) -> {text, language, duration_s, segments}

NEVER 5: HTTP timeouts. NEVER 8: kein shell=True.
Reboot-persistent via pc/run_stt_bridge_hidden.vbs.
"""
import logging
import os
import tempfile
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stt-bridge")

HOST = os.environ.get("MOLOCH_STT_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_STT_PORT", "9001"))
MODEL_SIZE = os.environ.get("MOLOCH_STT_MODEL", "medium")  # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.environ.get("MOLOCH_STT_COMPUTE", "int8")
DEFAULT_LANG = os.environ.get("MOLOCH_STT_LANG", "de")

app = FastAPI(title="MOLOCH STT-Bridge", version="1.0")

# Model-State (lazy-loaded)
_model = None
_model_load_attempted = False
_model_load_error: Optional[str] = None

# Stats — Audit-relevant
_stats = {
    "started_at": time.time(),
    "model_size": MODEL_SIZE,
    "compute_type": COMPUTE_TYPE,
    "request_count": 0,
    "error_count": 0,
    "last_call_ts": None,
    "last_text_preview": None,
    "last_audio_bytes": None,
    "last_duration_s": None,
    "total_audio_bytes": 0,
    "total_chars_out": 0,
}


def _get_model():
    """Lazy-Load des Modells. Gibt None zurueck bei Fehler (mit _model_load_error gesetzt)."""
    global _model, _model_load_attempted, _model_load_error
    if _model is not None:
        return _model
    if _model_load_attempted:
        return None  # vorheriger Load-Versuch fehlgeschlagen
    _model_load_attempted = True
    try:
        from faster_whisper import WhisperModel
        logger.info(f"[model] Lade faster-whisper {MODEL_SIZE} compute_type={COMPUTE_TYPE} ...")
        t0 = time.time()
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
        logger.info(f"[model] geladen in {time.time() - t0:.1f}s")
        return _model
    except Exception as e:
        _model_load_error = str(e)[:300]
        logger.error(f"[model] Load-Fehler: {_model_load_error}")
        return None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "moloch-stt-bridge",
        "engine": "faster-whisper",
        "model": MODEL_SIZE,
        "device": "cpu",
        "compute_type": COMPUTE_TYPE,
        "default_language": DEFAULT_LANG,
        "loaded": _model is not None,
        "model_load_error": _model_load_error,
    }


@app.get("/stats")
def stats():
    last = _stats["last_call_ts"]
    return {
        **_stats,
        "uptime_sec": int(time.time() - _stats["started_at"]),
        "seconds_since_last_call": int(time.time() - last) if last else None,
        "model_loaded": _model is not None,
    }


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration_s: float
    segments: list


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
):
    """Audio-Datei -> Transcript (deutsch by default).

    Akzeptiert: WAV, MP3, FLAC, OGG, etc. (alles was ffmpeg liest).
    """
    t0 = time.time()
    _stats["request_count"] += 1
    _stats["last_call_ts"] = t0

    # Audio in Temp-File speichern (faster-whisper nimmt Pfad)
    audio_bytes = await audio.read()
    if not audio_bytes:
        _stats["error_count"] += 1
        raise HTTPException(400, "audio file empty")
    _stats["last_audio_bytes"] = len(audio_bytes)
    _stats["total_audio_bytes"] += len(audio_bytes)

    suffix = "." + (audio.filename.rsplit(".", 1)[-1] if "." in audio.filename else "wav")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="moloch_stt_")
    try:
        os.write(fd, audio_bytes)
        os.close(fd)
        model = _get_model()
        if model is None:
            _stats["error_count"] += 1
            raise HTTPException(503, f"model not loaded: {_model_load_error or 'unknown'}")
        segments_iter, info = model.transcribe(
            tmp_path,
            language=(language or DEFAULT_LANG),
            initial_prompt=initial_prompt,
            beam_size=1,  # speed > accuracy
            vad_filter=True,  # silence-filter on
        )
        segments = []
        text_parts = []
        for seg in segments_iter:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            text_parts.append(seg.text.strip())
        full_text = " ".join(text_parts).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    duration = time.time() - t0
    _stats["last_duration_s"] = round(duration, 2)
    _stats["last_text_preview"] = full_text[:200]
    _stats["total_chars_out"] += len(full_text)
    logger.info(f"[transcribe] {len(audio_bytes)} bytes -> {len(full_text)} chars in {duration:.2f}s lang={info.language}")

    return TranscribeResponse(
        text=full_text,
        language=info.language,
        duration_s=round(info.duration, 2),
        segments=segments,
    )


def main():
    logger.info(f"MOLOCH STT-Bridge startet auf {HOST}:{PORT} (model: {MODEL_SIZE}, compute: {COMPUTE_TYPE})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
