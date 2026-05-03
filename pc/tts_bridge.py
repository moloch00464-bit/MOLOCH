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
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import edge_tts
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

# Persistent state fuer Voice-Picker-Auswahl
_LOCAL_APPDATA = os.environ.get("LOCALAPPDATA")
if _LOCAL_APPDATA:
    _STATE_DIR = Path(_LOCAL_APPDATA) / "moloch_pc_state"
else:
    _STATE_DIR = Path.home() / "moloch_pc_state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
VOICES_STATE_PATH = _STATE_DIR / "voices.json"

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


@app.get("/sample/{voice_name}")
def sample(voice_name: str, text: Optional[str] = None):
    """Voice-Sample fuer Cockpit-Voice-Picker.

    Gibt MP3 mit Default-Text oder optional eigener Text.
    Beispiel: GET /sample/de-DE-ConradNeural?text=Hallo+Markus
    """
    if not text:
        text = (
            "Hallo Markus. Ich bin Moloch. "
            "So klinge ich, wenn du mich auf diese Stimme einstellst."
        )
    if len(voice_name) > 80 or "/" in voice_name or ".." in voice_name:
        raise HTTPException(400, "invalid voice name")
    req = SpeakRequest(text=text, voice=voice_name)
    return speak(req)


# Vorgeschlagene 3-Voice-Mapping fuer Emotionen (Markus-Auswahl 2026-05-03):
# Edge-TTS hat nur 3 maennliche deutsche Stimmen + 3 weibliche.
# Markus kann via Cockpit-Voice-Picker die Mapping-Slots aendern.
EMOTION_VOICE_PRESETS = {
    "neutral":   "de-DE-ConradNeural",                # default, sachlich, klar
    "aufgeregt": "de-DE-KillianNeural",               # lebhaft, jung, energisch
    "ruhig":     "de-DE-FlorianMultilingualNeural",   # multilingual, sanfter Tonfall
}

ALL_GERMAN_VOICES = [
    "de-DE-ConradNeural",
    "de-DE-KillianNeural",
    "de-DE-FlorianMultilingualNeural",
    "de-DE-AmalaNeural",
    "de-DE-KatjaNeural",
    "de-DE-SeraphinaMultilingualNeural",
    "de-AT-JonasNeural",
    "de-AT-IngridNeural",
    "de-CH-JanNeural",
    "de-CH-LeniNeural",
]


def _load_persisted_presets() -> Dict[str, str]:
    if not VOICES_STATE_PATH.exists():
        return dict(EMOTION_VOICE_PRESETS)
    try:
        d = json.loads(VOICES_STATE_PATH.read_text(encoding="utf-8"))
        return d.get("presets", dict(EMOTION_VOICE_PRESETS))
    except Exception:
        return dict(EMOTION_VOICE_PRESETS)


def _atomic_write_voices(data: Dict) -> bool:
    try:
        fd, tmp = tempfile.mkstemp(dir=str(_STATE_DIR), prefix="voices.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(VOICES_STATE_PATH))
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
    except Exception:
        return False


@app.get("/presets")
def presets():
    """Voice-Presets fuer Emotionen — gespeicherte oder Default."""
    return {
        "presets": _load_persisted_presets(),
        "default_voice": DEFAULT_VOICE,
        "all_german_voices": ALL_GERMAN_VOICES,
        "state_path": str(VOICES_STATE_PATH),
    }


class PresetsRequest(BaseModel):
    neutral: str = Field(..., max_length=80)
    aufgeregt: str = Field(..., max_length=80)
    ruhig: str = Field(..., max_length=80)


@app.post("/presets")
def presets_set(req: PresetsRequest):
    """Markus' Voice-Picker-Auswahl speichern (persistent)."""
    presets_dict = {
        "neutral": req.neutral,
        "aufgeregt": req.aufgeregt,
        "ruhig": req.ruhig,
    }
    ok = _atomic_write_voices({
        "presets": presets_dict,
        "saved_at": time.time(),
    })
    if not ok:
        raise HTTPException(500, "save failed")
    return {"ok": True, "presets": presets_dict, "state_path": str(VOICES_STATE_PATH)}


@app.get("/picker", response_class=HTMLResponse)
def picker():
    """Standalone Voice-Picker-UI. Markus oeffnet im Browser, hoert, waehlt, speichert."""
    voices_options = "".join(
        f'<option value="{v}">{v}</option>' for v in ALL_GERMAN_VOICES
    )
    saved = _load_persisted_presets()
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>MOLOCH Voice-Picker</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1a1a1a; color: #ddd;
          max-width: 900px; margin: 2em auto; padding: 1em; }}
  h1 {{ color: #6cf; }}
  .slot {{ background: #2a2a2a; padding: 1em; margin: 1em 0; border-radius: 8px;
           border-left: 4px solid #6cf; }}
  .slot h3 {{ margin-top: 0; color: #fc6; }}
  select {{ background: #333; color: #ddd; padding: 0.5em; font-size: 1em;
            border: 1px solid #555; min-width: 320px; }}
  button {{ background: #6cf; color: #111; border: 0; padding: 0.5em 1em;
            margin-left: 0.5em; cursor: pointer; font-weight: bold; border-radius: 4px; }}
  button.secondary {{ background: #555; color: #ddd; }}
  button.save {{ background: #6f6; font-size: 1.2em; padding: 0.8em 2em; }}
  audio {{ display: block; margin-top: 0.5em; width: 100%; }}
  .desc {{ color: #999; font-size: 0.9em; margin: 0.3em 0 0.7em 0; }}
  #status {{ margin-top: 1em; padding: 0.5em; border-radius: 4px; }}
  #status.ok {{ background: #2a4; color: #fff; }}
  #status.err {{ background: #a24; color: #fff; }}
</style>
</head>
<body>
<h1>🎙 MOLOCH Voice-Picker</h1>
<p>Stimme pro Emotion auswählen. Play um anzuhören. Speichern macht's persistent.</p>

<div class="slot">
  <h3>🟢 Neutral (Default — sachlich, ruhig)</h3>
  <p class="desc">Wird genutzt wenn keine Emotion getriggert wird.</p>
  <select id="neutral">{voices_options}</select>
  <button onclick="play('neutral')">▶ Play</button>
</div>

<div class="slot">
  <h3>🟡 Aufgeregt (hohe Tension, alert-Zone)</h3>
  <p class="desc">Wird genutzt wenn Pi-Personality Tension >= 0.7 oder Zone=alert.</p>
  <select id="aufgeregt">{voices_options}</select>
  <button onclick="play('aufgeregt')">▶ Play</button>
</div>

<div class="slot">
  <h3>🔵 Ruhig (niedrige Tension, calm-Zone)</h3>
  <p class="desc">Wird genutzt wenn Tension <= 0.3 oder Zone=calm.</p>
  <select id="ruhig">{voices_options}</select>
  <button onclick="play('ruhig')">▶ Play</button>
</div>

<button class="save" onclick="save()">💾 Auswahl speichern</button>
<button class="secondary" onclick="reset()">↺ Default zurücksetzen</button>

<audio id="player" controls></audio>
<div id="status"></div>

<script>
const SAVED = {json.dumps(saved)};
document.getElementById('neutral').value = SAVED.neutral || '{EMOTION_VOICE_PRESETS["neutral"]}';
document.getElementById('aufgeregt').value = SAVED.aufgeregt || '{EMOTION_VOICE_PRESETS["aufgeregt"]}';
document.getElementById('ruhig').value = SAVED.ruhig || '{EMOTION_VOICE_PRESETS["ruhig"]}';

function play(slot) {{
  const voice = document.getElementById(slot).value;
  const player = document.getElementById('player');
  const text = encodeURIComponent('Hallo Markus, ich bin Moloch. So klinge ich für ' + slot + '.');
  player.src = '/sample/' + voice + '?text=' + text;
  player.play();
}}

function save() {{
  const data = {{
    neutral: document.getElementById('neutral').value,
    aufgeregt: document.getElementById('aufgeregt').value,
    ruhig: document.getElementById('ruhig').value,
  }};
  fetch('/presets', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(data),
  }}).then(r => r.json()).then(d => {{
    const s = document.getElementById('status');
    if (d.ok) {{ s.className = 'ok'; s.textContent = 'Gespeichert: ' + JSON.stringify(d.presets); }}
    else {{ s.className = 'err'; s.textContent = 'Fehler: ' + JSON.stringify(d); }}
  }}).catch(e => {{
    const s = document.getElementById('status');
    s.className = 'err'; s.textContent = 'Netzwerk-Fehler: ' + e;
  }});
}}

function reset() {{
  document.getElementById('neutral').value = '{EMOTION_VOICE_PRESETS["neutral"]}';
  document.getElementById('aufgeregt').value = '{EMOTION_VOICE_PRESETS["aufgeregt"]}';
  document.getElementById('ruhig').value = '{EMOTION_VOICE_PRESETS["ruhig"]}';
}}
</script>
</body>
</html>"""


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
