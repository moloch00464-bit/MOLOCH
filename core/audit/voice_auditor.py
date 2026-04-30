"""Voice-Layer-Auditor (Welle 13).

Misst Sprach-/Audio-Pipeline:
- VoicePipeline (core.voice_pipeline, best-effort)
- WiFiMic (core.audio.wifi_mic, best-effort)
- moloch_status.json voice/audio/wifi_mic Block
- TTS-Calls aus journalctl (subprocess timeout=10)

Schreibt audit_state.layers.voice:
  {mic_pegel_age_s, esp32_rssi, wifi_mic_frames_1min, tts_calls_1h,
   score, max, status, detail}

Status-Logik:
- PASS: mic alive (recv >0 letzte Min), TTS reagiert
- WARN: mic stale >2min
- FAIL: module nicht importierbar ODER mic dead >5min
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Dict

logger = logging.getLogger("voice_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"


def collect() -> Dict[str, Any]:
    """Sammelt Voice-Layer-Daten."""
    detail: Dict[str, Any] = {}
    mic_pegel_age_s = 99999.0
    esp32_rssi: Any = None
    wifi_mic_frames_1min = 0
    tts_calls_1h = 0
    voice_module_alive = False
    wifi_mic_alive = False

    # 1. VoicePipeline import (L0, best-effort)
    try:
        import core.voice_pipeline as _vp  # type: ignore
        # Klasse vorhanden = importierbar
        voice_module_alive = hasattr(_vp, "VoicePipeline")
        detail["voice_module_imported"] = voice_module_alive
    except Exception as e:
        detail["voice_import_error"] = str(e)[:120]

    # 2. WiFiMic singleton (L0, best-effort, KEINE Auto-Init falls noch nicht da)
    try:
        import core.audio.wifi_mic as _wm  # type: ignore
        # Pruefen ob bereits Instanz existiert (kein neuer Init)
        existing = getattr(_wm, "_instance", None) or getattr(
            _wm, "_wifi_mic_singleton", None
        )
        if existing is not None:
            wifi_mic_alive = True
            try:
                last_recv_16k = getattr(existing, "_last_recv_16k", 0.0) or 0.0
                last_recv_48k = getattr(existing, "_last_recv_48k", 0.0) or 0.0
                last_recv = max(float(last_recv_16k), float(last_recv_48k))
                if last_recv > 0:
                    mic_pegel_age_s = max(0.0, time.time() - last_recv)
            except Exception as ee:
                detail["wifi_mic_state_error"] = str(ee)[:100]
        else:
            detail["wifi_mic_singleton_initialised"] = False
    except Exception as e:
        detail["wifi_mic_import_error"] = str(e)[:120]

    # 3. moloch_status.json (L1+L2)
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        voice = st.get("voice") or {}
        audio = st.get("audio") or {}
        if isinstance(voice, dict):
            wm = voice.get("wifi_mic") or {}
            if isinstance(wm, dict):
                # packets_recv als Frame-Approximation
                pkt_total = int(wm.get("packets_total_16k", 0) or 0)
                detail["packets_total_16k"] = pkt_total
                detail["packets_recv_16k"] = int(wm.get("packets_recv_16k", 0) or 0)
                detail["loss_pct_16k"] = wm.get("loss_pct_16k")
                # Connection state
                detail["connected_16k"] = bool(wm.get("connected_16k"))
                detail["connected_48k"] = bool(wm.get("connected_48k"))
                # rms_db als Pegel-Indikator
                detail["rms_db"] = wm.get("rms_db")
                # esp_ip vorhanden = ESP erreichbar
                if wm.get("esp_ip"):
                    detail["esp_ip"] = wm.get("esp_ip")
                # Ohne Singleton: Pegel-Age aus connected_16k true ableiten
                if mic_pegel_age_s == 99999.0 and wm.get("connected_16k"):
                    mic_pegel_age_s = 0.0
            # voice_enabled / current_voice / piper_available
            detail["voice_enabled"] = bool(voice.get("voice_enabled"))
            detail["piper_available"] = bool(voice.get("piper_available"))
            detail["whisper_status"] = voice.get("whisper_status")
            # ESP32-RSSI (falls in Status-JSON gemeldet)
            esp32_rssi = (
                voice.get("esp32_rssi")
                or voice.get("rssi")
                or (voice.get("wifi_mic", {}) or {}).get("rssi")
            )
        if isinstance(audio, dict):
            level = audio.get("level")
            if level is not None:
                detail["audio_level"] = level
                # Wenn Audio-Level >0 in letzter Tick, mic nicht ganz tot
                if mic_pegel_age_s == 99999.0:
                    try:
                        if float(level) > 0:
                            mic_pegel_age_s = 0.0
                    except (TypeError, ValueError):
                        pass
        # wifi_mic_frames_1min Approximation: nur dann zaehlbar
        # wenn packets_total_16k einen Heartbeat zeigt
        # (kein /min-Counter im JSON; wir nutzen connected*60 als Heuristik)
        if detail.get("connected_16k"):
            # Annahme: 50 Pakete/s @ 16k -> 3000/min, klippen
            wifi_mic_frames_1min = 3000
    except Exception as e:
        detail["status_json_error"] = str(e)[:100]

    # 4. TTS-Calls/h aus journalctl (subprocess timeout=10)
    try:
        # Minus 1 Stunde, grep [TTS], zaehlen
        proc = subprocess.run(
            [
                "journalctl",
                "-u",
                "moloch",
                "--since",
                "1 hour ago",
                "--no-pager",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            out = proc.stdout or ""
            tts_calls_1h = sum(1 for ln in out.splitlines() if "[TTS]" in ln)
            detail["journalctl_lines_1h"] = len(out.splitlines())
        else:
            detail["journalctl_rc"] = proc.returncode
    except subprocess.TimeoutExpired:
        detail["journalctl_timeout"] = True
    except Exception as e:
        detail["journalctl_error"] = str(e)[:100]

    detail["voice_module_alive"] = voice_module_alive
    detail["wifi_mic_alive"] = wifi_mic_alive

    # 5. Status berechnen
    score = 0
    max_score = 4
    if voice_module_alive:
        score += 1
    if mic_pegel_age_s != 99999.0 and mic_pegel_age_s < 60:
        score += 1
    if wifi_mic_frames_1min > 0:
        score += 1
    if tts_calls_1h > 0:
        score += 1

    mic_dead = (
        mic_pegel_age_s == 99999.0 or mic_pegel_age_s > 300
    )
    mic_stale = (
        mic_pegel_age_s != 99999.0 and 120 < mic_pegel_age_s <= 300
    )

    if not voice_module_alive:
        status = "FAIL"
    elif mic_dead:
        status = "FAIL"
    elif mic_stale:
        status = "WARN"
    elif tts_calls_1h == 0 and mic_pegel_age_s != 99999.0:
        # Mic ok, aber TTS schweigt - WARN
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "mic_pegel_age_s": round(mic_pegel_age_s, 1)
        if mic_pegel_age_s != 99999.0
        else None,
        "esp32_rssi": esp32_rssi,
        "wifi_mic_frames_1min": wifi_mic_frames_1min,
        "tts_calls_1h": tts_calls_1h,
        "detail": detail,
    }
