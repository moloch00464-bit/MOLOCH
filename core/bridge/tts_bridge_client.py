#!/usr/bin/env python3
"""
TTS-Bridge-Client (Pi-Side)
============================
Schickt Text zur PC-TTS-Bridge (edge-tts auf Markus-PC), bekommt MP3,
spielt mit ffplay ab. Fallback-Verhalten: bei enabled=false oder Bridge-
Fehler returned False — Caller (z.B. core/tts.py) faellt dann auf
lokales Piper zurueck.

Config: settings.tts_bridge (host/port/voice/rate/pitch/timeout_sec/enabled).
"""
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

import requests

logger = logging.getLogger("TtsBridgeClient")

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json",
)


def _load_cfg() -> dict:
    defaults = {
        "enabled": False,
        "host": "192.168.178.20",
        "port": 9002,
        "voice": "de-DE-ConradNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "timeout_sec": 30,
    }
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f).get("tts_bridge", {}) or {}
        defaults.update(cfg)
    except Exception as e:
        logger.warning(f"settings.json Lesefehler: {e}, nutze Defaults")
    return defaults


def is_enabled() -> bool:
    return bool(_load_cfg().get("enabled"))


def health_check(timeout_sec: int = 3) -> Optional[dict]:
    cfg = _load_cfg()
    if not cfg.get("enabled"):
        return None
    try:
        r = requests.get(f"http://{cfg['host']}:{cfg['port']}/health", timeout=timeout_sec)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def speak_via_bridge(text: str, voice: Optional[str] = None,
                     rate: Optional[str] = None, pitch: Optional[str] = None) -> bool:
    """Schickt Text zur PC-Bridge, spielt MP3 mit ffplay ab. True bei Erfolg."""
    cfg = _load_cfg()
    if not cfg.get("enabled"):
        return False
    payload = {
        "text": text,
        "voice": voice or cfg.get("voice", "de-DE-ConradNeural"),
        "rate": rate or cfg.get("rate", "+0%"),
        "pitch": pitch or cfg.get("pitch", "+0Hz"),
    }
    url = f"http://{cfg['host']}:{cfg['port']}/speak"
    try:
        r = requests.post(url, json=payload, timeout=cfg.get("timeout_sec", 30))
        if r.status_code != 200:
            logger.warning(f"TTS-Bridge HTTP {r.status_code}: {r.text[:200]}")
            return False
        if not r.content:
            logger.warning("TTS-Bridge: leere Antwort")
            return False
        # MP3 in temp-File, mit ffplay abspielen (-nodisp -autoexit, kein Window)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", tmp_path],
                check=True, timeout=60,
            )
            logger.info(f"TTS-Bridge: {len(r.content)} Bytes abgespielt (voice={payload['voice']})")
            return True
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except (requests.RequestException, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"TTS-Bridge Fehler: {e}")
        return False


if __name__ == "__main__":
    # Smoke-Test
    print("CFG:", _load_cfg())
    h = health_check()
    print("HEALTH:", h)
    if h:
        ok = speak_via_bridge("Hallo Markus, hier ist dein TTS-Bruecken-Test ueber den Rechner.")
        print("SPEAK ergebnis:", ok)
