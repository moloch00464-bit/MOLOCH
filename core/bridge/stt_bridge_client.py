#!/usr/bin/env python3
"""
STT-Bridge-Client (Pi-Side)
============================
Schickt Audio zur PC-STT-Bridge (faster-whisper auf Markus-PC),
bekommt Transkription als dict zurueck.

Config: settings.stt_bridge (host/port/language/beam_size/vad_filter/timeout_sec/enabled).
"""
import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("SttBridgeClient")

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json",
)


def _load_cfg() -> dict:
    defaults = {
        "enabled": False,
        "host": "192.168.178.20",
        "port": 9001,
        "language": "de",
        "beam_size": 5,
        "vad_filter": True,
        "timeout_sec": 60,
    }
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f).get("stt_bridge", {}) or {}
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


def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Optional[dict]:
    """Schickt Audio-File zur PC-Bridge, returns {text, language, duration, segments} oder None."""
    cfg = _load_cfg()
    if not cfg.get("enabled"):
        return None
    if not os.path.exists(audio_path):
        logger.warning(f"Audio-File fehlt: {audio_path}")
        return None
    url = f"http://{cfg['host']}:{cfg['port']}/transcribe"
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "application/octet-stream")}
            data = {
                "language": language or cfg.get("language", "de"),
                "beam_size": str(cfg.get("beam_size", 5)),
                "vad_filter": str(cfg.get("vad_filter", True)).lower(),
            }
            r = requests.post(url, files=files, data=data, timeout=cfg.get("timeout_sec", 60))
        if r.status_code != 200:
            logger.warning(f"STT-Bridge HTTP {r.status_code}: {r.text[:200]}")
            return None
        result = r.json()
        logger.info(
            f"STT-Bridge: '{result.get('text', '')[:80]}' "
            f"(lang={result.get('language')}, dur={result.get('duration')}s)"
        )
        return result
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"STT-Bridge Fehler: {e}")
        return None


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav",
                     language: Optional[str] = None) -> Optional[dict]:
    """Wie transcribe_audio, aber direkt aus Bytes statt File-Pfad."""
    cfg = _load_cfg()
    if not cfg.get("enabled"):
        return None
    url = f"http://{cfg['host']}:{cfg['port']}/transcribe"
    try:
        files = {"audio": ("audio" + suffix, audio_bytes, "application/octet-stream")}
        data = {
            "language": language or cfg.get("language", "de"),
            "beam_size": str(cfg.get("beam_size", 5)),
            "vad_filter": str(cfg.get("vad_filter", True)).lower(),
        }
        r = requests.post(url, files=files, data=data, timeout=cfg.get("timeout_sec", 60))
        if r.status_code != 200:
            logger.warning(f"STT-Bridge HTTP {r.status_code}: {r.text[:200]}")
            return None
        return r.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"STT-Bridge Fehler: {e}")
        return None


if __name__ == "__main__":
    print("CFG:", _load_cfg())
    h = health_check()
    print("HEALTH:", h)
    # Smoke-Test mit Piper-generiertem WAV
    import subprocess
    import tempfile
    test_text = "Hallo Markus, hier ist ein Bruecken-Test fuer die Spracherkennung."
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        # Piper-TTS auf Pi nutzen um WAV zu generieren
        piper_cmd = (
            f'echo "{test_text}" | ~/.local/bin/piper '
            f'--model /home/molochzuhause/moloch/models/voices/de_DE-thorsten-medium.onnx '
            f'--output_file {wav_path}'
        )
        subprocess.run(piper_cmd, shell=True, check=True, timeout=15)
        print(f"Test-WAV erzeugt ({os.path.getsize(wav_path)} Bytes)")
        result = transcribe_audio(wav_path)
        print("TRANSCRIBE:", result)
    finally:
        try: os.unlink(wav_path)
        except OSError: pass
