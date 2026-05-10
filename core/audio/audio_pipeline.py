"""Audio-Pipeline (Sprint-2 Fix-2 NEU 2026-05-10).

Source-Wrapper ueber WiFi-Mic (ESP32 ReSpeaker) und USB-Mic.
Routet basierend auf settings.audio.mic_source:
  - 'auto': ESP32 wenn connected_48k=True, sonst USB-Fallback
  - 'esp32': forciert ESP32 (auch wenn connected_48k=False)
  - 'usb': forciert USB-Mic (umgeht ESP32 komplett)

Use-Case (Sprint 2): Spotify spielt -> mic_mode_controller switcht auf
48kHz-Mode -> wenn ESP32-48kHz-Bug zuschlaegt, Auto-Fallback auf USB-Mic
damit music_listener-FFT trotzdem laeuft.

Hinweis: das ist ein duenner Wrapper. Eigentliche Audio-Akquise laeuft
in core/audio/wifi_mic.py (UDP-Listener). USB-Mic wird hier per
sounddevice-Stream geoeffnet wenn aktiviert.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")
_USB_SAMPLE_RATE = 48000
_USB_CHANNELS = 1
_USB_BLOCKSIZE = 960  # ~20ms @ 48kHz


def _load_audio_config() -> dict:
    """Liest settings.audio mit Defaults. Existing key heisst audio_source
    (nicht mic_source) — kompatibel mit Pre-Sprint-2-Code."""
    defaults = {
        "audio_source": "auto",  # 'auto' | 'esp32' | 'usb'
        "usb_card_index": 0,
        "spotify_to_usb_speaker": False,
    }
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        audio = cfg.get("audio") or {}
        for k in defaults:
            if k in audio:
                defaults[k] = audio[k]
        # Backward-Compat: mic_source (Sprint-2-Plan) -> audio_source
        if "mic_source" in audio and "audio_source" not in audio:
            defaults["audio_source"] = audio["mic_source"]
    except Exception as e:
        logger.debug(f"[audio_pipeline] settings load err: {e}")
    return defaults


class AudioPipeline:
    """Source-Switch zwischen WiFi-Mic und USB-Mic."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cfg = _load_audio_config()
        self._wifi_mic = None
        self._usb_stream = None
        self._usb_callback = None
        self._active_source = "none"
        self._last_check = 0.0

    def get_active_source(self) -> str:
        """Returns 'wifi', 'usb', or 'none'."""
        return self._active_source

    def get_config(self) -> dict:
        """Liefert kopierte Audio-Config (read-only)."""
        return dict(self._cfg)

    def reload_config(self):
        """Re-read settings.json (bei Live-Aenderung)."""
        with self._lock:
            self._cfg = _load_audio_config()
            logger.info(f"[audio_pipeline] config reload: {self._cfg}")

    def select_source(self) -> str:
        """Bestimmt aktive Mic-Source nach audio_source-Setting + Fallback-Logik."""
        with self._lock:
            mic_source = self._cfg.get("audio_source", "auto")
            if mic_source == "usb":
                self._active_source = "usb"
                return "usb"
            if mic_source == "esp32":
                self._active_source = "wifi"
                return "wifi"
            # auto: ESP32 wenn connected_48k, sonst USB-Fallback
            try:
                from core.audio.wifi_mic import get_wifi_mic
                wm = get_wifi_mic()
                if wm and getattr(wm, "_connected_48k", False):
                    self._active_source = "wifi"
                    return "wifi"
            except Exception:
                pass
            self._active_source = "usb"
            return "usb"

    def start_usb_stream(self, callback) -> bool:
        """Oeffnet USB-Audio-Stream mit sounddevice. callback(audio_data, ...)."""
        try:
            import sounddevice as sd
            import numpy as np
            self._usb_callback = callback

            def _sd_callback(indata, frames, time_info, status):
                if status:
                    logger.debug(f"[usb_mic] status: {status}")
                if self._usb_callback:
                    try:
                        self._usb_callback(np.asarray(indata).copy())
                    except Exception as e:
                        logger.debug(f"[usb_mic] callback err: {e}")

            self._usb_stream = sd.InputStream(
                samplerate=_USB_SAMPLE_RATE,
                channels=_USB_CHANNELS,
                blocksize=_USB_BLOCKSIZE,
                dtype="float32",
                callback=_sd_callback,
                device=self._cfg.get("usb_card_index", 0),
            )
            self._usb_stream.start()
            self._active_source = "usb"
            logger.info(f"[audio_pipeline] USB-Mic Stream gestartet (rate={_USB_SAMPLE_RATE}, "
                        f"card={self._cfg.get('usb_card_index', 0)})")
            return True
        except Exception as e:
            logger.warning(f"[audio_pipeline] USB-Stream start fail: {e}")
            return False

    def stop_usb_stream(self):
        """Schliesst USB-Stream."""
        with self._lock:
            if self._usb_stream:
                try:
                    self._usb_stream.stop()
                    self._usb_stream.close()
                except Exception:
                    pass
                self._usb_stream = None
            if self._active_source == "usb":
                self._active_source = "none"


_instance: Optional[AudioPipeline] = None
_inst_lock = threading.Lock()


def get_audio_pipeline() -> AudioPipeline:
    global _instance
    with _inst_lock:
        if _instance is None:
            _instance = AudioPipeline()
    return _instance
