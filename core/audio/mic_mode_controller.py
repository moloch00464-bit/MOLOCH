#!/usr/bin/env python3
"""
M.O.L.O.C.H. Mic Mode Controller — ESP32 Modus-Umschaltung
===========================================================

State Machine: IDLE (16kHz) / MUSIC (48kHz) / PTT_ACTIVE (16kHz Override)

Reagiert auf Event Bus Events:
  music.playing  -> 48kHz (Musik-Analyse Modus)
  music.stopped  -> 16kHz (Whisper bereit)
  ptt.start      -> 16kHz (Override, egal ob Musik laeuft)
  ptt.release    -> 48kHz wenn Musik aktiv, sonst 16kHz

ESP32 API: POST http://10.42.0.2/audio/mode?rate=X (timeout=2s PFLICHT)

Singleton: get_mic_mode_controller()
"""

import logging
import threading
import time
from typing import Optional

import requests

from core.moloch_event_bus import get_event_bus, PRIO_INFO

logger = logging.getLogger("MicModeController")

# ESP32 Konfiguration
ESP32_IP = "10.42.0.2"
ESP32_TIMEOUT = 2.0  # Sekunden — NIEMALS ohne Timeout!
PORT_16K = 12345
PORT_48K = 12346


# Modi
MODE_IDLE = "IDLE"
MODE_MUSIC = "MUSIC"
MODE_PTT_ACTIVE = "PTT_ACTIVE"


class MicModeController:
    """
    Steuert den ESP32 Mikrofon-Modus basierend auf Musik- und PTT-Events.

    PTT hat immer Vorrang ueber Musik.
    Thread-safe mit Lock.
    Kein Crash bei ESP32 Fehler — graceful degradation.
    """

    def __init__(self):
        self._bus = get_event_bus()
        self._lock = threading.Lock()
        self._current_mode = MODE_IDLE
        self._music_active = False
        self._ptt_active = False
        self._current_rate = 16000

    def start(self):
        """Event-Subscriptions registrieren."""
        self._bus.subscribe("music.playing", self._on_music_playing, priority=5)
        self._bus.subscribe("music.stopped", self._on_music_stopped, priority=5)
        self._bus.subscribe("ptt.start", self._on_ptt_start, priority=2)
        self._bus.subscribe("ptt.release", self._on_ptt_release, priority=2)
        logger.info("[MIC-MODE] Gestartet (IDLE, 16kHz)")

    def stop(self):
        """Subscriptions entfernen."""
        self._bus.unsubscribe("music.playing", self._on_music_playing)
        self._bus.unsubscribe("music.stopped", self._on_music_stopped)
        self._bus.unsubscribe("ptt.start", self._on_ptt_start)
        self._bus.unsubscribe("ptt.release", self._on_ptt_release)
        logger.info("[MIC-MODE] Gestoppt")

    # =========================================================================
    # Event Handler
    # =========================================================================

    def _on_music_playing(self, event):
        with self._lock:
            self._music_active = True
            if not self._ptt_active:
                self._switch_to(MODE_MUSIC, 48000)

    def _on_music_stopped(self, event):
        with self._lock:
            self._music_active = False
            if not self._ptt_active:
                self._switch_to(MODE_IDLE, 16000)

    def _on_ptt_start(self, event):
        with self._lock:
            self._ptt_active = True
            self._switch_to(MODE_PTT_ACTIVE, 16000)

    def _on_ptt_release(self, event):
        with self._lock:
            self._ptt_active = False
            if self._music_active:
                self._switch_to(MODE_MUSIC, 48000)
            else:
                self._switch_to(MODE_IDLE, 16000)

    # =========================================================================
    # ESP32 Umschaltung
    # =========================================================================

    def _switch_to(self, mode: str, rate_hz: int):
        """Modus-Wechsel und ESP32 HTTP-Switch. Lock wird vom Caller gehalten."""
        if self._current_mode == mode and self._current_rate == rate_hz:
            return  # Kein Wechsel noetig

        prev_mode = self._current_mode
        self._current_mode = mode
        self._current_rate = rate_hz
        port = PORT_48K if rate_hz == 48000 else PORT_16K

        logger.info(f"[MIC-MODE] {prev_mode} → {mode} ({rate_hz}Hz, Port {port})")

        # ESP32 ausserhalb des Locks ansprechen (in neuem Thread)
        threading.Thread(
            target=self._switch_mic_rate,
            args=(rate_hz, port),
            daemon=True,
        ).start()

    def _switch_mic_rate(self, rate_hz: int, port: int):
        """ESP32 HTTP-Call fuer Modus-Wechsel (laeuft in separatem Thread)."""
        try:
            url = f"http://{ESP32_IP}/audio/mode?rate={rate_hz}"
            resp = requests.post(url, timeout=ESP32_TIMEOUT)
            if resp.status_code == 200:
                logger.debug(f"[MIC-MODE] ESP32 OK: {rate_hz}Hz")
            else:
                logger.warning(f"[MIC-MODE] ESP32 HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning(f"[MIC-MODE] ESP32 Timeout bei {rate_hz}Hz")
        except Exception as e:
            logger.warning(f"[MIC-MODE] ESP32 Fehler: {e}")
        finally:
            # Event publishen unabhaengig vom ESP32-Ergebnis
            self._bus.publish(
                event_type="mic.mode_changed",
                source="mic_mode_controller",
                priority=PRIO_INFO,
                payload={"rate": rate_hz, "port": port, "mode": self._current_mode},
            )

    # =========================================================================
    # Status
    # =========================================================================

    @property
    def current_mode(self) -> str:
        with self._lock:
            return self._current_mode

    @property
    def current_rate(self) -> int:
        with self._lock:
            return self._current_rate


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[MicModeController] = None
_instance_lock = threading.Lock()


def get_mic_mode_controller() -> MicModeController:
    """Singleton-Zugriff auf den MicModeController."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MicModeController()
    return _instance
