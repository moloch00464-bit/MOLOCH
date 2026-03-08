#!/usr/bin/env python3
"""
M.O.L.O.C.H. Spotify Bridge — Track-Info + Mood via Event Bus
===============================================================

Pollt alle 5s die laufende Spotify-Session (via spotipy) und publisht:
  - music_track_started   (Prio 5) — neuer Track erkannt
  - music_track_finished  (Prio 5) — Track zu Ende (progress >= duration)

HINWEIS: Spotify audio_features API seit 2024 deprecated (403 Forbidden).
Energy/Tempo/Valence kommen stattdessen aus FFT-Analyse (music_visualizer.py).

Kein direkter Zugriff auf Tension/CoreIntegrator — NUR Events publishen.
moloch_service.py abonniert spaeter.

Singleton: get_spotify_bridge()
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

from core.moloch_event_bus import get_event_bus, PRIO_INFO

logger = logging.getLogger("MolochSpotifyBridge")


class SpotifyBridge:
    """
    Pollt Spotify alle 5s, publisht Track-Events.

    Nutzt SpotifyController fuer Auth/API-Zugriff (kein eigenes spotipy).
    Audio Features (energy, tempo, valence) kommen aus FFT (music_visualizer),
    NICHT aus Spotify API (deprecated seit 2024).
    """

    _POLL_INTERVAL = 5.0  # Sekunden

    def __init__(self):
        self._bus = get_event_bus()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # State fuer Change-Detection
        self._current_uri: Optional[str] = None
        self._last_progress_ms: int = 0
        self._last_duration_ms: int = 0
        self._was_playing: bool = False

    def start(self):
        """Poll-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="SpotifyBridge"
        )
        self._thread.start()
        logger.info("[SPOTIFY-BRIDGE] Gestartet (Poll alle 5s)")

    def stop(self):
        """Poll-Thread stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("[SPOTIFY-BRIDGE] Gestoppt")

    def _poll_loop(self):
        """Haupt-Poll-Schleife: Track-Info alle 5s."""
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                logger.error(f"[SPOTIFY-BRIDGE] Poll-Fehler: {e}")
            time.sleep(self._POLL_INTERVAL)

    def _poll_once(self):
        """Ein Poll-Zyklus: Track pruefen, Events publishen."""
        from core.spotify_controller import get_spotify
        sp_ctrl = get_spotify()

        # Aktuellen Track holen
        track = sp_ctrl.get_current_track()
        if not track:
            # Nichts laeuft — pruefen ob vorher was lief (= Track finished)
            if self._was_playing and self._current_uri:
                self._publish_track_finished()
            self._was_playing = False
            return

        uri = track.get("uri", "")
        is_playing = track.get("is_playing", False)
        progress_ms = track.get("progress_ms", 0)
        duration_ms = track.get("duration_ms", 0)

        # Track-Wechsel erkannt?
        if uri != self._current_uri:
            # Alten Track als finished melden (wenn vorher einer lief)
            if self._was_playing and self._current_uri:
                self._publish_track_finished()

            # Neuen Track melden
            self._current_uri = uri
            self._bus.publish(
                event_type="music_track_started",
                source="spotify_bridge",
                priority=PRIO_INFO,
                payload={
                    "artist": track.get("artist", "?"),
                    "track": track.get("track", "?"),
                    "album": track.get("album", "?"),
                    "uri": uri,
                    "duration_ms": duration_ms,
                },
            )
            logger.info(f"[SPOTIFY-BRIDGE] Track: {track.get('artist')} - {track.get('track')}")

        # Track zu Ende? (progress nahe duration und vorher weiter weg)
        if (is_playing and duration_ms > 0
                and progress_ms >= duration_ms - 2000
                and self._last_progress_ms < duration_ms - 5000):
            self._publish_track_finished()

        self._last_progress_ms = progress_ms
        self._last_duration_ms = duration_ms
        self._was_playing = is_playing

    def _publish_track_finished(self):
        """Track-Ende Event publishen."""
        self._bus.publish(
            event_type="music_track_finished",
            source="spotify_bridge",
            priority=PRIO_INFO,
            payload={"uri": self._current_uri or ""},
        )
        logger.debug(f"[SPOTIFY-BRIDGE] Track finished: {self._current_uri}")


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[SpotifyBridge] = None
_instance_lock = threading.Lock()


def get_spotify_bridge() -> SpotifyBridge:
    """Singleton-Zugriff auf die Spotify Bridge."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SpotifyBridge()
    return _instance
