#!/usr/bin/env python3
"""
M.O.L.O.C.H. Spotify Bridge — Track-Info + Audio Features via Event Bus
========================================================================

Pollt alle 5s die laufende Spotify-Session (via spotipy) und publisht:
  - music_track_started   (Prio 5) — neuer Track erkannt
  - music_features_received (Prio 5) — Audio Features fuer aktuellen Track
  - music_mood_changed    (Prio 5) — Mood-Cluster hat sich geaendert
  - music_track_finished  (Prio 5) — Track zu Ende (progress >= duration)

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

# Mood-Cluster Schwellwerte (aus Audio Features)
_MOOD_THRESHOLDS = {
    "aggressive": {"energy": 0.75, "valence_max": 0.4},
    "dark":       {"energy_max": 0.6, "valence_max": 0.35},
    "euphoric":   {"energy": 0.7, "valence": 0.6},
    "melancholic": {"energy_max": 0.5, "valence_max": 0.4},
    "neutral":    {},  # Fallback
}


def _classify_mood(features: Dict[str, float]) -> str:
    """Audio Features -> Mood-Cluster."""
    energy = features.get("energy", 0.5)
    valence = features.get("valence", 0.5)

    if energy >= 0.75 and valence < 0.4:
        return "aggressive"
    if energy < 0.6 and valence < 0.35:
        if energy < 0.4:
            return "melancholic"
        return "dark"
    if energy >= 0.7 and valence >= 0.6:
        return "euphoric"
    return "neutral"


class SpotifyBridge:
    """
    Pollt Spotify alle 5s, publisht Track- und Feature-Events.

    Nutzt SpotifyController fuer Auth/API-Zugriff (kein eigenes spotipy).
    """

    _POLL_INTERVAL = 5.0  # Sekunden

    def __init__(self):
        self._bus = get_event_bus()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # State fuer Change-Detection
        self._current_uri: Optional[str] = None
        self._current_mood: Optional[str] = None
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
        """Haupt-Poll-Schleife: Track-Info + Audio Features alle 5s."""
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                logger.error(f"[SPOTIFY-BRIDGE] Poll-Fehler: {e}")
            time.sleep(self._POLL_INTERVAL)

    def _poll_once(self):
        """Ein Poll-Zyklus: Track pruefen, Features holen, Events publishen."""
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

            # Audio Features holen (nur bei neuem Track)
            self._fetch_and_publish_features(sp_ctrl, uri)

        # Track zu Ende? (progress nahe duration und vorher weiter weg)
        if (is_playing and duration_ms > 0
                and progress_ms >= duration_ms - 2000
                and self._last_progress_ms < duration_ms - 5000):
            self._publish_track_finished()

        self._last_progress_ms = progress_ms
        self._last_duration_ms = duration_ms
        self._was_playing = is_playing

    def _fetch_and_publish_features(self, sp_ctrl, uri: str):
        """Audio Features via spotipy holen und als Event publishen."""
        try:
            if not sp_ctrl._ensure_auth():
                return
            # audio_features gibt Liste zurueck
            result = sp_ctrl._sp.audio_features([uri])
            if not result or not result[0]:
                return

            raw = result[0]
            features = {
                "energy": raw.get("energy", 0.0),
                "valence": raw.get("valence", 0.0),
                "tempo": raw.get("tempo", 0.0),
                "danceability": raw.get("danceability", 0.0),
                "loudness": raw.get("loudness", 0.0),
                "acousticness": raw.get("acousticness", 0.0),
            }

            self._bus.publish(
                event_type="music_features_received",
                source="spotify_bridge",
                priority=PRIO_INFO,
                payload={"uri": uri, "features": features},
            )

            # Mood-Klassifikation
            mood = _classify_mood(features)
            if mood != self._current_mood:
                self._current_mood = mood
                self._bus.publish(
                    event_type="music_mood_changed",
                    source="spotify_bridge",
                    priority=PRIO_INFO,
                    payload={"mood": mood, "features": features},
                )
                logger.info(f"[SPOTIFY-BRIDGE] Mood: {mood}")

        except Exception as e:
            logger.warning(f"[SPOTIFY-BRIDGE] Audio Features Fehler: {e}")

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
