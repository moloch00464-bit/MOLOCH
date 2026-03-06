#!/usr/bin/env python3
"""
M.O.L.O.C.H. Atmosphere Controller — Raumatmosphaere als Einheit
==================================================================

Steuert Musik + LED + PTZ als zusammenhaengende Atmosphaere.
Nicht einzelne Aktionen, sondern ein Gesamtzustand.

States:
  - intimate: Leise Musik, LED Standlicht, Kamera ruhig
  - focused:  Keine/leise Musik, LED an, Kamera am Schreibtisch
  - party:    Laute Musik, LED pulsierend, Kamera frei
  - alert:    Keine Musik, LED blink, Kamera zur Tuer
  - night:    Alles aus/minimal, Kamera Park-Position

Reagiert auf activity_changed + mood_changed Events.
Publiziert atmosphere_changed Event (Priority 5) bei Zustandswechsel.

Singleton: get_atmosphere_controller()
Gate 5: Autonomous Environmental Agent
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("MolochAtmosphere")

# Hysterese: State muss N Sekunden stabil sein
STABILITY_SECONDS = 5.0

# Atmosphaere-Definitionen: Was jeder State bewirkt
ATMOSPHERE_PROFILES = {
    "intimate": {
        "music_energy_target": 0.3,
        "music_command": "volume_low",
        "led": "on",
        "ptz_behavior": "stay",
        "description": "Leise, ruhig, Standlicht",
    },
    "focused": {
        "music_energy_target": 0.2,
        "music_command": "volume_low",
        "led": "on",
        "ptz_behavior": "desk",
        "description": "Arbeitsatmosphaere, minimal",
    },
    "party": {
        "music_energy_target": 0.8,
        "music_command": "volume_up",
        "led": "blink_slow",
        "ptz_behavior": "free",
        "description": "Laut, pulsierend, frei",
    },
    "alert": {
        "music_energy_target": 0.0,
        "music_command": "pause",
        "led": "blink",
        "ptz_behavior": "door",
        "description": "Wachsam, Musik aus, Tuer",
    },
    "night": {
        "music_energy_target": 0.0,
        "music_command": "pause",
        "led": "off",
        "ptz_behavior": "park",
        "description": "Nachtmodus, alles minimal",
    },
}


class AtmosphereController:
    """Steuert Raumatmosphaere: Musik+LED+PTZ als Einheit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "night"
        self._candidate = "night"
        self._candidate_since: float = 0.0
        self._last_publish: float = 0.0

        # Input-Signale
        self._mood: str = "calm"
        self._activity: str = "away"
        self._hour: int = 0
        self._face_id: Optional[str] = None
        self._tension: float = 0.0

    def on_activity_changed(self, event: Dict[str, Any]):
        """Event-Handler fuer activity_changed Events."""
        payload = event.get("payload", {})
        with self._lock:
            self._activity = payload.get("activity", self._activity)
        self.evaluate()

    def on_mood_changed(self, event: Dict[str, Any]):
        """Event-Handler fuer mood_changed Events."""
        payload = event.get("payload", {})
        with self._lock:
            self._mood = payload.get("mood", self._mood)
            self._tension = payload.get("tension", self._tension)
        self.evaluate()

    def update_signals(self, hour: Optional[int] = None,
                       face_id: Optional[str] = None):
        """Zusaetzliche Signale updaten (aus Perception Loop)."""
        with self._lock:
            if hour is not None:
                self._hour = hour
            self._face_id = face_id

    def evaluate(self) -> Optional[str]:
        """Atmosphaere neu berechnen.

        Returns:
            Neuer State wenn gewechselt, None wenn gleich
        """
        with self._lock:
            candidate = self._classify()
            now = time.monotonic()

            if candidate != self._candidate:
                self._candidate = candidate
                self._candidate_since = now
                return None

            if candidate == self._state:
                return None

            if (now - self._candidate_since) < STABILITY_SECONDS:
                return None

            old_state = self._state
            self._state = candidate
            profile = ATMOSPHERE_PROFILES.get(candidate, {})

        # Event publizieren (ausserhalb Lock)
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="atmosphere_changed",
                source="atmosphere_controller",
                priority=5,
                payload={
                    "atmosphere": candidate,
                    "previous": old_state,
                    "profile": profile,
                    "mood": self._mood,
                    "activity": self._activity,
                },
            )
            logger.info(f"[ATMO] {old_state} -> {candidate}: "
                        f"{profile.get('description', '?')}")
        except Exception as e:
            logger.debug(f"[ATMO] Event publish: {e}")

        return candidate

    def _classify(self) -> str:
        """Atmosphaere aus Signalen klassifizieren (unter Lock)."""
        hour = self._hour
        mood = self._mood
        activity = self._activity
        tension = self._tension

        # Nacht: 23:00-06:00 und niemand aktiv
        if (hour >= 23 or hour < 6) and activity in ("away", "alone"):
            return "night"

        # Alert: Hohe Tension oder unbekannte Person
        if mood in ("alert", "agitated", "dark") or tension > 0.6:
            return "alert"

        # Party: Party-Aktivitaet oder euphoric Mood
        if activity == "party" or mood == "euphoric":
            return "party"

        # Focused: Am Schreibtisch arbeiten
        if activity == "working" or mood == "focused":
            return "focused"

        # Intimate: Allein/Gespraech, ruhig
        if activity in ("alone", "conversation") and mood == "calm":
            return "intimate"

        # Default: Nacht bei away, intimate sonst
        if activity == "away":
            return "night"
        return "intimate"

    # =====================================================================
    # Public API
    # =====================================================================

    @property
    def current_atmosphere(self) -> str:
        with self._lock:
            return self._state

    @property
    def current_profile(self) -> Dict[str, Any]:
        with self._lock:
            return dict(ATMOSPHERE_PROFILES.get(self._state, {}))

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "atmosphere": self._state,
                "mood": self._mood,
                "activity": self._activity,
                "hour": self._hour,
                "profile": ATMOSPHERE_PROFILES.get(self._state, {}),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[AtmosphereController] = None
_instance_lock = threading.Lock()


def get_atmosphere_controller() -> AtmosphereController:
    """Singleton-Zugriff auf Atmosphere Controller."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AtmosphereController()
    return _instance
