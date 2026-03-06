#!/usr/bin/env python3
"""
M.O.L.O.C.H. Activity Analyzer — Aktivitaetserkennung aus kombinierten Signalen
==================================================================================

Kombiniert mehrere Signale zu einem Aktivitaetszustand:
  - alone: Markus allein, ruhig
  - working: Markus am Schreibtisch, wenig Bewegung
  - conversation: Mehrere Personen oder Sprach-Aktivitaet
  - party: Hohe Energy, Musik laut, Bewegung
  - away: Niemand da

Signale: Personen-Anzahl, Bewegung, Musik-Energy, Tageszeit, Zone.

Publiziert activity_changed Event bei Zustandswechsel.

Singleton: get_activity_analyzer()
Gate 3: Situational Awareness
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("MolochActivityAnalyzer")

# Hysterese: State muss N Sekunden stabil sein bevor Wechsel
STABILITY_SECONDS = 5.0


class ActivityAnalyzer:
    """Aktivitaetserkennung aus kombinierten Signalen."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "away"
        self._candidate = "away"
        self._candidate_since: float = 0.0
        self._last_publish: float = 0.0

        # Input-Signale (werden laufend aktualisiert)
        self._person_count: int = 0
        self._motion_state: str = "stationary"
        self._music_energy: float = 0.0
        self._zone: Optional[str] = None
        self._voice_active: bool = False
        self._face_id: Optional[str] = None

    def update_signals(self, person_count: int = 0,
                       motion_state: str = "stationary",
                       music_energy: float = 0.0,
                       zone: Optional[str] = None,
                       voice_active: bool = False,
                       face_id: Optional[str] = None):
        """Alle Signale auf einmal updaten.

        Args:
            person_count: Anzahl erkannter Personen
            motion_state: Aus MotionAnalyzer (stationary/walking/approaching/leaving)
            music_energy: Musik-Energy aus Spotify Bridge (0.0-1.0)
            zone: Aktuelle Raumzone aus RoomMap
            voice_active: Ist gerade Sprach-Aktivitaet?
            face_id: Erkannte Person (oder None)
        """
        with self._lock:
            self._person_count = person_count
            self._motion_state = motion_state
            self._music_energy = music_energy
            self._zone = zone
            self._voice_active = voice_active
            self._face_id = face_id

    def evaluate(self) -> Optional[str]:
        """Aktivitaetszustand neu berechnen.

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

            # Hysterese: Kandidat muss stabil sein
            if candidate == self._state:
                return None

            elapsed = now - self._candidate_since
            if elapsed < STABILITY_SECONDS:
                return None

            old_state = self._state
            self._state = candidate

        # Event publizieren (ausserhalb Lock)
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="activity_changed",
                source="activity_analyzer",
                priority=5,
                payload={
                    "activity": candidate,
                    "previous_activity": old_state,
                    "person_count": self._person_count,
                    "motion_state": self._motion_state,
                    "music_energy": round(self._music_energy, 2),
                    "zone": self._zone,
                },
            )
        except Exception as e:
            logger.debug(f"[ACTIVITY] Event publish: {e}")

        return candidate

    def _classify(self) -> str:
        """Aktivitaet aus Signalen klassifizieren (unter Lock aufrufen)."""
        pc = self._person_count
        motion = self._motion_state
        energy = self._music_energy
        voice = self._voice_active

        # Niemand da
        if pc == 0:
            return "away"

        # Party: Viele Leute ODER hohe Musik-Energy + Bewegung
        if pc >= 3 or (energy > 0.7 and motion in ("walking", "approaching")):
            return "party"

        # Conversation: Voice aktiv oder 2 Personen
        if voice or pc >= 2:
            return "conversation"

        # Working: Am Schreibtisch, still
        if self._zone == "schreibtisch" and motion == "stationary":
            return "working"

        # Alone: 1 Person, ruhig
        return "alone"

    @property
    def current_activity(self) -> str:
        with self._lock:
            return self._state

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "activity": self._state,
                "person_count": self._person_count,
                "motion_state": self._motion_state,
                "music_energy": round(self._music_energy, 2),
                "zone": self._zone,
                "voice_active": self._voice_active,
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[ActivityAnalyzer] = None
_instance_lock = threading.Lock()


def get_activity_analyzer() -> ActivityAnalyzer:
    """Singleton-Zugriff auf Activity Analyzer."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ActivityAnalyzer()
    return _instance
