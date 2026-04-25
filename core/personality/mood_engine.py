#!/usr/bin/env python3
"""
M.O.L.O.C.H. Mood Engine — Emergenter Mood-State aus kombinierten Signalen
=============================================================================

Kombiniert tension + dominance + music_mood + activity zu einem emergenten
Mood-State. Der Mood bestimmt wie Moloch sich "fuehlt".

States:
  - calm:     Niedrige Tension, Guardian-Zone, allein/working
  - focused:  Mittlere Tension, Schreibtisch, Person anwesend
  - alert:    Hohe Tension ODER unbekannte Person
  - agitated: Sehr hohe Tension, Shadow-Zone
  - euphoric: Party-Aktivitaet ODER hohe Musik-Energy + Guardian
  - dark:     Shadow-Zone + hohe Tension + dunkle Musik

Publiziert mood_changed Event bei Zustandswechsel.

Singleton: get_mood_engine()
Gate 4: Emergent Personality
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("MolochMoodEngine")

# Hysterese: Mood muss N Sekunden stabil sein
STABILITY_SECONDS = 3.0


class MoodEngine:
    """Emergenter Mood-State aus kombinierten Signalen."""

    def __init__(self):
        self._lock = threading.Lock()
        self._mood = "calm"
        self._candidate = "calm"
        self._candidate_since: float = 0.0

        # Input-Signale (laufend aktualisiert)
        self._tension: float = 0.0
        self._dominance: float = 0.5
        self._personality_zone: str = "guardian"
        self._music_mood: Optional[str] = None
        self._activity: str = "away"
        self._face_id: Optional[str] = None
        self._music_energy: float = 0.0

        # Gate 1.5 Phase 4: Charakter-Drift-Baseline vom Distiller
        # mood_baseline > 0 = positiv gedriftet (effective_t SINKT, mehr calm)
        # energy_baseline > 0 = energischer (effective_e steigt)
        self._drift_mood: float = 0.0
        self._drift_energy: float = 0.0

    def update_signals(self, tension: float = 0.0, dominance: float = 0.5,
                       personality_zone: str = "guardian",
                       music_mood: Optional[str] = None,
                       activity: str = "away",
                       face_id: Optional[str] = None,
                       music_energy: float = 0.0):
        """Alle Signale updaten.

        Args:
            tension: CoreIntegrator Tension (0.0-1.0)
            dominance: CoreIntegrator Dominance (-1.0 bis +1.0)
            personality_zone: "guardian" / "shadow" / "berserker"
            music_mood: Aus Spotify Bridge ("aggressive", "dark", "euphoric", etc.)
            activity: Aus ActivityAnalyzer
            face_id: Erkannte Person
            music_energy: Spotify Audio Energy (0.0-1.0)
        """
        with self._lock:
            self._tension = tension
            self._dominance = dominance
            self._personality_zone = personality_zone
            self._music_mood = music_mood
            self._activity = activity
            self._face_id = face_id
            self._music_energy = music_energy

    def evaluate(self) -> Optional[str]:
        """Mood neu berechnen.

        Returns:
            Neuer Mood wenn gewechselt, None wenn gleich
        """
        with self._lock:
            candidate = self._classify()
            now = time.monotonic()

            if candidate != self._candidate:
                self._candidate = candidate
                self._candidate_since = now
                return None

            if candidate == self._mood:
                return None

            if (now - self._candidate_since) < STABILITY_SECONDS:
                return None

            old_mood = self._mood
            self._mood = candidate

        # Event publizieren (ausserhalb Lock)
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="mood_changed",
                source="mood_engine",
                priority=5,
                payload={
                    "mood": candidate,
                    "previous_mood": old_mood,
                    "tension": round(self._tension, 3),
                    "dominance": round(self._dominance, 3),
                    "activity": self._activity,
                    "music_mood": self._music_mood,
                },
            )
            logger.info(f"[MOOD] {old_mood} → {candidate} "
                        f"(T={self._tension:.2f} D={self._dominance:+.2f})")
        except Exception as e:
            logger.debug(f"[MOOD] Event publish: {e}")

        return candidate

    def set_drift_baseline(self, mood: float = 0.0, energy: float = 0.0) -> None:
        """Charakter-Drift-Baseline anwenden (vom Distiller, Phase 4 Gate 1.5).

        Args:
            mood:   Long-term Mood-Shift (-1.0..+1.0). Positiv = besser gelaunt.
                    Wird als negative Bias auf Tension angewendet (effective_t = t - mood).
            energy: Long-term Energy-Shift (-1.0..+1.0). Positiv = mehr Energie.
                    Wird additiv auf music_energy angewendet.
        """
        with self._lock:
            self._drift_mood = max(-1.0, min(1.0, float(mood)))
            self._drift_energy = max(-1.0, min(1.0, float(energy)))
        logger.info(
            f"[MOOD] Drift-Baseline gesetzt: mood={self._drift_mood:+.3f} "
            f"energy={self._drift_energy:+.3f}"
        )

    def _classify(self) -> str:
        """Mood aus Signalen klassifizieren (unter Lock aufrufen).

        Drift-Baseline wird hier additiv eingerechnet:
          - effective_t = tension - drift_mood (positive Drift = beruhigend)
          - effective_e = music_energy + drift_energy (positive Drift = energischer)
        """
        t = max(0.0, min(1.0, self._tension - self._drift_mood))
        zone = self._personality_zone
        music = self._music_mood
        activity = self._activity
        energy = max(0.0, min(1.0, self._music_energy + self._drift_energy))

        # Dark: Shadow-Zone + hohe Tension + dunkle Musik
        if zone == "shadow" and t > 0.6 and music in ("dark", "aggressive"):
            return "dark"

        # Agitated: Sehr hohe Tension oder Berserker
        if t > 0.8 or zone == "berserker":
            return "agitated"

        # Alert: Hohe Tension oder unbekannte Person
        if t > 0.5 or (self._face_id == "unknown"):
            return "alert"

        # Euphoric: Party oder hohe Musik-Energy + Guardian
        if activity == "party" or (energy > 0.7 and zone == "guardian"):
            return "euphoric"

        # Focused: Person anwesend, moderate Tension, working
        if self._face_id and self._face_id != "unknown" and activity == "working":
            return "focused"

        # Calm: Default bei niedriger Tension
        return "calm"

    @property
    def current_mood(self) -> str:
        with self._lock:
            return self._mood

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "mood": self._mood,
                "tension": round(self._tension, 3),
                "dominance": round(self._dominance, 3),
                "personality_zone": self._personality_zone,
                "music_mood": self._music_mood,
                "activity": self._activity,
                "music_energy": round(self._music_energy, 2),
                "drift_mood": round(self._drift_mood, 3),
                "drift_energy": round(self._drift_energy, 3),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[MoodEngine] = None
_instance_lock = threading.Lock()


def get_mood_engine() -> MoodEngine:
    """Singleton-Zugriff auf Mood Engine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MoodEngine()
    return _instance
