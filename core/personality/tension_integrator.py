#!/usr/bin/env python3
"""
M.O.L.O.C.H. Tension Integrator — Gate-3 Awareness → CoreIntegrator Bridge
=============================================================================

Empfaengt Gate-3 Events (context_update, activity_changed, motion_state_changed)
und mappt sie auf CoreIntegrator Inputs (tension/dominance Deltas).

Erweitert den bestehenden CoreIntegrator NICHT — fuettert ihn nur mit
neuen Signalen aus den Awareness-Modulen.

Mapping:
  - context_update Score hoch → tension sinkt (alles gut)
  - context_update alertness hoch → tension steigt (Wachsamkeit)
  - activity party → dominance hoch (Energie)
  - activity alone → dominance leicht negativ (ruhig)
  - motion approaching → tension leicht hoch (Aufmerksamkeit)

Singleton: get_tension_integrator()
Gate 4: Emergent Personality
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("MolochTensionIntegrator")

# Gewichtung der Awareness-Signale auf CoreIntegrator
CONTEXT_SCORE_TENSION_WEIGHT = -0.05     # Hoher Score → Tension sinkt
ALERTNESS_TENSION_WEIGHT = 0.08          # Hohe Alertness → Tension steigt
ACTIVITY_DOMINANCE_MAP = {
    "party": 0.15,           # Party → Dominance hoch (Energie/Guardian)
    "conversation": 0.08,    # Gespraech → leicht Guardian
    "working": 0.03,         # Arbeiten → neutral-positiv
    "alone": -0.05,          # Allein → leicht Shadow
    "away": -0.02,           # Weg → minimal Shadow-Drift
}
MOTION_TENSION_MAP = {
    "approaching": 0.04,     # Jemand kommt → leichte Wachsamkeit
    "leaving": -0.02,        # Jemand geht → Entspannung
    "walking": 0.01,         # Bewegung → minimal
    "stationary": 0.0,       # Still → kein Effekt
}

# Beleidigung-Keywords (Deutsch + Englisch) — Tension-Spike bei verbaler Aggression
_RUDENESS_KEYWORDS = [
    "blöd", "dumm", "scheiß", "idiot", "nutzlos", "kaputt", "schrott", "müll",
    "bescheuert", "depp", "doof", "schwachsinn", "mist", "dreck", "arschloch",
    "wichser", "hurensohn", "vollidiot", "trottel", "spacken",
    "stupid", "useless", "trash", "garbage", "broken", "crap", "fuck", "shit",
    "asshole", "idiot", "moron", "dumbass",
]
# Rate-Limiting: Minimum Sekunden zwischen Rudeness-Spikes
_RUDENESS_COOLDOWN_S = 10.0


class TensionIntegrator:
    """Bridge zwischen Gate-3 Awareness und CoreIntegrator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._core_integrator = None
        self._last_context_score = 0.5
        self._last_alertness = 0.2
        self._last_activity = "away"
        self._last_motion = "stationary"
        self._last_rudeness_ts = 0.0
        self._last_rudeness_boost = 0.0

    def set_core_integrator(self, ci):
        """CoreIntegrator-Referenz setzen (lazy init) + Event-Subscriptions."""
        self._core_integrator = ci
        # Whisper-Rudeness Subscription — self-subscribe statt Service-Verdrahtung
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().subscribe("whisper.result", self.on_whisper_result, priority=5)
            logger.info("[TENSION] whisper.result Subscription aktiv (Rudeness-Detection)")
        except Exception as e:
            logger.warning(f"[TENSION] whisper.result Subscription fehlgeschlagen: {e}")

    def on_context_update(self, event: Dict[str, Any]):
        """Event-Handler fuer context_update Events.

        Mappt Score und Alertness auf Tension-Deltas.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        score = payload.get("score", 0.5)
        alertness = payload.get("alertness", 0.2)

        with self._lock:
            self._last_context_score = score
            self._last_alertness = alertness

        # Score → Tension: hoher Score = alles gut = Tension sinkt
        tension_delta = (score - 0.5) * CONTEXT_SCORE_TENSION_WEIGHT
        # Alertness → Tension: direkte Zuordnung
        tension_delta += alertness * ALERTNESS_TENSION_WEIGHT

        # Via update_input an CoreIntegrator — positiv = tension steigt
        # GEDAEMPFT: tension_delta direkt als conflict_input erzeugt Feedback-Loop!
        # Stattdessen: nur starke Deltas (>0.3) als schwachen Input weitergeben
        if tension_delta > 0.3:
            self._core_integrator.update_input("awareness", "conflict_input", min(0.2, tension_delta * 0.3))
        elif tension_delta < -0.1:
            self._core_integrator.update_input("awareness", "respect_score", min(0.3, abs(tension_delta)))

    def on_activity_changed(self, event: Dict[str, Any]):
        """Event-Handler fuer activity_changed Events.

        Mappt Activity auf Dominance-Shifts.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        activity = payload.get("activity", "away")

        with self._lock:
            self._last_activity = activity

        dominance_delta = ACTIVITY_DOMINANCE_MAP.get(activity, 0.0)
        if dominance_delta > 0:
            self._core_integrator.update_input("awareness", "markus_recognized", dominance_delta)
        elif dominance_delta < 0:
            self._core_integrator.update_input("awareness", "unknown_person", abs(dominance_delta))

    def on_motion_state_changed(self, event: Dict[str, Any]):
        """Event-Handler fuer motion_state_changed Events.

        Mappt Motion auf Tension-Shifts.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        motion = payload.get("state", "stationary")

        with self._lock:
            self._last_motion = motion

        tension_delta = MOTION_TENSION_MAP.get(motion, 0.0)
        if tension_delta > 0:
            self._core_integrator.update_input("awareness", "person_detected", tension_delta)
        elif tension_delta < 0:
            self._core_integrator.update_input("awareness", "respect_score", abs(tension_delta))

    # ================================================================
    # WHISPER RUDENESS DETECTION — Tension-Spike bei Beleidigungen
    # ================================================================

    def on_whisper_result(self, event: Dict[str, Any]):
        """Event-Handler fuer whisper.result Events.

        Prueft transkribierten Text auf Beleidigungen und erhoeht Tension.
        Rate-Limited: max 1 Spike pro _RUDENESS_COOLDOWN_S Sekunden.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        text = payload.get("text", "")
        if not text or len(text) < 3:
            return

        boost = self._detect_rudeness(text)
        if boost <= 0.0:
            return

        now = time.time()
        with self._lock:
            # Rate-Limiting — kein Dauerfeuer
            if now - self._last_rudeness_ts < _RUDENESS_COOLDOWN_S:
                logger.debug(f"[TENSION] Rudeness cooldown aktiv, ignoriere ({boost:.2f})")
                return
            self._last_rudeness_ts = now
            self._last_rudeness_boost = boost

        # Tension-Spike via CoreIntegrator — conflict_input erhoeht Tension
        self._core_integrator.update_input("voice", "conflict_input", boost)
        logger.info(f"[TENSION] Rudeness erkannt! Boost={boost:.2f} Text='{text[:50]}'")

    def _detect_rudeness(self, text: str) -> float:
        """Gibt Tension-Boost zurueck: 0.0 (keine Beleidigung) bis 0.8 (massive Beleidigung)."""
        text_lower = text.lower()
        hits = sum(1 for kw in _RUDENESS_KEYWORDS if kw in text_lower)
        if hits == 0:
            return 0.0
        elif hits == 1:
            return 0.3
        else:
            return min(0.5 + (hits - 2) * 0.1, 0.8)

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer Debugging/IPC."""
        with self._lock:
            return {
                "context_score": round(self._last_context_score, 3),
                "alertness": round(self._last_alertness, 3),
                "activity": self._last_activity,
                "motion": self._last_motion,
                "last_rudeness_boost": round(self._last_rudeness_boost, 3),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[TensionIntegrator] = None
_instance_lock = threading.Lock()


def get_tension_integrator() -> TensionIntegrator:
    """Singleton-Zugriff auf Tension Integrator."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TensionIntegrator()
    return _instance
