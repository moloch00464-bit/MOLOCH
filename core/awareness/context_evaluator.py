#!/usr/bin/env python3
"""
M.O.L.O.C.H. Context Evaluator — Situationsbewertung aus allen Awareness-Modulen
==================================================================================

Kombiniert RoomMap + MotionAnalyzer + ActivityAnalyzer + EpisodicMemory
zu einer Kontextbewertung. Score 0-1 pro Situation.

Methode evaluate() berechnet Kontext und publiziert context_update Event.

Singleton: get_context_evaluator()
Gate 3: Situational Awareness
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger("MolochContextEvaluator")

# Gewichtung der Situationsfaktoren
WEIGHTS = {
    "familiarity": 0.30,    # Wie bekannt ist die Person? (Episodic Memory)
    "comfort": 0.25,        # Wie komfortabel ist die Situation? (Activity + Zone)
    "alertness": 0.25,      # Wie aufmerksam sollte Moloch sein? (Motion + Unknown)
    "engagement": 0.20,     # Wie engagiert ist die Interaktion? (Voice + Approaching)
}


class ContextEvaluator:
    """Kombiniert alle Awareness-Module zu einer Kontextbewertung."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_context: Dict[str, Any] = {}
        self._last_score: float = 0.0
        self._last_evaluate: float = 0.0

    def evaluate(self, room_zone: Optional[str] = None,
                 motion_state: str = "stationary",
                 activity: str = "away",
                 face_id: Optional[str] = None,
                 face_confidence: float = 0.0,
                 person_count: int = 0,
                 voice_active: bool = False,
                 episodic_score: float = 0.0) -> Dict[str, Any]:
        """Kontext evaluieren und Event publizieren.

        Args:
            room_zone: Aktuelle Raumzone
            motion_state: Aus MotionAnalyzer
            activity: Aus ActivityAnalyzer
            face_id: Erkannte Person
            face_confidence: ArcFace Confidence
            person_count: Anzahl Personen
            voice_active: Sprach-Aktivitaet
            episodic_score: Vertrautheits-Score aus EpisodicMemory (0-1)

        Returns:
            Dict mit Scores und Kontext
        """
        # Familiarity: Bekannte Person + Episodic Memory
        familiarity = 0.0
        if face_id and face_id != "unknown":
            familiarity = min(1.0, face_confidence + episodic_score * 0.5)
        elif face_id == "unknown":
            familiarity = 0.1

        # Comfort: Ruhige Aktivitaet + bekannte Zone
        comfort_map = {"alone": 0.7, "working": 0.8, "conversation": 0.6,
                       "party": 0.3, "away": 0.5}
        zone_comfort = {"schreibtisch": 0.2, "sofa": 0.2, "mitte": 0.1,
                        "tuer": -0.1, "fenster": 0.0}
        comfort = comfort_map.get(activity, 0.5)
        comfort += zone_comfort.get(room_zone, 0.0)
        comfort = max(0.0, min(1.0, comfort))

        # Alertness: Unbekannte Personen, Bewegung, Tuer-Zone
        alertness = 0.2  # Baseline
        if face_id == "unknown" or (person_count > 0 and not face_id):
            alertness += 0.4
        if motion_state == "approaching":
            alertness += 0.2
        if room_zone == "tuer":
            alertness += 0.2
        if person_count > 2:
            alertness += 0.1
        alertness = min(1.0, alertness)

        # Engagement: Interaktion aktiv
        engagement = 0.1  # Baseline
        if voice_active:
            engagement += 0.4
        if motion_state == "approaching":
            engagement += 0.2
        if face_id and face_id != "unknown":
            engagement += 0.2
        if activity == "conversation":
            engagement += 0.1
        engagement = min(1.0, engagement)

        # Gewichteter Gesamtscore
        score = (
            WEIGHTS["familiarity"] * familiarity +
            WEIGHTS["comfort"] * comfort +
            WEIGHTS["alertness"] * (1.0 - alertness) +  # Invertiert: hohe Alertness = niedrigerer Score
            WEIGHTS["engagement"] * engagement
        )
        score = round(max(0.0, min(1.0, score)), 3)

        context = {
            "score": score,
            "familiarity": round(familiarity, 3),
            "comfort": round(comfort, 3),
            "alertness": round(alertness, 3),
            "engagement": round(engagement, 3),
            "activity": activity,
            "zone": room_zone,
            "motion": motion_state,
            "face_id": face_id,
            "person_count": person_count,
            "timestamp": time.time(),
        }

        with self._lock:
            self._last_context = context
            self._last_score = score
            self._last_evaluate = time.time()

        # Event publizieren
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="context_update",
                source="context_evaluator",
                priority=5,
                payload=context,
            )
        except Exception as e:
            logger.debug(f"[CONTEXT] Event publish: {e}")

        return context

    @property
    def last_score(self) -> float:
        with self._lock:
            return self._last_score

    def get_state(self) -> Dict[str, Any]:
        """Letzter Kontext fuer IPC/Panel."""
        with self._lock:
            return dict(self._last_context) if self._last_context else {"score": 0.0}


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[ContextEvaluator] = None
_instance_lock = threading.Lock()


def get_context_evaluator() -> ContextEvaluator:
    """Singleton-Zugriff auf Context Evaluator."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ContextEvaluator()
    return _instance
