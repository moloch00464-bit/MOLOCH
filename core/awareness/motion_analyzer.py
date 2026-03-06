#!/usr/bin/env python3
"""
M.O.L.O.C.H. Motion Analyzer — Bewegungsanalyse aus BBox-Deltas
=================================================================

Analysiert Person-BBox Positionen ueber Zeit und erkennt Bewegungszustaende:
  - stationary: Person steht/sitzt still
  - walking: Moderate Bewegung
  - approaching: Person kommt naeher (BBox wird groesser)
  - leaving: Person entfernt sich (BBox wird kleiner)

Publiziert motion_state_changed Event bei Zustandswechsel.

Singleton: get_motion_analyzer()
Gate 3: Situational Awareness
"""

import logging
import threading
import time
from collections import deque
from typing import Optional, Dict, Any

logger = logging.getLogger("MolochMotionAnalyzer")

# Schwellwerte fuer Bewegungserkennung
POSITION_DELTA_STATIONARY = 0.02   # Normalisierte BBox-Center Bewegung pro Tick
POSITION_DELTA_WALKING = 0.05      # Ab hier "walking"
SIZE_DELTA_APPROACHING = 0.01      # BBox-Flaeche waechst → approaching
SIZE_DELTA_LEAVING = -0.01         # BBox-Flaeche schrumpft → leaving
HISTORY_SIZE = 10                  # Frames fuer Glaettung
MIN_STABLE_TICKS = 3              # Ticks bevor State-Wechsel


class MotionAnalyzer:
    """Bewegungsanalyse aus Person-BBox Deltas."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "stationary"
        self._state_counter = 0
        self._history: deque = deque(maxlen=HISTORY_SIZE)
        self._last_bbox: Optional[tuple] = None
        self._last_area: float = 0.0

    def update(self, person_detected: bool, face_bbox: Optional[tuple] = None,
               distance_ratio: float = 0.0) -> Optional[str]:
        """Neuen Frame verarbeiten und Bewegungszustand bestimmen.

        Args:
            person_detected: Ist eine Person sichtbar?
            face_bbox: (x1, y1, x2, y2) normalisiert, oder None
            distance_ratio: BBox-Flaeche / Frame-Flaeche

        Returns:
            Neuer State wenn gewechselt, None wenn gleich
        """
        if not person_detected:
            with self._lock:
                self._last_bbox = None
                self._last_area = 0.0
                self._history.clear()
                if self._state != "stationary":
                    self._state = "stationary"
                    self._state_counter = 0
                    return "stationary"
            return None

        # BBox-Center und Flaeche berechnen
        current_area = distance_ratio
        center_delta = 0.0
        size_delta = 0.0

        if face_bbox and len(face_bbox) == 4:
            cx = (face_bbox[0] + face_bbox[2]) / 2
            cy = (face_bbox[1] + face_bbox[3]) / 2
            area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
            current_area = area if current_area == 0 else current_area

            with self._lock:
                if self._last_bbox and len(self._last_bbox) == 4:
                    last_cx = (self._last_bbox[0] + self._last_bbox[2]) / 2
                    last_cy = (self._last_bbox[1] + self._last_bbox[3]) / 2
                    center_delta = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                    size_delta = current_area - self._last_area

                self._last_bbox = face_bbox
                self._last_area = current_area
        else:
            with self._lock:
                # Nur distance_ratio verfuegbar
                if self._last_area > 0:
                    size_delta = current_area - self._last_area
                self._last_area = current_area

        # History fuer Glaettung
        with self._lock:
            self._history.append({
                "center_delta": center_delta,
                "size_delta": size_delta,
                "timestamp": time.time(),
            })

            if len(self._history) < 2:
                return None

            # Durchschnittliche Deltas
            avg_center = sum(h["center_delta"] for h in self._history) / len(self._history)
            avg_size = sum(h["size_delta"] for h in self._history) / len(self._history)

            # State bestimmen
            if avg_size > SIZE_DELTA_APPROACHING and avg_center > POSITION_DELTA_STATIONARY:
                candidate = "approaching"
            elif avg_size < SIZE_DELTA_LEAVING and avg_center > POSITION_DELTA_STATIONARY:
                candidate = "leaving"
            elif avg_center > POSITION_DELTA_WALKING:
                candidate = "walking"
            else:
                candidate = "stationary"

            # Hysterese: State muss MIN_STABLE_TICKS stabil sein
            if candidate == self._state:
                self._state_counter = 0
                return None

            self._state_counter += 1
            if self._state_counter >= MIN_STABLE_TICKS:
                old_state = self._state
                self._state = candidate
                self._state_counter = 0

                # Event publizieren
                try:
                    from core.moloch_event_bus import get_event_bus
                    get_event_bus().publish(
                        event_type="motion_state_changed",
                        source="motion_analyzer",
                        priority=5,
                        payload={
                            "state": candidate,
                            "previous_state": old_state,
                            "avg_center_delta": round(avg_center, 4),
                            "avg_size_delta": round(avg_size, 4),
                        },
                    )
                except Exception as e:
                    logger.debug(f"[MOTION] Event publish: {e}")

                return candidate

        return None

    @property
    def current_state(self) -> str:
        with self._lock:
            return self._state

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "motion_state": self._state,
                "history_size": len(self._history),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[MotionAnalyzer] = None
_instance_lock = threading.Lock()


def get_motion_analyzer() -> MotionAnalyzer:
    """Singleton-Zugriff auf Motion Analyzer."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MotionAnalyzer()
    return _instance
