#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Hand Gesture Detection
=====================================

Erkennt Handgesten aus 21 MediaPipe Hand-Landmarks.
Ersetzt den falschen Aufruf des Body-Pose GestureDetectors
mit Hand-Landmarks (Audit-Fix W1, 2026-02-23).

MediaPipe Hand Landmark Indizes:
  0: WRIST
  1: THUMB_CMC, 2: THUMB_MCP, 3: THUMB_IP, 4: THUMB_TIP
  5: INDEX_MCP, 6: INDEX_PIP, 7: INDEX_DIP, 8: INDEX_TIP
  9: MIDDLE_MCP, 10: MIDDLE_PIP, 11: MIDDLE_DIP, 12: MIDDLE_TIP
  13: RING_MCP, 14: RING_PIP, 15: RING_DIP, 16: RING_TIP
  17: PINKY_MCP, 18: PINKY_PIP, 19: PINKY_DIP, 20: PINKY_TIP
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class HandGestureType(Enum):
    """Erkennbare Handgesten."""
    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    OPEN_HAND = "open_hand"
    FIST = "fist"
    POINTING = "pointing"


@dataclass
class HandGesture:
    """Erkannte Geste mit Confidence und Dauer."""
    type: HandGestureType
    confidence: float
    duration_ms: int = 0
    hand: str = "R"

    def __str__(self):
        return f"{self.type.value} ({self.confidence:.0%}, {self.duration_ms}ms)"


# Landmark-Indizes
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class HandGestureDetector:
    """Erkennt Handgesten aus 21 MediaPipe Hand-Landmarks."""

    def __init__(self):
        self._current_gesture: Optional[HandGesture] = None
        self._gesture_start_time: float = 0
        self.gestures_detected: int = 0
        logger.info("HandGestureDetector initialized (21-Punkt MediaPipe)")

    def detect(self, landmarks: np.ndarray, handedness: str = "R") -> Optional[HandGesture]:
        """Geste aus 21 Hand-Landmarks erkennen.

        Args:
            landmarks: np.ndarray (21, 2) oder (21, 3) — x, y normalisiert [0,1]
            handedness: "L" oder "R"

        Returns:
            Erkannte HandGesture oder None
        """
        if landmarks is None or len(landmarks) < 21:
            return None

        lm = landmarks[:, :2]  # Nur x, y verwenden

        # Finger-Status ermitteln
        fingers = self._get_finger_states(lm, handedness)
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers

        # Gesten nach Prioritaet pruefen
        gesture = None

        # 1. Thumbs Up: Nur Daumen offen, Rest geschlossen
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            # Pruefen ob Daumen nach OBEN zeigt (tip.y < mcp.y)
            if lm[THUMB_TIP][1] < lm[THUMB_MCP][1]:
                gesture = HandGesture(HandGestureType.THUMBS_UP, 0.9, hand=handedness)
            else:
                gesture = HandGesture(HandGestureType.THUMBS_DOWN, 0.85, hand=handedness)

        # 2. Peace: Index + Middle offen, Ring + Pinky geschlossen
        if not gesture and index_up and middle_up and not ring_up and not pinky_up:
            gesture = HandGesture(HandGestureType.PEACE, 0.9, hand=handedness)

        # 3. Pointing: Nur Index offen
        if not gesture and index_up and not middle_up and not ring_up and not pinky_up:
            gesture = HandGesture(HandGestureType.POINTING, 0.85, hand=handedness)

        # 4. Open Hand: Alle 5 Finger offen
        if not gesture and thumb_up and index_up and middle_up and ring_up and pinky_up:
            gesture = HandGesture(HandGestureType.OPEN_HAND, 0.9, hand=handedness)

        # 5. Fist: Alle Finger geschlossen
        if not gesture and not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            gesture = HandGesture(HandGestureType.FIST, 0.85, hand=handedness)

        if gesture:
            return self._update_gesture(gesture)

        # Keine erkannte Geste
        if self._current_gesture:
            self._current_gesture = None
        return None

    def _get_finger_states(self, lm: np.ndarray, handedness: str) -> tuple:
        """Ermittle ob jeder Finger offen oder geschlossen ist.

        Finger offen = Tip hoeher als PIP (y-Achse invertiert: kleiner = hoeher)
        Daumen-Sonderfall: x-Achse (seitlich vom MCP weg)

        Returns:
            (thumb_up, index_up, middle_up, ring_up, pinky_up)
        """
        # Daumen: Tip weiter vom Handgelenk als MCP (horizontal)
        # Rechte Hand: Tip links von MCP = offen (tip.x < mcp.x in norm. Coords)
        # Linke Hand: Tip rechts von MCP = offen
        if handedness == "R":
            thumb_up = lm[THUMB_TIP][0] < lm[THUMB_MCP][0]
        else:
            thumb_up = lm[THUMB_TIP][0] > lm[THUMB_MCP][0]

        # Andere Finger: Tip hoeher als PIP-Gelenk (y kleiner = hoeher)
        index_up = lm[INDEX_TIP][1] < lm[INDEX_PIP][1]
        middle_up = lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1]
        ring_up = lm[RING_TIP][1] < lm[RING_PIP][1]
        pinky_up = lm[PINKY_TIP][1] < lm[PINKY_PIP][1]

        return thumb_up, index_up, middle_up, ring_up, pinky_up

    def _update_gesture(self, gesture: HandGesture) -> HandGesture:
        """Gesten-State aktualisieren und Dauer berechnen."""
        now = time.time()

        if (self._current_gesture and
                self._current_gesture.type == gesture.type):
            gesture.duration_ms = int((now - self._gesture_start_time) * 1000)
        else:
            self._gesture_start_time = now
            self._current_gesture = gesture
            self.gestures_detected += 1
            logger.info(f"[HAND_GESTURE] Erkannt: {gesture.type.value}")

        return gesture

    def reset(self):
        """Gesten-State zuruecksetzen."""
        self._current_gesture = None
        self._gesture_start_time = 0
