#!/usr/bin/env python3
"""
M.O.L.O.C.H. Near Field Handler
=================================
Erkennt wenn Person zu nah an der Kamera ist → Smart-Tracking verliert den Kopf.
Triggert MOLOCH_TRACKING_NEAR State.

Trigger: Person-BBox-Hoehe > 70% des Frames UND keine Schulter/Kopf-Keypoints
         UND BBox-Mitte unten im Bild (y > 50% des Frames)
Rueckkehr: BBox-Hoehe < 40%

Events:
  tracking.near_field_enter  → an ptz_handover_controller
  tracking.near_field_exit   → an ptz_handover_controller
"""

import time
import logging
import threading
from typing import Optional, Callable, Tuple, List

logger = logging.getLogger("NearFieldHandler")

# Schwellwerte
NEAR_FIELD_ENTER_RATIO = 0.70  # BBox-Hoehe / Frame-Hoehe
NEAR_FIELD_EXIT_RATIO = 0.40   # Rueckkehr wenn kleiner
NEAR_FIELD_CENTER_Y_MIN = 0.5  # BBox-Mitte muss im unteren Bilddrittel sein

# Debounce: N aufeinanderfolgende Frames bis Zustand wechselt
ENTER_FRAMES = 3
EXIT_FRAMES = 5

# Keypoints die fuer "Kopf sichtbar" benoetigt werden
# YOLOv8-Pose Keypoint-Index: 0=nose, 5/6=shoulders, 7/8=elbows
HEAD_KEYPOINT_INDICES = [0, 5, 6]  # Nase + Schultern


class NearFieldHandler:
    """Erkennt Nah-Feld-Situationen aus PFrame-Daten."""

    def __init__(self):
        self._lock = threading.Lock()
        self._near_field_active: bool = False

        # Debounce-Zaehler
        self._enter_count: int = 0
        self._exit_count: int = 0

        # Callback (optional)
        # Signatur: cb(topic: str, data: dict)
        self.on_event: Optional[Callable[[str, dict], None]] = None

    # =========================================================================
    # Haupt-Update
    # =========================================================================

    def update(
        self,
        person_bbox: Optional[Tuple[float, float, float, float]],
        pose_keypoints: Optional[List] = None,
        frame_h: int = 1080,
    ) -> bool:
        """PFrame-Daten auswerten.

        Args:
            person_bbox:     (x1, y1, x2, y2) normalisiert 0..1 oder None
            pose_keypoints:  Liste von [x, y, confidence] pro Keypoint (oder None)
            frame_h:         Frame-Hoehe (fuer Logging; Berechnung auf normalisierten Coords)
        Returns:
            True wenn Near-Field aktiv
        """
        with self._lock:
            if person_bbox is None:
                # Keine Person → Near-Field immer verlassen
                self._enter_count = 0
                if self._near_field_active:
                    self._exit_count += 1
                    if self._exit_count >= EXIT_FRAMES:
                        self._set_near_field(False, "person_gone")
                return self._near_field_active

            near = self._check_near_field(person_bbox, pose_keypoints)

            if near and not self._near_field_active:
                self._enter_count += 1
                self._exit_count = 0
                if self._enter_count >= ENTER_FRAMES:
                    self._set_near_field(True, "bbox_height_trigger")
            elif not near and self._near_field_active:
                self._exit_count += 1
                self._enter_count = 0
                if self._exit_count >= EXIT_FRAMES:
                    self._set_near_field(False, "bbox_height_normal")
            else:
                self._enter_count = 0
                self._exit_count = 0

            return self._near_field_active

    # =========================================================================
    # Trigger-Logik
    # =========================================================================

    def _check_near_field(
        self,
        bbox: Tuple[float, float, float, float],
        keypoints: Optional[List],
    ) -> bool:
        """Alle Bedingungen pruefen."""
        x1, y1, x2, y2 = bbox

        # Bedingung 1: BBox-Hoehe > 70% des Frames
        bbox_height = y2 - y1
        if bbox_height < NEAR_FIELD_ENTER_RATIO:
            return False

        # Bedingung 2: BBox-Mitte unten im Bild
        bbox_center_y = (y1 + y2) / 2.0
        if bbox_center_y < NEAR_FIELD_CENTER_Y_MIN:
            return False

        # Bedingung 3: Keine sichtbaren Kopf/Schulter-Keypoints
        head_missing = self._head_keypoints_missing(keypoints)
        if not head_missing:
            return False

        return True

    def _check_exit(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        """Rueckkehrbedingung: BBox-Hoehe < 40%."""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        return bbox_height < NEAR_FIELD_EXIT_RATIO

    def _head_keypoints_missing(self, keypoints: Optional[List]) -> bool:
        """True wenn Kopf/Schultern nicht sichtbar (Confidence < 0.5)."""
        if keypoints is None or len(keypoints) == 0:
            # Keine Pose-Daten → konservativ: als fehlend werten
            return True
        for idx in HEAD_KEYPOINT_INDICES:
            if idx >= len(keypoints):
                continue
            kp = keypoints[idx]
            if len(kp) >= 3:
                conf = kp[2]
            elif len(kp) >= 1:
                conf = 1.0  # Kein Confidence-Feld → als sichtbar annehmen
            else:
                continue
            if conf >= 0.5:
                # Mindestens ein Head-Keypoint sichtbar
                return False
        return True

    # =========================================================================
    # State-Wechsel
    # =========================================================================

    def _set_near_field(self, active: bool, reason: str):
        self._near_field_active = active
        self._enter_count = 0
        self._exit_count = 0
        if active:
            logger.info(f"[NFH] Near-Field ENTER ({reason})")
            self._emit("tracking.near_field_enter", {
                "reason": reason,
                "timestamp": time.time(),
            })
        else:
            logger.info(f"[NFH] Near-Field EXIT ({reason})")
            self._emit("tracking.near_field_exit", {
                "reason": reason,
                "timestamp": time.time(),
            })

    # =========================================================================
    # Helpers
    # =========================================================================

    def _emit(self, topic: str, data: dict):
        if self.on_event:
            try:
                self.on_event(topic, data)
            except Exception as e:
                logger.warning(f"[NFH] on_event Fehler: {e}")

    @property
    def near_field_active(self) -> bool:
        with self._lock:
            return self._near_field_active

    def get_status(self) -> dict:
        with self._lock:
            return {
                "near_field_active": self._near_field_active,
                "enter_count": self._enter_count,
                "exit_count": self._exit_count,
            }


# Singleton
_instance: Optional[NearFieldHandler] = None
_lock = threading.Lock()


def get_near_field_handler() -> NearFieldHandler:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = NearFieldHandler()
    return _instance
