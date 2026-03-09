#!/usr/bin/env python3
"""
M.O.L.O.C.H. Tracking Quality Monitor
=======================================
Bewertet wie gut das aktuelle Tracking laeuft.
Score 0.0-1.0. Unter 0.4 → MOLOCH übernimmt Steuerung.

Score-Berechnung (4 Komponenten):
  bbox_stability    = 1.0 - (bbox_jitter / bbox_size)    Gewicht 0.3
  frame_centering   = 1.0 - (center_offset / frame_size) Gewicht 0.3
  jitter_penalty    = 1.0 wenn PTZ-Wechsel < 3x in 2s   Gewicht 0.2
  camera_confidence = geschaetzt aus bbox-Stabilitaet    Gewicht 0.2

Events:
  tracking.quality_update      — Score-Update
  tracking.takeover_required   — Score < 0.4
  tracking.camera_sufficient   — Score > 0.8
  tracking.jitter_detected     — PTZ-Richtungswechsel > 3x in 2s
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, Callable, Tuple

import numpy as np

logger = logging.getLogger("TrackingQualityMonitor")

# Schwellwerte
SCORE_TAKEOVER = 0.4
SCORE_SUFFICIENT = 0.8

# PTZ Jitter: mehr als N Richtungswechsel in WINDOW Sekunden = Jitter
JITTER_MAX_SWITCHES = 3
JITTER_WINDOW_SEC = 2.0

# Ringbuffer-Groesse
PTZ_HISTORY_LEN = 20   # ~2s bei 10 Hz Sampling
BBOX_HISTORY_LEN = 30  # ~3s bei 10 Hz


class TrackingQualityMonitor:
    """Bewertet Tracking-Qualitaet aus PFrame-Daten."""

    def __init__(self):
        self._lock = threading.Lock()

        # Ringbuffer: PTZ-Positionen (pan, tilt) + Zeitstempel
        self._ptz_history: deque = deque(maxlen=PTZ_HISTORY_LEN)
        # Ringbuffer: bbox_center (cx, cy) normalisiert + Zeitstempel
        self._bbox_history: deque = deque(maxlen=BBOX_HISTORY_LEN)

        # Aktueller Score
        self._score: float = 1.0
        self._last_state: str = "ok"   # "ok" | "takeover" | "sufficient"

        # Callbacks (optional, werden von Service gesetzt)
        # Signatur: cb(topic: str, data: dict)
        self.on_event: Optional[Callable[[str, dict], None]] = None

    # =========================================================================
    # Haupt-Update (aufgerufen vom Vision-Poll-Thread ~10 Hz)
    # =========================================================================

    def update(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        ptz_pan: Optional[float],
        ptz_tilt: Optional[float],
        face_confidence: float = 0.0,
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> float:
        """Neuen Frame verarbeiten. Gibt aktuellen Score zurueck.

        Args:
            bbox:             (x1, y1, x2, y2) normalisiert 0..1 oder None
            ptz_pan:          Aktueller Pan-Winkel (oder None)
            ptz_tilt:         Aktueller Tilt-Winkel (oder None)
            face_confidence:  SCRFD Face-Confidence (0..1)
            frame_w/h:        Frame-Groesse fuer Zentrierung
        Returns:
            Score 0.0 - 1.0
        """
        now = time.time()
        with self._lock:
            # PTZ-History aktualisieren
            if ptz_pan is not None and ptz_tilt is not None:
                self._ptz_history.append((now, ptz_pan, ptz_tilt))

            # BBox-History aktualisieren
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1
                self._bbox_history.append((now, cx, cy, bw, bh))

            score = self._compute_score(bbox, face_confidence)
            self._score = score
            self._check_transitions(score)
            return score

    # =========================================================================
    # Score-Berechnung
    # =========================================================================

    def _compute_score(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        face_confidence: float,
    ) -> float:
        """Score 0.0-1.0 berechnen."""

        s_bbox = self._score_bbox_stability()
        s_center = self._score_frame_centering(bbox)
        s_jitter = self._score_jitter_penalty()
        s_cam = self._score_camera_confidence(face_confidence)

        score = (
            s_bbox * 0.3
            + s_center * 0.3
            + s_jitter * 0.2
            + s_cam * 0.2
        )
        return float(np.clip(score, 0.0, 1.0))

    def _score_bbox_stability(self) -> float:
        """1.0 - (jitter / bbox_size). 1.0 wenn stabil."""
        if len(self._bbox_history) < 4:
            return 1.0
        entries = list(self._bbox_history)[-10:]
        centers = np.array([[e[1], e[2]] for e in entries])
        sizes = np.array([(e[3] + e[4]) / 2.0 for e in entries])
        if sizes.mean() < 1e-6:
            return 0.5
        jitter = np.std(centers, axis=0).mean()
        ratio = jitter / sizes.mean()
        return float(np.clip(1.0 - ratio * 5.0, 0.0, 1.0))

    def _score_frame_centering(
        self, bbox: Optional[Tuple[float, float, float, float]]
    ) -> float:
        """1.0 wenn bbox-Mitte nahe Bild-Mitte."""
        if bbox is None:
            return 0.5
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # Abstand zur Bildmitte (0.5, 0.5) in normalisierten Koordinaten
        offset = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        return float(np.clip(1.0 - offset * 2.0, 0.0, 1.0))

    def _score_jitter_penalty(self) -> float:
        """1.0 wenn PTZ-Richtungswechsel < JITTER_MAX_SWITCHES in 2s."""
        switches = self._count_ptz_direction_switches()
        if switches >= JITTER_MAX_SWITCHES:
            return 0.0
        return 1.0

    def _score_camera_confidence(self, face_confidence: float) -> float:
        """Kamera-Konfidenz: face_confidence als Proxy."""
        if face_confidence > 0:
            return float(np.clip(face_confidence, 0.0, 1.0))
        # Kein Face → aus BBox-Stabilitaet schaetzen
        if len(self._bbox_history) < 2:
            return 0.5
        entries = list(self._bbox_history)[-5:]
        sizes = np.array([(e[3] + e[4]) / 2.0 for e in entries])
        return float(np.clip(sizes.mean() * 2.0, 0.0, 1.0))

    # =========================================================================
    # PTZ Jitter Analyse
    # =========================================================================

    def _count_ptz_direction_switches(self) -> int:
        """Zaehlt Pan-Richtungswechsel im Zeitfenster JITTER_WINDOW_SEC."""
        now = time.time()
        cutoff = now - JITTER_WINDOW_SEC
        entries = [e for e in self._ptz_history if e[0] >= cutoff]
        if len(entries) < 3:
            return 0
        pans = [e[1] for e in entries]
        switches = 0
        last_dir = 0
        for i in range(1, len(pans)):
            delta = pans[i] - pans[i - 1]
            if abs(delta) < 0.1:
                continue
            direction = 1 if delta > 0 else -1
            if last_dir != 0 and direction != last_dir:
                switches += 1
            last_dir = direction
        return switches

    # =========================================================================
    # State-Transitions
    # =========================================================================

    def _check_transitions(self, score: float):
        """Events publishen bei Schwellwert-Ueberschreitung."""
        new_state = self._last_state

        if score < SCORE_TAKEOVER:
            new_state = "takeover"
        elif score > SCORE_SUFFICIENT:
            new_state = "sufficient"
        else:
            new_state = "ok"

        # PTZ Jitter: immer pruefen, nicht nur bei State-Wechsel
        switches = self._count_ptz_direction_switches()
        if switches >= JITTER_MAX_SWITCHES:
            self._emit("tracking.jitter_detected", {
                "ptz_switches": switches,
                "window_sec": JITTER_WINDOW_SEC,
            })
            logger.warning(f"[TQM] PTZ Jitter: {switches} Wechsel in {JITTER_WINDOW_SEC}s")

        if new_state != self._last_state:
            self._last_state = new_state
            if new_state == "takeover":
                self._emit("tracking.takeover_required", {
                    "score": round(score, 3),
                    "reason": "score_below_threshold",
                })
                logger.warning(f"[TQM] Takeover required (score={score:.3f})")
            elif new_state == "sufficient":
                self._emit("tracking.camera_sufficient", {
                    "score": round(score, 3),
                })
                logger.info(f"[TQM] Camera sufficient (score={score:.3f})")

        self._emit("tracking.quality_update", {
            "score": round(score, 3),
            "state": new_state,
        })

    def _emit(self, topic: str, data: dict):
        """Event publishen via Callback (falls gesetzt)."""
        if self.on_event:
            try:
                self.on_event(topic, data)
            except Exception as e:
                logger.warning(f"[TQM] on_event Fehler: {e}")

    # =========================================================================
    # Status
    # =========================================================================

    @property
    def score(self) -> float:
        with self._lock:
            return self._score

    def get_status(self) -> dict:
        with self._lock:
            return {
                "tracking_quality_score": round(self._score, 3),
                "tracking_quality_state": self._last_state,
                "ptz_history_len": len(self._ptz_history),
                "bbox_history_len": len(self._bbox_history),
            }


# Singleton
_instance: Optional[TrackingQualityMonitor] = None
_lock = threading.Lock()


def get_tracking_quality_monitor() -> TrackingQualityMonitor:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TrackingQualityMonitor()
    return _instance
