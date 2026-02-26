#!/usr/bin/env python3
"""
M.O.L.O.C.H. PTZ Tracker — Bewegungs-Analyse + restless_score
================================================================

Analysiert die Kamera-Bewegung (Pan/Tilt) und berechnet daraus:
  - ptz_velocity: Wie schnell bewegt sich die Kamera (Grad/Sekunde)
  - ptz_direction_changes: Richtungswechsel pro Minute
  - ptz_restless_score: 0.0 (still) bis 1.0 (hektisches Hin-und-Her)

Der restless_score wird als Input in den CoreIntegrator eingespeist:
  - Hoher Score (>0.6) → Tension steigt, Shadow-Gewichtung +0.1
  - Niedriger Score (<0.2) → Tension sinkt, Guardian-Gewichtung +0.1

Ring-Buffer der letzten 60 Sekunden PTZ-Positionen (1x pro Sekunde).

Regel 10: 1 Datei = 1 Aufgabe. Kommuniziert NUR ueber CoreIntegrator.
"""

import time
import threading
import logging
from typing import Optional, Dict
from collections import deque

logger = logging.getLogger("PTZTracker")

# Max-Velocity fuer Normalisierung (Kamera: ~30 deg/s bei Vollgas)
MAX_VELOCITY_DEG_S = 30.0
# Ring-Buffer Groesse (60 Sekunden bei 1Hz)
BUFFER_SIZE = 60


class PTZTracker:
    """Bewegungs-Analyse der PTZ-Kamera fuer emotionalen Input."""

    def __init__(self):
        # Ring-Buffer: (timestamp, pan_deg, tilt_deg)
        self._positions = deque(maxlen=BUFFER_SIZE)
        self._lock = threading.Lock()

        # Berechnete Werte
        self._velocity = 0.0          # Grad/Sekunde (Durchschnitt)
        self._direction_changes = 0   # Richtungswechsel in Buffer
        self._restless_score = 0.0    # 0.0-1.0

        # Tracker-Stage (fuer Status-JSON)
        self._stage = "idle"  # "idle" | "locked" | "searching"

        # CoreIntegrator (lazy init)
        self._core_integrator = None

        # Thread
        self._running = False
        self._thread = None

        logger.info("[PTZ-TRACKER] Initialisiert (Buffer=%d)", BUFFER_SIZE)

    def start(self):
        """1Hz Analyse-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._analysis_loop, daemon=True, name="PTZTracker"
        )
        self._thread.start()
        logger.info("[PTZ-TRACKER] Analyse-Thread gestartet (1Hz)")

    def stop(self):
        """Thread stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def record_position(self, pan: float, tilt: float):
        """Neue PTZ-Position aufzeichnen (wird vom Service aufgerufen)."""
        with self._lock:
            self._positions.append((time.time(), pan, tilt))

    def set_stage(self, stage: str):
        """Tracker-Stage setzen (idle/locked/searching)."""
        self._stage = stage

    def get_state(self) -> Dict:
        """Aktuellen Zustand abfragen."""
        with self._lock:
            return {
                "ptz_velocity": round(self._velocity, 2),
                "ptz_direction_changes": self._direction_changes,
                "ptz_restless_score": round(self._restless_score, 3),
                "ptz_stage": self._stage,
            }

    @property
    def restless_score(self) -> float:
        return self._restless_score

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def stage(self) -> str:
        return self._stage

    def _get_core_integrator(self):
        """CoreIntegrator lazy init."""
        if self._core_integrator is None:
            try:
                from core.core_integrator import get_core_integrator
                self._core_integrator = get_core_integrator()
            except Exception:
                pass
        return self._core_integrator

    def _analysis_loop(self):
        """1Hz Hauptschleife: Positionen analysieren, Score berechnen, Core fuettern."""
        while self._running:
            try:
                self._analyze()
                self._feed_core()
            except Exception as e:
                logger.error(f"[PTZ-TRACKER] Fehler: {e}")
            time.sleep(1.0)

    def _analyze(self):
        """Ring-Buffer analysieren: Velocity + Direction Changes + Score berechnen."""
        with self._lock:
            positions = list(self._positions)

        if len(positions) < 3:
            self._velocity = 0.0
            self._direction_changes = 0
            self._restless_score = 0.0
            return

        # Differenzen berechnen
        velocities = []
        pan_diffs = []

        for i in range(1, len(positions)):
            t0, p0, ti0 = positions[i - 1]
            t1, p1, ti1 = positions[i]
            dt = t1 - t0
            if dt <= 0:
                continue

            dp = abs(p1 - p0) + abs(ti1 - ti0)
            vel = dp / dt
            velocities.append(vel)
            pan_diffs.append(p1 - p0)

        if not velocities:
            self._velocity = 0.0
            self._direction_changes = 0
            self._restless_score = 0.0
            return

        # Durchschnittliche Velocity
        self._velocity = sum(velocities) / len(velocities)

        # Richtungswechsel in Pan-Differenzen zaehlen
        direction_changes = 0
        for i in range(1, len(pan_diffs)):
            # Vorzeichenwechsel: positiv->negativ oder umgekehrt
            if pan_diffs[i - 1] * pan_diffs[i] < 0:
                # Nur zaehlen wenn die Bewegung signifikant ist (>0.5 Grad)
                if abs(pan_diffs[i]) > 0.5 and abs(pan_diffs[i - 1]) > 0.5:
                    direction_changes += 1

        self._direction_changes = direction_changes

        # restless_score berechnen:
        # 50% direction_changes (normalisiert auf 0-10)
        # 50% velocity (normalisiert auf max_velocity)
        dir_score = min(1.0, direction_changes / 10.0)
        vel_score = min(1.0, self._velocity / MAX_VELOCITY_DEG_S)
        self._restless_score = max(0.0, min(1.0,
            dir_score * 0.5 + vel_score * 0.5
        ))

    def _feed_core(self):
        """restless_score an CoreIntegrator weiterleiten."""
        ci = self._get_core_integrator()
        if not ci:
            return

        score = self._restless_score

        # Hoher Score (>0.6) → Tension steigt, Shadow-Gewichtung
        if score > 0.6:
            ci.update_inputs("ptz_tracker", {
                "environmental_stress": min(1.0, score * 0.5),
            })
        # Niedriger Score (<0.2) → Beruhigung
        elif score < 0.2:
            ci.update_inputs("ptz_tracker", {
                "environmental_stress": 0.0,
            })
        else:
            # Mittlerer Bereich — leichter Stress proportional zum Score
            ci.update_inputs("ptz_tracker", {
                "environmental_stress": score * 0.3,
            })


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[PTZTracker] = None
_instance_lock = threading.Lock()


def get_ptz_tracker() -> PTZTracker:
    """Singleton-Zugriff auf den PTZTracker."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PTZTracker()
    return _instance
