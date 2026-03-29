#!/usr/bin/env python3
"""
M.O.L.O.C.H. Motor Learner — Adaptiver PTZ-Gain aus Bewegungsfeedback
=======================================================================

Beobachtet jeden Tracking-Schritt passiv:
  1. Fehler VOR dem Move (pre_error_x/y normiert, -0.5 bis +0.5)
  2. Befohlenes Delta (pan_delta, tilt_delta in Grad)
  3. Fehler NACH dem Move (post_error_x/y normiert, naechster Cycle)

Daraus wird berechnet:
  - Wie viel hat der Gain tatsaechlich gebracht? (Fehler-Reduktionsrate per EMA)
  - Empfohlener base_pan_gain / base_tilt_gain

NUR Vorschlaege — der Tracker entscheidet selbst ob er sie uebernimmt.
Kein Befehl, kein PTZ-Zugriff, kein Arbiter-Eingriff. Christian-Prinzip.

Persistenz: config/motor_learning.json (SSD1, ueberlebt Reboot)
Eingebunden in autonomous_tracker.py (aktiv seit 2026-03-29).
"""

import json
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger("MotorLearner")

_CONFIG_PATH = __import__("os").path.expanduser("~/moloch/config/motor_learning.json")

# Sichere Grenzen — nie ausserhalb dieser Werte
_PAN_GAIN_MIN  = 0.35
_PAN_GAIN_MAX  = 1.00
_TILT_GAIN_MIN = 0.25
_TILT_GAIN_MAX = 0.85

# Standard-Werte (identisch mit TrackingConfig-Defaults)
_PAN_GAIN_DEFAULT  = 0.65
_TILT_GAIN_DEFAULT = 0.50

# Lernrate: langsam lernen, nicht springen
_EMA_ALPHA = 0.05                   # 5% Gewicht pro Observation
_MIN_OBSERVATIONS = 20              # Erst nach N Beobachtungen Gain aendern
_MAX_GAIN_DELTA_PER_UPDATE = 0.03   # Max Aenderung pro Update-Runde
_SAVE_INTERVAL_CYCLES = 100         # Alle N Cycles auf Disk schreiben


class MotorLearner:
    """
    Lernt aus Kamera-Bewegungsfehlern den optimalen Basis-Gain.

    Thread-safe. Singleton via get_motor_learner().
    """

    def __init__(self):
        self._lock = threading.Lock()

        self._base_pan_gain  = _PAN_GAIN_DEFAULT
        self._base_tilt_gain = _TILT_GAIN_DEFAULT

        # EMA der Fehler-Reduktionsrate: 1.0 = ideal, >1 = Ueberschuss, <1 = zu wenig
        self._ema_pan  = 1.0
        self._ema_tilt = 1.0

        self._obs_count      = 0  # Zaehlt pan UND tilt Beobachtungen
        self._cycles_since_save = 0

        self._load()
        logger.info(
            f"[MOTOR-LEARNER] Init: pan_gain={self._base_pan_gain:.3f} "
            f"tilt_gain={self._base_tilt_gain:.3f}"
        )

    # =========================================================================
    # Oeffentliche API
    # =========================================================================

    def record_step(self,
                    pre_error_x: float, pre_error_y: float,
                    pan_delta: float,   tilt_delta: float,
                    post_error_x: float, post_error_y: float):
        """
        Einen Tracking-Step beobachten.

        Aufruf vom Tracker NACH AbsoluteMove, wenn post_error bekannt ist.
        pre/post_error normiert (-0.5 bis +0.5). Deltas in Grad.
        """
        with self._lock:
            updated = False

            if abs(pre_error_x) > 0.03 and abs(pan_delta) > 0.5:
                self._ema_pan = self._update_ema(
                    self._ema_pan, abs(pre_error_x), abs(post_error_x)
                )
                updated = True

            if abs(pre_error_y) > 0.03 and abs(tilt_delta) > 0.5:
                self._ema_tilt = self._update_ema(
                    self._ema_tilt, abs(pre_error_y), abs(post_error_y)
                )
                updated = True

            if updated:
                self._obs_count += 1

            if self._obs_count >= _MIN_OBSERVATIONS:
                self._update_gains()
                self._obs_count = 0

            self._cycles_since_save += 1
            if self._cycles_since_save >= _SAVE_INTERVAL_CYCLES:
                self._save()
                self._cycles_since_save = 0

    def get_base_pan_gain(self) -> float:
        """Empfohlener base_pan_gain. Safe-clamped."""
        with self._lock:
            return self._base_pan_gain

    def get_base_tilt_gain(self) -> float:
        """Empfohlener base_tilt_gain. Safe-clamped."""
        with self._lock:
            return self._base_tilt_gain

    def get_status(self) -> dict:
        """Debug-Status fuer GUI/Logs."""
        with self._lock:
            return {
                "base_pan_gain":  round(self._base_pan_gain,  3),
                "base_tilt_gain": round(self._base_tilt_gain, 3),
                "ema_pan":        round(self._ema_pan,  3),
                "ema_tilt":       round(self._ema_tilt, 3),
            }

    # =========================================================================
    # Interne Logik
    # =========================================================================

    @staticmethod
    def _update_ema(ema: float, pre_mag: float, post_mag: float) -> float:
        """EMA der Fehler-Reduktionsrate aktualisieren.

        Gibt neuen EMA-Wert zurueck. expected_reduction = 70% des Ausgangsfehlers.
        ratio > 1: Kamera hat zu viel korrigiert (Ueberschuss).
        ratio < 1: Kamera hat zu wenig korrigiert (Undershoot).
        """
        expected = pre_mag * 0.7
        if expected <= 0:
            return ema
        ratio = (pre_mag - post_mag) / expected
        return (1.0 - _EMA_ALPHA) * ema + _EMA_ALPHA * ratio

    def _adjust_gain(self, gain: float, ema: float,
                     gain_min: float, gain_max: float) -> float:
        """Gain-Wert aus EMA ableiten. Gedeckelt auf [gain_min, gain_max]."""
        if ema > 1.05:
            # Ueberschuss → senken
            gain -= min(_MAX_GAIN_DELTA_PER_UPDATE, (ema - 1.0) * 0.1)
        elif ema < 0.85:
            # Undershoot → erhoehen
            gain += min(_MAX_GAIN_DELTA_PER_UPDATE, (1.0 - ema) * 0.1)
        return max(gain_min, min(gain_max, gain))

    def _update_gains(self):
        """Gain-Vorschlag aus EMA ableiten. Unter Lock aufrufen."""
        old_pan  = self._base_pan_gain
        old_tilt = self._base_tilt_gain

        self._base_pan_gain  = self._adjust_gain(
            self._base_pan_gain,  self._ema_pan,  _PAN_GAIN_MIN,  _PAN_GAIN_MAX
        )
        self._base_tilt_gain = self._adjust_gain(
            self._base_tilt_gain, self._ema_tilt, _TILT_GAIN_MIN, _TILT_GAIN_MAX
        )

        if (abs(self._base_pan_gain - old_pan) > 0.001 or
                abs(self._base_tilt_gain - old_tilt) > 0.001):
            logger.info(
                f"[MOTOR-LEARNER] Gain aktualisiert: "
                f"pan {old_pan:.3f}→{self._base_pan_gain:.3f}  "
                f"tilt {old_tilt:.3f}→{self._base_tilt_gain:.3f} "
                f"(ema_pan={self._ema_pan:.3f} ema_tilt={self._ema_tilt:.3f})"
            )

    def _save(self):
        """Gelernte Gains auf Disk schreiben."""
        try:
            data = {
                "base_pan_gain":  round(self._base_pan_gain,  4),
                "base_tilt_gain": round(self._base_tilt_gain, 4),
                "ema_pan":        round(self._ema_pan,  4),
                "ema_tilt":       round(self._ema_tilt, 4),
                "ts":             time.time(),
            }
            with open(_CONFIG_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[MOTOR-LEARNER] Speichern fehlgeschlagen: {e}")

    def _load(self):
        """Gespeicherte Gains laden. Faellt auf Defaults zurueck bei Fehler."""
        try:
            with open(_CONFIG_PATH) as f:
                data = json.load(f)
            # Nur laden wenn < 30 Tage alt (Kamera-Hardware kann sich aendern)
            if time.time() - data.get("ts", 0) > 30 * 86400:
                logger.info("[MOTOR-LEARNER] Gespeicherte Gains veraltet (>30 Tage), ignoriert")
                return
            pan  = data.get("base_pan_gain",  _PAN_GAIN_DEFAULT)
            tilt = data.get("base_tilt_gain", _TILT_GAIN_DEFAULT)
            self._base_pan_gain  = max(_PAN_GAIN_MIN,  min(_PAN_GAIN_MAX,  pan))
            self._base_tilt_gain = max(_TILT_GAIN_MIN, min(_TILT_GAIN_MAX, tilt))
            self._ema_pan  = data.get("ema_pan",  1.0)
            self._ema_tilt = data.get("ema_tilt", 1.0)
            logger.info(
                f"[MOTOR-LEARNER] Geladen: pan={self._base_pan_gain:.3f} "
                f"tilt={self._base_tilt_gain:.3f}"
            )
        except FileNotFoundError:
            pass  # Noch kein gespeicherter Zustand — OK
        except Exception as e:
            logger.warning(f"[MOTOR-LEARNER] Laden fehlgeschlagen: {e}")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[MotorLearner] = None
_instance_lock = threading.Lock()


def get_motor_learner() -> MotorLearner:
    """Singleton-Zugriff."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MotorLearner()
    return _instance
