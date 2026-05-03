"""
PTZExpressionLayer — Hardware-als-Ausdruck via Sonoff-CAM-PT2 PTZ.

Markus hoert Tension am Noctua, sieht Zone an Cam-LED + RGB-LED — und nun
auch BEWEGUNG: die Kamera schwenkt sichtbar je nach emotionalem Zustand.

Sub-Agent-Recherche tracking 2026-05-03:
- ptz_arbiter ist 2-Mode-Gate (MOLOCH_AUTONOM | MOLOCH_MANUELL), kein expression-mode.
- Saubere Loesung: SonoffCameraController.acquire_exclusive("expression") +
  pattern-Loop + release_exclusive. Tracker sieht via _exclusive_owner +
  ueberspringt sauber. Kein Race.
- NEVER 2 bestaetigt: pan_deg positiv = physisch LINKS auf Sonoff.

Patterns:
  nervous_micro    : 2x kleine Pans (+/-3 grad) bei tension>=0.7
  scan_left_right  : 1x weiter Schwenk (+/-15 grad) bei shadow + tension>0.5
  hectic_jitter    : 4x schnelle Mikro-Pans (+/-5 grad) bei berserker
  calm_center      : zur 0/0-Position bei guardian + tension<0.3
  alert_freeze     : KEIN Move (Sleep/idle) — placeholder

Rate-Limit: max 4 Expressions/Min (Cam-Mechanik schonen).
Nacht-Lockout: 23:00-06:00 keine Expression-Schwenks (Markus schlaeft).
Skip wenn tracker auf Markus locked + intensity<0.7 (face_lock_active).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("ptz_expression")


# Pattern-Definitionen: Liste von (pan_offset, tilt_offset, hold_seconds)
# pan_offset/tilt_offset sind RELATIV zur aktuellen Position (delta-Schwenk).
_PATTERNS: Dict[str, List[Tuple[float, float, float]]] = {
    "nervous_micro": [
        (3.0, 0.0, 0.4),
        (-6.0, 0.0, 0.4),
        (3.0, 0.0, 0.3),
    ],
    "scan_left_right": [
        (15.0, 0.0, 0.6),
        (-30.0, 0.0, 0.8),
        (15.0, 0.0, 0.4),
    ],
    "hectic_jitter": [
        (5.0, 2.0, 0.2),
        (-10.0, -4.0, 0.2),
        (8.0, 3.0, 0.2),
        (-5.0, -1.0, 0.2),
        (2.0, 0.0, 0.2),
    ],
    "calm_center": [
        # absolute Move auf 0/0 (Sonderfall, wird in _execute behandelt)
    ],
    "alert_freeze": [],  # No-op
}

_NIGHT_LOCKOUT_HOUR_START = 23   # 23:00
_NIGHT_LOCKOUT_HOUR_END = 6      # 06:00
_RATE_LIMIT_PER_MINUTE = 4


class PTZExpressionLayer:
    """Bewegung-als-Ausdruck via PTZ. Kollisions-frei mit autonomous_tracker."""

    def __init__(self):
        self._lock = threading.RLock()
        self._last_express_ts: float = 0.0
        self._express_history: List[float] = []  # Timestamps der letzten Calls
        self._last_kind: Optional[str] = None
        self._last_intensity: float = 0.0
        self._express_count_total: int = 0
        self._skipped_count_total: int = 0
        self._busy = False  # sichtbar fuer get_state
        self._worker_thread: Optional[threading.Thread] = None

    def from_personality(self, mood_zone: Optional[str], tension: float) -> Optional[str]:
        """Mappt Zone+Tension auf Pattern-Kind. None = kein Override (Tracker bleibt)."""
        zone = (mood_zone or "").lower().strip()
        if tension >= 0.95:
            return "hectic_jitter"
        if tension >= 0.7:
            return "nervous_micro"
        if zone == "berserker":
            return "hectic_jitter"
        if zone == "shadow" and tension >= 0.5:
            return "scan_left_right"
        if zone == "guardian" and tension < 0.3:
            return "calm_center"
        return None

    def express(self, kind: str, intensity: float = 1.0, duration_s: float = 3.0) -> bool:
        """Fuehrt ein Pattern aus. Returns True wenn ausgefuehrt, False wenn skipped."""
        if kind not in _PATTERNS:
            logger.debug(f"[PTZ-EXPR] kind '{kind}' unbekannt -> skip")
            return False
        if kind == "alert_freeze":
            return False
        if self._busy:
            self._skipped_count_total += 1
            logger.debug(f"[PTZ-EXPR] busy -> skip {kind}")
            return False
        if self._is_night_lockout():
            self._skipped_count_total += 1
            logger.debug("[PTZ-EXPR] Nacht-Lockout 23-06h -> skip")
            return False
        if self._is_rate_limited():
            self._skipped_count_total += 1
            logger.debug("[PTZ-EXPR] Rate-Limit (>4/min) -> skip")
            return False
        if intensity < 0.7 and self._face_lock_active():
            self._skipped_count_total += 1
            logger.debug("[PTZ-EXPR] face_lock_active + low intensity -> skip")
            return False

        with self._lock:
            self._busy = True
            self._last_kind = kind
            self._last_intensity = intensity
            self._last_express_ts = time.time()
            self._express_history.append(self._last_express_ts)
            self._express_count_total += 1

        t = threading.Thread(
            target=self._execute,
            args=(kind, intensity, duration_s),
            daemon=True,
            name=f"PTZExpr-{kind}",
        )
        self._worker_thread = t
        t.start()
        return True

    def get_state(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            recent = [t for t in self._express_history if now - t <= 60.0]
            return {
                "alive": True,
                "busy": self._busy,
                "last_kind": self._last_kind,
                "last_intensity": round(self._last_intensity, 2),
                "last_express_ts": self._last_express_ts,
                "last_action_age": (now - self._last_express_ts
                                    if self._last_express_ts else None),
                "count_total": self._express_count_total,
                "skipped_total": self._skipped_count_total,
                "in_last_minute": len(recent),
                "night_lockout": self._is_night_lockout(),
            }

    def _is_night_lockout(self) -> bool:
        h = datetime.now().hour
        return h >= _NIGHT_LOCKOUT_HOUR_START or h < _NIGHT_LOCKOUT_HOUR_END

    def _is_rate_limited(self) -> bool:
        now = time.time()
        with self._lock:
            self._express_history = [t for t in self._express_history if now - t <= 60.0]
            return len(self._express_history) >= _RATE_LIMIT_PER_MINUTE

    def _face_lock_active(self) -> bool:
        try:
            import json
            with open("/dev/shm/moloch_status.json", "r") as f:
                d = json.load(f)
            return bool(d.get("face_lock_active"))
        except Exception:
            return False

    def _execute(self, kind: str, intensity: float, duration_s: float) -> None:
        """Acquired exclusive 'expression', fuehrt Pattern aus, released."""
        try:
            from core.hardware.camera import get_camera_controller
            cam = get_camera_controller(auto_connect=False)
        except Exception as e:
            logger.warning(f"[PTZ-EXPR] camera_controller nicht verfuegbar: {e}")
            self._busy = False
            return
        if cam is None or not cam.is_connected:
            logger.debug("[PTZ-EXPR] cam nicht connected -> abort")
            self._busy = False
            return

        try:
            anchor_pan = float(cam.current_position.pan)
            anchor_tilt = float(cam.current_position.tilt)
        except Exception:
            anchor_pan = 0.0
            anchor_tilt = 0.0

        ok_lock = False
        try:
            ok_lock = cam.acquire_exclusive("expression")
            if not ok_lock:
                logger.debug("[PTZ-EXPR] acquire_exclusive=False -> skip")
                return

            t_start = time.time()
            speed = self._intensity_to_speed(intensity)

            if kind == "calm_center":
                cam.move_absolute(pan_deg=0.0, tilt_deg=0.0, speed=speed)
                time.sleep(min(2.0, duration_s))
                logger.info("[PTZ-EXPR] calm_center -> 0/0")
                return

            steps = _PATTERNS.get(kind, [])
            for (dp, dt, hold) in steps:
                if (time.time() - t_start) > duration_s:
                    break
                target_pan = self._clamp_pan(anchor_pan + dp * intensity)
                target_tilt = self._clamp_tilt(anchor_tilt + dt * intensity)
                try:
                    cam.move_absolute(
                        pan_deg=target_pan, tilt_deg=target_tilt, speed=speed,
                    )
                except Exception as e:
                    logger.debug(f"[PTZ-EXPR] move_absolute Fehler: {e}")
                time.sleep(max(0.1, hold * intensity))

            try:
                cam.move_absolute(
                    pan_deg=anchor_pan, tilt_deg=anchor_tilt, speed=speed,
                )
            except Exception:
                pass
            logger.info(
                f"[PTZ-EXPR] {kind} fertig (intensity={intensity:.2f}, "
                f"steps={len(steps)}, dur={time.time()-t_start:.1f}s)"
            )
        finally:
            if ok_lock:
                try:
                    cam.release_exclusive("expression")
                except Exception as e:
                    logger.debug(f"[PTZ-EXPR] release_exclusive Fehler: {e}")
            self._busy = False

    def _clamp_pan(self, p: float) -> float:
        return max(-168.0, min(170.0, p))

    def _clamp_tilt(self, t: float) -> float:
        return max(-78.0, min(78.0, t))

    def _intensity_to_speed(self, intensity: float) -> float:
        return 0.3 + 0.7 * max(0.0, min(1.0, intensity))


_instance: Optional[PTZExpressionLayer] = None
_instance_lock = threading.Lock()


def get_ptz_expression() -> PTZExpressionLayer:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = PTZExpressionLayer()
        return _instance
