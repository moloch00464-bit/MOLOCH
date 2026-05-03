"""
TensionToFan — Tension steuert Luefter-PWM (zusaetzlich zu thermal)

Subscribed: tension_changed, mood_changed
Mapping (Tension 0..1 -> PWM 0..100):
  0.0-0.3 calm:       25%
  0.3-0.5 alert:      35%
  0.5-0.7 focused:    50%
  0.7-0.85 high:      75%
  0.85-1.0 berserker: 100%

Thermal-Override: Wenn cpu_temp > 70 deg C, gilt thermal_pwm.
Effektive PWM: max(tension_pwm, thermal_pwm) — niemals unter thermal-Bedarf.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("expression.tension_to_fan")

_TENSION_TO_PWM = [
    (0.30, 25),
    (0.50, 35),
    (0.70, 50),
    (0.85, 75),
    (1.01, 100),
]


def _tension_to_pwm(value: float) -> int:
    """Mappt Tension-Wert auf PWM-Stufe."""
    try:
        v = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 25
    for threshold, pwm in _TENSION_TO_PWM:
        if v < threshold:
            return pwm
    return 100


class TensionToFan:
    """Steuert Luefter-PWM basierend auf Tension-Level."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._last_value: float = 0.0
        self._last_pwm: int = 25
        self._last_apply_ts: float = 0.0
        self._min_apply_interval: float = 2.0  # debounce: 2s zwischen Hardware-Calls
        self._bus = None
        self._subscribed = False

    def start(self) -> bool:
        """Subscribe an EventBus, beginne Tension-Monitoring."""
        with self._lock:
            if self._running:
                return True
            try:
                from core.moloch_event_bus import get_event_bus
                self._bus = get_event_bus()
                self._bus.subscribe("tension_changed", self._on_tension_event, priority=5)
                self._bus.subscribe("mood_changed", self._on_mood_event, priority=5)
                self._subscribed = True
                self._running = True
                logger.info("TensionToFan gestartet (subscribed: tension_changed, mood_changed)")
                return True
            except Exception as e:
                logger.warning(f"TensionToFan start fehlgeschlagen: {e}")
                return False

    def stop(self):
        """Graceful shutdown."""
        with self._lock:
            self._running = False
            self._subscribed = False
            logger.info("TensionToFan gestoppt")

    def _on_tension_event(self, payload):
        """EventBus Callback fuer tension_changed.

        EventBus uebergibt event_dict (timestamp/event_type/source/payload).
        Werte liegen im NESTED 'payload'-Key. Fallback auf toplevel falls
        Publisher payload direkt als top-level dict schickt.
        """
        try:
            data = payload if isinstance(payload, dict) else {}
            inner = data.get("payload", data)  # event_dict.payload oder direkt
            if not isinstance(inner, dict):
                inner = {}
            value = inner.get("value", inner.get("tension",
                              data.get("value", data.get("tension", 0.0))))
            self.on_tension(float(value))
        except Exception as e:
            logger.debug(f"TensionToFan _on_tension_event Fehler: {e}")

    def _on_mood_event(self, payload):
        """EventBus Callback fuer mood_changed — nutzt mood-tension falls verfuegbar."""
        try:
            data = payload if isinstance(payload, dict) else {}
            inner = data.get("payload", data)
            if not isinstance(inner, dict):
                inner = {}
            tension = inner.get("tension", data.get("tension"))
            if tension is not None:
                self.on_tension(float(tension))
        except Exception as e:
            logger.debug(f"TensionToFan _on_mood_event Fehler: {e}")

    def on_tension(self, value: float):
        """Externe API: setzt Luefter-PWM nach Tension-Wert."""
        with self._lock:
            self._last_value = value
            new_pwm = _tension_to_pwm(value)
            now = time.time()
            if new_pwm == self._last_pwm and (now - self._last_apply_ts) < self._min_apply_interval:
                return
            self._last_pwm = new_pwm
            self._last_apply_ts = now
        self._apply_pwm(new_pwm)

    def _apply_pwm(self, tension_pwm: int):
        """Best-effort: ruft thermal_manager.set_tension_pwm() falls API existiert."""
        try:
            from core.hardware.thermal_manager import get_thermal_manager
            tm = get_thermal_manager()
            if hasattr(tm, "set_tension_pwm"):
                tm.set_tension_pwm(int(tension_pwm))
                logger.debug(f"TensionToFan: PWM={tension_pwm}% an thermal_manager")
            else:
                logger.debug(
                    f"TensionToFan: thermal_manager.set_tension_pwm() fehlt — "
                    f"PWM={tension_pwm}% nur registriert (waiting for API)"
                )
        except Exception as e:
            logger.debug(f"TensionToFan _apply_pwm: {e}")

    def get_state(self) -> dict:
        """Status fuer Audit-Layer."""
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "last_value": self._last_value,
                "last_pwm": self._last_pwm,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": time.time() - self._last_apply_ts if self._last_apply_ts else None,
            }


_instance: Optional[TensionToFan] = None
_instance_lock = threading.Lock()


def get_tension_to_fan() -> TensionToFan:
    """Singleton-Getter."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TensionToFan()
        return _instance
