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

Welle DH-4 (Drei-Hirn-Synthese): Seufzer-Spike
==============================================
Bei Tension-Delta >= 0.10 wird ein kurzer Spike ueber den aktuellen PWM-Wert
hinaus ausgeloest (Gemini-Idee + DeepSeek-Refinement: 800ms Dauer, 30s Cooldown).
Der Effekt ist akustisch wahrnehmbar - Markus hoert das Aufheulen als physische
Reaktion auf Tension-Wechsel. Tension wirkt damit als Meta-Parameter (ChatGPT-
Synthese-Position) auf die Lueftermodulation, NICHT als direkter State-Trigger.
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
        # Welle DH-4: Seufzer-Spike (akustische Reaktion auf Tension-Delta)
        self._last_spike_ts: float = 0.0
        self._spike_cooldown_sec: float = 30.0
        self._spike_delta_threshold: float = 0.10
        self._spike_duration_sec: float = 0.8
        self._spike_pwm_boost: int = 25  # Spike addiert max +25% auf aktuellen PWM

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
        """Externe API: setzt Luefter-PWM nach Tension-Wert.

        Welle DH-4: bei Tension-Delta >= threshold wird ein Seufzer-Spike
        ausgeloest (kurzer PWM-Boost ueber den aktuellen Wert hinaus).
        """
        spike_target: Optional[int] = None
        spike_reset_to: Optional[int] = None
        with self._lock:
            try:
                _v = float(value)
            except (TypeError, ValueError):
                _v = 0.0
            delta = abs(_v - self._last_value)
            self._last_value = _v
            new_pwm = _tension_to_pwm(_v)
            now = time.time()

            # Seufzer-Spike Trigger
            if (delta >= self._spike_delta_threshold
                    and (now - self._last_spike_ts) >= self._spike_cooldown_sec):
                spike_target = min(100, new_pwm + self._spike_pwm_boost)
                spike_reset_to = new_pwm
                self._last_spike_ts = now

            # Standard-PWM-Apply mit Debounce
            do_apply = not (new_pwm == self._last_pwm
                            and (now - self._last_apply_ts) < self._min_apply_interval)
            if do_apply:
                self._last_pwm = new_pwm
                self._last_apply_ts = now

        if spike_target is not None:
            self._apply_pwm(spike_target)
            logger.info(f"TensionToFan: Seufzer-Spike PWM={spike_target}% delta={delta:.2f}")
            t = threading.Timer(self._spike_duration_sec, self._apply_pwm, args=(spike_reset_to,))
            t.daemon = True
            t.start()
        elif do_apply:
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
