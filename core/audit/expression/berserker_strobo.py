"""
BerserkerStrobo — Spezialeffekt fuer Mode-Uebergang nach 'berserker'

Subscribed: mode_changed (alt -> 'berserker')
Effekt: 3 Blitze rot/aus ueber 600ms, dann zurueck zu zone_to_led.
Cooldown: max 1x / 30s.
DARF KEIN dauerhaftes Strobo machen — nur 600ms Burst.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("expression.berserker_strobo")

_COOLDOWN_S = 30.0
_FLASH_COUNT = 3
_FLASH_ON_S = 0.1
_FLASH_OFF_S = 0.1


class BerserkerStrobo:
    """3-Blitz-Burst bei Mode-Wechsel zu berserker."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._bus = None
        self._last_apply_ts: float = 0.0
        self._strobo_active = False
        self._subscribed = False

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            try:
                from core.moloch_event_bus import get_event_bus
                self._bus = get_event_bus()
                self._bus.subscribe("mode_changed", self._on_mode_event, priority=5)
                self._subscribed = True
                self._running = True
                logger.info("BerserkerStrobo gestartet (subscribed: mode_changed)")
                return True
            except Exception as e:
                logger.warning(f"BerserkerStrobo start fehlgeschlagen: {e}")
                return False

    def stop(self):
        with self._lock:
            self._running = False
            self._subscribed = False
            logger.info("BerserkerStrobo gestoppt")

    def _on_mode_event(self, payload):
        try:
            data = payload if isinstance(payload, dict) else {}
            old = data.get("old_mode") or data.get("from") or ""
            new = data.get("new_mode") or data.get("to") or data.get("mode") or ""
            new_l = str(new).lower().strip()
            old_l = str(old).lower().strip()
            if new_l == "berserker" and old_l != "berserker":
                self._trigger_strobo()
        except Exception as e:
            logger.debug(f"BerserkerStrobo _on_mode_event Fehler: {e}")

    def _trigger_strobo(self):
        with self._lock:
            now = time.time()
            if (now - self._last_apply_ts) < _COOLDOWN_S:
                logger.debug("BerserkerStrobo: Cooldown aktiv, skip")
                return
            if self._strobo_active:
                return
            self._strobo_active = True
            self._last_apply_ts = now
        # Strobo in eigenem Thread (blockt ~600ms — nicht im EventBus-Callback halten)
        t = threading.Thread(target=self._run_strobo, daemon=True)
        t.start()

    def _run_strobo(self):
        """3 Blitze ueber 600ms — bevorzugt flash_sequence(), sonst manuell."""
        try:
            led = self._get_led()
            if led is None:
                return
            sequence = []
            for _ in range(_FLASH_COUNT):
                sequence.append({"color": (255, 0, 0), "duration": _FLASH_ON_S})
                sequence.append({"color": (0, 0, 0), "duration": _FLASH_OFF_S})
            # Bevorzugt: flash_sequence
            if hasattr(led, "flash_sequence"):
                try:
                    led.flash_sequence(sequence)
                    logger.info("BerserkerStrobo: flash_sequence ausgefuehrt")
                    return
                except Exception as e:
                    logger.debug(f"BerserkerStrobo flash_sequence: {e}")
            # Fallback: manuelle set_color-Schleife
            self._manual_strobo(led)
        finally:
            with self._lock:
                self._strobo_active = False

    def _manual_strobo(self, led):
        """Manueller Strobo via set_color()."""
        try:
            for i in range(_FLASH_COUNT):
                self._led_color(led, "ff0000", "statisch")
                time.sleep(_FLASH_ON_S)
                self._led_color(led, "000000", "statisch")
                time.sleep(_FLASH_OFF_S)
            logger.info(f"BerserkerStrobo: manueller Strobo ({_FLASH_COUNT}x) ausgefuehrt")
        except Exception as e:
            logger.debug(f"BerserkerStrobo _manual_strobo: {e}")

    def _led_color(self, led, farbe: str, modus: str):
        try:
            if hasattr(led, "set_color"):
                try:
                    led.set_color(farbe, modus)
                except TypeError:
                    led.set_color(farbe)
        except Exception:
            pass

    def _get_led(self):
        try:
            from core.led_controller import get_led_controller
            return get_led_controller()
        except Exception:
            pass
        try:
            from core.hardware.rgb_led_controller import get_rgb_led
            return get_rgb_led()
        except Exception:
            return None

    def get_state(self) -> dict:
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "strobo_active": self._strobo_active,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": time.time() - self._last_apply_ts if self._last_apply_ts else None,
            }


_instance: Optional[BerserkerStrobo] = None
_instance_lock = threading.Lock()


def get_berserker_strobo() -> BerserkerStrobo:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = BerserkerStrobo()
        return _instance
