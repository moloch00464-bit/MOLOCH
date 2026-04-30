"""
ZoneToLed — Zone-Wechsel triggert LED-Pattern

Subscribed: zone_changed
Mapping Zone -> LED:
  guardian:  solid blue (0,80,255)
  shadow:    pulsing magenta (200,0,150) breath 4s
  berserker: pulsing red (255,30,0) breath 1s
  sleeping:  dim warm white (50,30,15)

Cooldown: min 2s zwischen Hardware-Calls (debounce).
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("expression.zone_to_led")

_ZONE_TO_PATTERN = {
    "guardian": {
        "name": "solid",
        "color": (0, 80, 255),
        "modus": "statisch",
    },
    "shadow": {
        "name": "breath",
        "color": (200, 0, 150),
        "modus": "atmend",
        "speed": 4.0,
    },
    "berserker": {
        "name": "breath",
        "color": (255, 30, 0),
        "modus": "atmend",
        "speed": 1.0,
    },
    "sleeping": {
        "name": "solid",
        "color": (50, 30, 15),
        "modus": "statisch",
    },
}

_DEBOUNCE_S = 2.0


class ZoneToLed:
    """LED-Pattern bei Zone-Wechseln (debounced)."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._bus = None
        self._last_zone: Optional[str] = None
        self._last_apply_ts: float = 0.0
        self._subscribed = False

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            try:
                from core.moloch_event_bus import get_event_bus
                self._bus = get_event_bus()
                self._bus.subscribe("zone_changed", self._on_zone_event, priority=5)
                self._subscribed = True
                self._running = True
                logger.info("ZoneToLed gestartet (subscribed: zone_changed)")
                return True
            except Exception as e:
                logger.warning(f"ZoneToLed start fehlgeschlagen: {e}")
                return False

    def stop(self):
        with self._lock:
            self._running = False
            self._subscribed = False
            logger.info("ZoneToLed gestoppt")

    def _on_zone_event(self, payload):
        try:
            data = payload if isinstance(payload, dict) else {}
            zone = data.get("zone") or data.get("new_zone") or data.get("to")
            if not zone:
                return
            self.on_zone_change(str(zone))
        except Exception as e:
            logger.debug(f"ZoneToLed _on_zone_event Fehler: {e}")

    def on_zone_change(self, zone: str):
        """Externe API: setzt LED-Pattern fuer Zone."""
        zone = (zone or "").lower().strip()
        pattern = _ZONE_TO_PATTERN.get(zone)
        if not pattern:
            logger.debug(f"ZoneToLed: zone '{zone}' nicht gemappt")
            return
        with self._lock:
            now = time.time()
            if (now - self._last_apply_ts) < _DEBOUNCE_S:
                logger.debug(f"ZoneToLed: debounce active, skip {zone}")
                return
            self._last_zone = zone
            self._last_apply_ts = now
        self._apply_pattern(zone, pattern)

    def _apply_pattern(self, zone: str, pattern: dict):
        """Best-effort: rgb_led_controller.set_pattern() oder set_color()."""
        try:
            # Bevorzugt zentralen Singleton; fallback rgb_led
            led = None
            try:
                from core.led_controller import get_led_controller
                led = get_led_controller()
            except Exception:
                pass
            if led is None:
                from core.hardware.rgb_led_controller import get_rgb_led
                led = get_rgb_led()
            # API-Versuche in Reihenfolge
            if hasattr(led, "set_pattern"):
                led.set_pattern(pattern.get("name", "solid"), pattern)
                logger.info(f"ZoneToLed: set_pattern({pattern.get('name')}) fuer zone={zone}")
                return
            if hasattr(led, "set_color"):
                color = pattern.get("color", (0, 0, 0))
                modus = pattern.get("modus", "statisch")
                # rgb_led_controller.set_color erwartet str-farbe — sende als hex/tuple-string
                if isinstance(color, tuple):
                    farbe = f"{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                else:
                    farbe = str(color)
                speed = pattern.get("speed", 1.0)
                # Versuch mit signature (farbe, modus, geschwindigkeit) — wenn das fehlschlaegt: nur (farbe)
                try:
                    led.set_color(farbe, modus, geschwindigkeit=speed)
                except TypeError:
                    led.set_color(farbe, modus)
                logger.info(f"ZoneToLed: set_color({farbe},{modus}) fuer zone={zone}")
                return
            logger.debug(f"ZoneToLed: keine LED-API verfuegbar fuer zone={zone}")
        except Exception as e:
            logger.debug(f"ZoneToLed _apply_pattern: {e}")

    def get_state(self) -> dict:
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "last_zone": self._last_zone,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": time.time() - self._last_apply_ts if self._last_apply_ts else None,
            }


_instance: Optional[ZoneToLed] = None
_instance_lock = threading.Lock()


def get_zone_to_led() -> ZoneToLed:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ZoneToLed()
        return _instance
