"""
CamLedToState — Sonoff-Cam-LED als Hardware-Ausdrucks-Layer.

Subscribed: zone_changed, tension_changed
Mapping (Zone -> Cam-LED via eWeLink Cloud):
  guardian:  set_night('day') + set_led(1)   (Tagmodus, schwache LED)
  shadow:    set_night('day') + set_led(0)   (Tagmodus, LED aus)
  berserker: set_night('night') + set_led(3) (Nachtsicht-Farbe, LED max)
  sleeping:  set_night('day') + set_led(0)   (LED aus)

Tension-Override (auch bei gleichem zone): tension >= 0.85 -> set_led(3),
forciert max-Hellig fuer kurzen Akzent.

Cooldown: 5s zwischen Cloud-Calls (eWeLink throtteln bei >1 req/s).

Sub-Agent tentacle (2026-05-03): camera_cloud_bridge.py existiert (1209 LOC,
HMAC-SHA256-Auth + Token-Refresh + LED-APIs). cloud_controller-Singleton
laeuft schon im moloch_service. set_led(level: 0-3) und
set_night(mode: 'auto'|'day'|'night') sind LIVE.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("expression.cam_led_to_state")


# Zone -> (night_mode, led_level)
_ZONE_TO_LIGHT: Dict[str, tuple] = {
    "guardian": ("day", 1),
    "shadow": ("day", 0),
    "berserker": ("night", 3),
    "sleeping": ("day", 0),
}

_DEBOUNCE_S = 5.0
_TENSION_HIGH_THRESHOLD = 0.85   # ueber dieser tension: forciere max-LED


class CamLedToState:
    """Sonoff-Cam-LED-Ausdruck via eWeLink Cloud."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._bus = None
        self._last_zone: Optional[str] = None
        self._last_tension: float = 0.0
        self._last_night: Optional[str] = None
        self._last_led: Optional[int] = None
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
                self._bus.subscribe("tension_changed", self._on_tension_event, priority=5)
                self._subscribed = True
                self._running = True
                logger.info("CamLedToState gestartet (subscribed: zone_changed, tension_changed)")
                return True
            except Exception as e:
                logger.warning(f"CamLedToState start fehlgeschlagen: {e}")
                return False

    def stop(self):
        with self._lock:
            self._running = False
            self._subscribed = False
            logger.info("CamLedToState gestoppt")

    def _extract_payload(self, payload: Any) -> Dict[str, Any]:
        """EventBus uebergibt event_dict (nested 'payload' key) oder direkt dict."""
        data = payload if isinstance(payload, dict) else {}
        inner = data.get("payload", data)
        return inner if isinstance(inner, dict) else {}

    def _on_zone_event(self, payload):
        try:
            inner = self._extract_payload(payload)
            zone = inner.get("zone") or inner.get("new_zone") or inner.get("to")
            if not zone:
                return
            self.on_zone(str(zone))
        except Exception as e:
            logger.debug(f"CamLedToState _on_zone_event Fehler: {e}")

    def _on_tension_event(self, payload):
        try:
            inner = self._extract_payload(payload)
            value = inner.get("value", inner.get("tension"))
            if value is None:
                return
            self.on_tension(float(value))
        except Exception as e:
            logger.debug(f"CamLedToState _on_tension_event Fehler: {e}")

    def on_zone(self, zone: str) -> None:
        """Externe API: setzt LED-Pattern fuer Zone."""
        zone = (zone or "").lower().strip()
        cfg = _ZONE_TO_LIGHT.get(zone)
        if cfg is None:
            logger.debug(f"CamLedToState: zone '{zone}' nicht gemappt")
            return
        night_mode, led_level = cfg
        with self._lock:
            self._last_zone = zone
        self._apply(night_mode, led_level, source=f"zone={zone}")

    def on_tension(self, tension: float) -> None:
        """Externe API: tension >= TENSION_HIGH -> max-LED-Akzent."""
        with self._lock:
            self._last_tension = tension
        if tension >= _TENSION_HIGH_THRESHOLD:
            # Forciere Akzent-LED (kurzfristig, debounced)
            self._apply("night", 3, source=f"tension={tension:.2f}")

    def _apply(self, night_mode: str, led_level: int, source: str = "") -> None:
        """Cloud-Call mit Debounce. Best-effort, blockiert nie."""
        with self._lock:
            now = time.time()
            if (now - self._last_apply_ts) < _DEBOUNCE_S:
                logger.debug(f"CamLedToState: debounce ({source}, skip)")
                return
            # Skip wenn nichts geaendert hat
            if night_mode == self._last_night and led_level == self._last_led:
                return
            self._last_apply_ts = now
            self._last_night = night_mode
            self._last_led = led_level

        # Cloud-Call asynchron via cloud_controller
        def _cloud_dispatch():
            try:
                from core.cloud_controller import get_cloud_controller
                cc = get_cloud_controller()
                if cc is None:
                    logger.debug("CamLedToState: cloud_controller nicht verfuegbar")
                    return
                # cloud_controller.run() schedules an seiner asyncio-Loop
                cc.run(cc.bridge.set_night(night_mode))
                cc.run(cc.bridge.set_led(led_level))
                logger.info(
                    f"CamLedToState: night={night_mode} led={led_level} ({source})"
                )
            except Exception as e:
                logger.debug(f"CamLedToState _apply Cloud-Fehler: {e}")

        threading.Thread(target=_cloud_dispatch, daemon=True).start()

    def get_state(self) -> Dict[str, Any]:
        """Status fuer Audit-Layer (expression-Auditor liest das)."""
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "last_zone": self._last_zone,
                "last_tension": round(self._last_tension, 3),
                "last_night": self._last_night,
                "last_led": self._last_led,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": (time.time() - self._last_apply_ts
                                    if self._last_apply_ts else None),
            }


_instance: Optional[CamLedToState] = None
_instance_lock = threading.Lock()


def get_cam_led_to_state() -> CamLedToState:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = CamLedToState()
        return _instance
