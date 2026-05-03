"""
ZoneToPtz — Zone+Tension triggert PTZ-Ausdrucks-Schwenk.

Subscribed: zone_changed, tension_changed
Mapping (via PTZExpressionLayer.from_personality):
  guardian + tension<0.3   -> calm_center
  shadow   + tension>=0.5  -> scan_left_right
  berserker                -> hectic_jitter
  tension >= 0.7           -> nervous_micro (zonen-unabhaengig)
  tension >= 0.95          -> hectic_jitter

PTZExpressionLayer hat eingebaut:
- Rate-Limit 4/min
- Nacht-Lockout 23-06h
- Skip wenn face_lock_active + low intensity
- Skip wenn busy

Debounce hier zusaetzlich: 12s zwischen Events (verhindert PTZ-Spam bei
schwankender Tension). Cam-Mechanik schonen.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("expression.zone_to_ptz")


_DEBOUNCE_S = 12.0  # min Sekunden zwischen Expression-Triggern


class ZoneToPtz:
    """Subscriber-Modul: ruft PTZExpressionLayer auf zone/tension-Events."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._bus = None
        self._last_zone: Optional[str] = None
        self._last_tension: float = 0.0
        self._last_kind: Optional[str] = None
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
                logger.info(
                    "ZoneToPtz gestartet (subscribed: zone_changed, tension_changed)"
                )
                return True
            except Exception as e:
                logger.warning(f"ZoneToPtz start fehlgeschlagen: {e}")
                return False

    def stop(self):
        with self._lock:
            self._running = False
            self._subscribed = False
            logger.info("ZoneToPtz gestoppt")

    def _extract(self, payload: Any) -> Dict[str, Any]:
        data = payload if isinstance(payload, dict) else {}
        inner = data.get("payload", data)
        return inner if isinstance(inner, dict) else {}

    def _on_zone_event(self, payload):
        try:
            inner = self._extract(payload)
            zone = inner.get("zone") or inner.get("new_zone") or inner.get("to")
            if not zone:
                return
            with self._lock:
                self._last_zone = str(zone).lower().strip()
            self._maybe_express()
        except Exception as e:
            logger.debug(f"ZoneToPtz _on_zone_event Fehler: {e}")

    def _on_tension_event(self, payload):
        try:
            inner = self._extract(payload)
            value = inner.get("value", inner.get("tension"))
            if value is None:
                return
            with self._lock:
                self._last_tension = float(value)
            self._maybe_express()
        except Exception as e:
            logger.debug(f"ZoneToPtz _on_tension_event Fehler: {e}")

    def _maybe_express(self) -> None:
        """Pruefe Debounce + ruefe PTZExpressionLayer.from_personality + express."""
        now = time.time()
        with self._lock:
            if (now - self._last_apply_ts) < _DEBOUNCE_S:
                return
            zone = self._last_zone
            tension = self._last_tension

        try:
            from core.mpo.ptz_expression import get_ptz_expression
            ptz = get_ptz_expression()
        except Exception as e:
            logger.debug(f"ZoneToPtz: ptz_expression nicht verfuegbar: {e}")
            return

        kind = ptz.from_personality(zone, tension)
        if kind is None:
            return

        # Intensity skaliert mit tension (clipped 0.4..1.0 fuer hörbare/sichtbare Wirkung)
        intensity = max(0.4, min(1.0, abs(tension) if tension > 0 else 0.4))
        ok = ptz.express(kind, intensity=intensity, duration_s=3.5)

        if ok:
            with self._lock:
                self._last_apply_ts = now
                self._last_kind = kind
            logger.info(
                f"[ZoneToPtz] zone={zone} tension={tension:.2f} -> express({kind}, "
                f"intensity={intensity:.2f})"
            )

    def get_state(self) -> Dict[str, Any]:
        """Status fuer Audit-Layer (expression-Auditor liest das)."""
        now = time.time()
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "last_zone": self._last_zone,
                "last_tension": round(self._last_tension, 3),
                "last_kind": self._last_kind,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": (now - self._last_apply_ts
                                    if self._last_apply_ts else None),
            }


_instance: Optional[ZoneToPtz] = None
_instance_lock = threading.Lock()


def get_zone_to_ptz() -> ZoneToPtz:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ZoneToPtz()
        return _instance
