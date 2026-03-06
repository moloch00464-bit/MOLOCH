#!/usr/bin/env python3
"""
M.O.L.O.C.H. Room Map — PTZ-Winkel zu Raumzonen Mapping
=========================================================

Mappt PTZ Pan-Winkel auf physische Raumzonen (Tuer, Schreibtisch, Sofa).
Publiziert zone_entered Event bei Zonenwechsel.

Zonen-Konfiguration kalibriert fuer Markus' Zimmer:
- Kamera Position: an der Wand, Blickrichtung ins Zimmer
- Pan-Bereiche in Grad (Sonoff CAM-PT2, invertiertes Pan)

Singleton: get_room_map()
Gate 3: Situational Awareness
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochRoomMap")

# Raumzonen: Pan-Winkel-Bereiche (Grad, Sonoff-Koordinaten)
# WICHTIG: Sonoff Pan ist INVERTIERT — positiv = physisch LINKS
DEFAULT_ZONES = {
    "tuer": {"pan_min": -170.0, "pan_max": -120.0, "label": "Tuer"},
    "schreibtisch": {"pan_min": -120.0, "pan_max": -30.0, "label": "Schreibtisch"},
    "mitte": {"pan_min": -30.0, "pan_max": 30.0, "label": "Mitte"},
    "sofa": {"pan_min": 30.0, "pan_max": 120.0, "label": "Sofa"},
    "fenster": {"pan_min": 120.0, "pan_max": 170.0, "label": "Fenster"},
}


class RoomMap:
    """PTZ-Winkel zu Raumzonen Mapping mit Zone-Change Detection."""

    def __init__(self, zones: Optional[Dict[str, Dict]] = None):
        self._zones = zones or DEFAULT_ZONES
        self._current_zone: Optional[str] = None
        self._zone_since: float = 0.0
        self._lock = threading.Lock()

    def update(self, pan_deg: float) -> Optional[str]:
        """PTZ-Pan-Winkel updaten und Zone bestimmen.

        Args:
            pan_deg: Aktueller Pan-Winkel in Grad

        Returns:
            Zone-Name wenn Wechsel, None wenn gleiche Zone
        """
        new_zone = self._pan_to_zone(pan_deg)

        with self._lock:
            if new_zone == self._current_zone:
                return None

            old_zone = self._current_zone
            self._current_zone = new_zone
            self._zone_since = time.time()

        # Event publizieren bei Zonenwechsel
        if new_zone:
            try:
                from core.moloch_event_bus import get_event_bus
                get_event_bus().publish(
                    event_type="zone_entered",
                    source="room_map",
                    priority=5,
                    payload={
                        "zone": new_zone,
                        "label": self._zones.get(new_zone, {}).get("label", new_zone),
                        "previous_zone": old_zone,
                        "pan_deg": round(pan_deg, 1),
                    },
                )
            except Exception as e:
                logger.debug(f"[ROOM-MAP] Event publish: {e}")

        return new_zone

    def _pan_to_zone(self, pan_deg: float) -> Optional[str]:
        """Pan-Winkel auf Zone mappen."""
        for name, zone in self._zones.items():
            if zone["pan_min"] <= pan_deg < zone["pan_max"]:
                return name
        return None

    @property
    def current_zone(self) -> Optional[str]:
        with self._lock:
            return self._current_zone

    @property
    def zone_duration(self) -> float:
        """Sekunden in aktueller Zone."""
        with self._lock:
            if self._zone_since == 0.0:
                return 0.0
            return time.time() - self._zone_since

    def get_zones(self) -> Dict[str, Dict]:
        """Alle konfigurierten Zonen zurueckgeben."""
        return dict(self._zones)

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "current_zone": self._current_zone,
                "zone_duration": round(self.zone_duration, 1),
                "zones": list(self._zones.keys()),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[RoomMap] = None
_instance_lock = threading.Lock()


def get_room_map() -> RoomMap:
    """Singleton-Zugriff auf Room Map."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = RoomMap()
    return _instance
