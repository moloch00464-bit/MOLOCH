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
# Gate 8: Raeumliche Intelligenz — Objekt-Persistenz
# =========================================================================

class SpatialMemory:
    """Merkt sich welche Objekte in welcher Zone zuletzt gesehen wurden.

    Baut ueber Zeit eine Karte auf: "Stuhl steht bei Schreibtisch",
    "Flasche steht bei Sofa" — auch wenn Kamera woanders hinschaut.
    Persistent auf SSD2.
    """

    OBJECT_TIMEOUT_S = 3600.0  # Objekt nach 1h vergessen
    PERSIST_PATH = "/mnt/moloch-data/memory/spatial_objects.json"

    def __init__(self):
        self._zone_objects: Dict[str, Dict[str, Dict]] = {}
        # zone → {label → {"confidence": float, "last_seen": float, "count": int}}
        self._load()

    def _load(self):
        import json
        try:
            with open(self.PERSIST_PATH, "r") as f:
                self._zone_objects = json.load(f)
            logger.info(f"[SPATIAL] {sum(len(v) for v in self._zone_objects.values())} Objekte geladen")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"[SPATIAL] Laden: {e}")

    def _save(self):
        import json, os
        try:
            os.makedirs(os.path.dirname(self.PERSIST_PATH), exist_ok=True)
            with open(self.PERSIST_PATH, "w") as f:
                json.dump(self._zone_objects, f, indent=2)
        except Exception as e:
            logger.warning(f"[SPATIAL] Speichern: {e}")

    def update(self, zone: Optional[str], detections: List[Dict]):
        """Objekte in der aktuellen Zone aktualisieren.

        Args:
            zone: Aktuelle Raumzone (oder None)
            detections: Liste von {"class": str, "confidence": float}
        """
        if not zone:
            return
        now = time.time()
        if zone not in self._zone_objects:
            self._zone_objects[zone] = {}

        zone_objs = self._zone_objects[zone]
        for det in detections:
            label = det.get("class", "")
            conf = det.get("confidence", 0)
            if label in ("person", "face"):
                continue  # Personen sind nicht ortsgebunden
            if not label or conf < 0.3:
                continue
            if label not in zone_objs:
                zone_objs[label] = {"confidence": conf, "last_seen": now, "count": 1}
            else:
                obj = zone_objs[label]
                obj["confidence"] = max(obj["confidence"], conf)
                obj["last_seen"] = now
                obj["count"] = obj.get("count", 0) + 1

        # Timeout: alte Objekte entfernen
        for z in list(self._zone_objects.keys()):
            for label in list(self._zone_objects[z].keys()):
                if now - self._zone_objects[z][label]["last_seen"] > self.OBJECT_TIMEOUT_S:
                    del self._zone_objects[z][label]

        # Alle 100 Updates speichern
        total = sum(o.get("count", 0) for z in self._zone_objects.values() for o in z.values())
        if total % 100 == 0:
            self._save()

    def get_zone_objects(self, zone: str) -> Dict[str, Dict]:
        """Alle bekannten Objekte in einer Zone."""
        return dict(self._zone_objects.get(zone, {}))

    def get_full_map(self) -> Dict[str, List[str]]:
        """Komplette Raumkarte: Zone → Objektliste."""
        return {zone: list(objs.keys())
                for zone, objs in self._zone_objects.items()
                if objs}

    def query(self, object_label: str) -> Optional[str]:
        """Wo wurde ein bestimmtes Objekt zuletzt gesehen?

        Returns: Zone-Name oder None
        """
        best_zone = None
        best_time = 0
        for zone, objs in self._zone_objects.items():
            if object_label in objs:
                seen = objs[object_label]["last_seen"]
                if seen > best_time:
                    best_time = seen
                    best_zone = zone
        return best_zone

    def get_status(self) -> Dict:
        return {
            "zones_mapped": len(self._zone_objects),
            "total_objects": sum(len(v) for v in self._zone_objects.values()),
            "map": self.get_full_map(),
        }


# =========================================================================
# SINGLETONS
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


_spatial_instance: Optional[SpatialMemory] = None
_spatial_lock = threading.Lock()


def get_spatial_memory() -> SpatialMemory:
    """Singleton-Zugriff auf Spatial Memory."""
    global _spatial_instance
    if _spatial_instance is None:
        with _spatial_lock:
            if _spatial_instance is None:
                _spatial_instance = SpatialMemory()
    return _spatial_instance
