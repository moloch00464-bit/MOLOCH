#!/usr/bin/env python3
"""Spatial Learning - MOLOCH lernt wo Fehldetektionen auftreten.

Konzept:
- Trackt Gesichter die NIE als bekannte Person erkannt werden
- Loggt Position (pan/tilt) + Uhrzeit
- Nach 50 "Unbekannt" von gleicher Position -> Zone als False-Positive markiert
- SCRFD-Score fuer diese Zone automatisch gesenkt

Zones:
- Pan ±10 Grad, Tilt ±10 Grad = eine Zone
- Jede Zone hat counter fuer "unknown" Detektionen
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

HISTORY_FILE = Path.home() / "moloch" / "config" / "perception_history.json"
ZONE_RADIUS = 10.0  # Grad - Zone ist ±10 Grad um Zentrum


class SpatialLearning:
    """Lernt welche Kamera-Positionen False-Positive Detections produzieren."""

    def __init__(self):
        self.history: List[Dict] = []
        self.zone_stats: Dict[Tuple[int, int], int] = {}  # (pan_zone, tilt_zone) -> counter
        self.penalty_zones: List[Tuple[int, int]] = []  # Zones mit >=50 unknowns
        self._load_history()

    def _load_history(self):
        """Lade History + berechne Zone-Stats."""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE) as f:
                    data = json.load(f)
                self.history = data.get("detections", [])
                self.penalty_zones = [tuple(z) for z in data.get("penalty_zones", [])]

                # Zone-Stats aus History berechnen
                for entry in self.history:
                    if entry.get("type") == "unknown_repeated":
                        pan = entry.get("pan", 0)
                        tilt = entry.get("tilt", 0)
                        zone = self._get_zone(pan, tilt)
                        self.zone_stats[zone] = self.zone_stats.get(zone, 0) + 1

                logger.info(f"[SpatialLearning] Loaded {len(self.history)} entries, "
                           f"{len(self.penalty_zones)} penalty zones")
        except Exception as e:
            logger.error(f"[SpatialLearning] Load failed: {e}")

    def _save_history(self):
        """Speichere History + Penalty-Zones."""
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "detections": self.history[-1000:],  # Keep last 1000
                "penalty_zones": self.penalty_zones,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[SpatialLearning] Save failed: {e}")

    def _get_zone(self, pan: float, tilt: float) -> Tuple[int, int]:
        """Berechne Zone-ID aus Position (gerundet auf ZONE_RADIUS)."""
        pan_zone = int(round(pan / ZONE_RADIUS))
        tilt_zone = int(round(tilt / ZONE_RADIUS))
        return (pan_zone, tilt_zone)

    def log_unknown_face(self, pan: float, tilt: float):
        """Logge Unbekannt-Detection an dieser Position."""
        zone = self._get_zone(pan, tilt)
        self.zone_stats[zone] = self.zone_stats.get(zone, 0) + 1

        entry = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pan": round(pan, 1),
            "tilt": round(tilt, 1),
            "zone": zone,
            "type": "unknown_repeated"
        }
        self.history.append(entry)

        # Check ob Zone jetzt Penalty kriegt
        if self.zone_stats[zone] >= 50 and zone not in self.penalty_zones:
            self.penalty_zones.append(zone)
            logger.warning(f"[SpatialLearning] Zone {zone} -> PENALTY (pan={pan:.1f}, tilt={tilt:.1f}, "
                          f"{self.zone_stats[zone]} unknowns)")

        self._save_history()

    def is_penalty_zone(self, pan: float, tilt: float) -> bool:
        """Pruefe ob diese Position in einer Penalty-Zone liegt."""
        zone = self._get_zone(pan, tilt)
        return zone in self.penalty_zones

    def get_penalty_factor(self, pan: float, tilt: float) -> float:
        """Penalty-Faktor fuer SCRFD-Score (0.5 = halbe Score)."""
        if self.is_penalty_zone(pan, tilt):
            return 0.5  # Score halbieren
        return 1.0  # Keine Penalty

    def get_stats(self) -> Dict:
        """Stats fuer Debug/Display."""
        return {
            "total_detections": len(self.history),
            "zones_tracked": len(self.zone_stats),
            "penalty_zones": len(self.penalty_zones),
            "top_zones": sorted(self.zone_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        }


# Singleton
_spatial_learning: Optional[SpatialLearning] = None


def get_spatial_learning() -> SpatialLearning:
    """Get/create Singleton."""
    global _spatial_learning
    if _spatial_learning is None:
        _spatial_learning = SpatialLearning()
    return _spatial_learning
