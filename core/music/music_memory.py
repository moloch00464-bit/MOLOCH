#!/usr/bin/env python3
"""
M.O.L.O.C.H. Music Memory — Track-Person-Mood Assoziationen
=============================================================

Speichert welche Tracks bei welcher Person/Mood liefen und schlaegt
basierend auf frueheren Assoziationen passende Tracks vor.

Persistenz: /mnt/moloch-data/memory/music_memory.json (SSD2)
Singleton: get_music_memory()

Gate 2: Identity (ReID + Qdrant VITALE)
"""

import json
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochMusicMemory")

MEMORY_PATH = "/mnt/moloch-data/memory/music_memory.json"


class MusicMemory:
    """Track-Person-Mood Assoziationen persistent speichern und abrufen."""

    def __init__(self):
        self._lock = threading.Lock()
        self._associations: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Assoziationen von Disk laden."""
        if os.path.exists(MEMORY_PATH):
            try:
                with open(MEMORY_PATH, "r") as f:
                    self._associations = json.load(f)
                logger.info(f"[MUSIC-MEM] {len(self._associations)} Assoziationen geladen")
            except Exception as e:
                logger.error(f"[MUSIC-MEM] Laden fehlgeschlagen: {e}")
                self._associations = []
        else:
            logger.info("[MUSIC-MEM] Keine bestehenden Assoziationen, starte leer")

    def _save(self):
        """Assoziationen auf Disk schreiben (muss unter Lock aufgerufen werden)."""
        try:
            os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
            with open(MEMORY_PATH, "w") as f:
                json.dump(self._associations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[MUSIC-MEM] Speichern fehlgeschlagen: {e}")

    def store_association(self, track_id: str, track_name: str,
                          person: str, mood: Optional[str] = None,
                          tension: Optional[float] = None):
        """Track-Person-Mood Assoziation speichern.

        Args:
            track_id: Spotify Track URI/ID
            track_name: Anzeige-Name des Tracks
            person: Erkannte Person (z.B. "markus")
            mood: Aktueller Mood-Cluster (z.B. "guardian", "shadow")
            tension: Aktuelle Tension (0.0 - 1.0)
        """
        entry = {
            "track_id": track_id,
            "track_name": track_name,
            "person": person.lower(),
            "mood": mood,
            "tension": round(tension, 3) if tension is not None else None,
            "timestamp": time.time(),
            "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with self._lock:
            self._associations.append(entry)
            self._save()

        logger.info(f"[MUSIC-MEM] Assoziation: {person} + {track_name} (mood={mood})")

    def suggest_track(self, person: str, mood: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Passendsten Track vorschlagen basierend auf frueheren Assoziationen.

        Scoring: Person-Match + Mood-Match + Recency.
        Gibt den Track mit hoechstem Score zurueck.

        Args:
            person: Person fuer die ein Track gesucht wird
            mood: Optionaler Mood-Filter

        Returns:
            Dict mit track_id, track_name, score oder None
        """
        with self._lock:
            if not self._associations:
                return None

            person_lower = person.lower()
            now = time.time()
            scored: Dict[str, Dict[str, Any]] = {}

            for entry in self._associations:
                # Nur Eintraege dieser Person
                if entry.get("person") != person_lower:
                    continue

                tid = entry["track_id"]
                score = 1.0

                # Mood-Match bonus
                if mood and entry.get("mood") == mood:
                    score += 2.0

                # Recency bonus (hoeher fuer juengere Eintraege, max +1.0)
                age_hours = (now - entry.get("timestamp", 0)) / 3600
                recency = max(0.0, 1.0 - (age_hours / 168))  # 1 Woche Fenster
                score += recency

                # Haeufigkeit: Track mehrfach gehoert = besser
                if tid in scored:
                    scored[tid]["score"] += score
                    scored[tid]["count"] += 1
                else:
                    scored[tid] = {
                        "track_id": tid,
                        "track_name": entry["track_name"],
                        "score": score,
                        "count": 1,
                    }

            if not scored:
                return None

            best = max(scored.values(), key=lambda x: x["score"])
            return best

    def get_stats(self) -> Dict[str, Any]:
        """Statistiken fuer IPC/Panel."""
        with self._lock:
            persons = set(e.get("person") for e in self._associations)
            tracks = set(e.get("track_id") for e in self._associations)
            return {
                "total_associations": len(self._associations),
                "unique_persons": len(persons),
                "unique_tracks": len(tracks),
                "path": MEMORY_PATH,
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[MusicMemory] = None
_instance_lock = threading.Lock()


def get_music_memory() -> MusicMemory:
    """Singleton-Zugriff auf Music Memory."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MusicMemory()
    return _instance
