#!/usr/bin/env python3
"""
M.O.L.O.C.H. Temporal Perception Memory
=========================================
NPU Memory System: Temporale Wahrnehmung mit Gedaechtnis.

Basierend auf ChatGPT-Konzept, adaptiert fuer Hailo-10H:
- Embeddings leben in CPU-RAM (nicht NPU-RAM)
- NPU liefert nur die Inference
- Kein Re-Compute wenn Entity bereits bekannt

Komponenten:
1. EntityTracker — Persistente Entities mit Stability/Motion/Familiarity
2. AttentionMap — 8x8 Spatial Grid fuer Bewegung/Aktivitaet
3. SmoothedState — Geglaettete Signale fuer Scheduler (kein Flattern)
"""

import time
import threading
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque

_logger = logging.getLogger("TemporalMemory")


# =============================================================================
# Entity Tracker — Persistente Entitaeten
# =============================================================================

@dataclass
class EntityState:
    """Getrackte Entitaet mit temporalem State."""
    entity_id: str                    # z.B. "markus", "unknown_1"
    embedding: Optional[np.ndarray] = None  # Letztes ArcFace-Embedding
    last_seen: float = 0.0            # Timestamp
    first_seen: float = 0.0           # Wann erstmals erkannt
    stability_score: float = 0.0      # 0-1: Wie konsistent erkannt
    motion_level: float = 0.0         # 0-1: BBox-Bewegung
    familiarity: float = 0.0          # 0-1: Waechst mit wiederholter Erkennung
    last_bbox: Optional[Tuple] = None # (x1,y1,x2,y2) normalisiert
    detection_count: int = 0          # Gesamt-Erkennungen
    miss_count: int = 0               # Frames seit letzter Erkennung


class EntityTracker:
    """Trackt persistente Entities ueber Zeit."""

    FAMILIARITY_GROW = 0.002    # +0.002 pro Erkennung (~1.0 nach 500 Frames = 25s)
    FAMILIARITY_DECAY = 0.0005  # -0.0005 pro Miss
    STABILITY_WINDOW = 20       # Fenster fuer Stability-Score (Frames)
    MOTION_SMOOTHING = 0.3      # EMA-Faktor fuer Motion
    ENTITY_TIMEOUT_S = 60.0     # Entity vergessen nach 60s ohne Erkennung

    def __init__(self):
        self._entities: Dict[str, EntityState] = {}
        self._detection_history: Dict[str, deque] = {}  # bool-Ring pro Entity

    def update(self, face_id: Optional[str], similarity: float,
               embedding: Optional[np.ndarray], bbox: Optional[Tuple]):
        """Entity-State mit neuester Detection aktualisieren."""
        now = time.time()

        if face_id and similarity > 0:
            if face_id not in self._entities:
                self._entities[face_id] = EntityState(
                    entity_id=face_id, first_seen=now)
                self._detection_history[face_id] = deque(
                    maxlen=self.STABILITY_WINDOW)

            ent = self._entities[face_id]
            ent.last_seen = now
            ent.detection_count += 1
            ent.miss_count = 0

            # Embedding aktualisieren
            if embedding is not None:
                ent.embedding = embedding

            # Motion: BBox-Verschiebung messen
            if bbox and ent.last_bbox:
                dx = abs(bbox[0] - ent.last_bbox[0])
                dy = abs(bbox[1] - ent.last_bbox[1])
                motion = min(1.0, (dx + dy) * 5.0)
                ent.motion_level = (self.MOTION_SMOOTHING * motion +
                                    (1 - self.MOTION_SMOOTHING) * ent.motion_level)
            ent.last_bbox = bbox

            # Familiarity waechst
            ent.familiarity = min(1.0, ent.familiarity + self.FAMILIARITY_GROW)

            # Stability: Erkennung in History eintragen
            self._detection_history[face_id].append(True)
            hist = self._detection_history[face_id]
            ent.stability_score = sum(hist) / max(len(hist), 1)

        # Alle Entities: Decay fuer nicht-erkannte
        for eid, ent in list(self._entities.items()):
            if eid == face_id:
                continue
            ent.miss_count += 1
            ent.familiarity = max(0.0, ent.familiarity - self.FAMILIARITY_DECAY)
            if eid in self._detection_history:
                self._detection_history[eid].append(False)
                hist = self._detection_history[eid]
                ent.stability_score = sum(hist) / max(len(hist), 1)

            # Timeout: Entity komplett vergessen
            if now - ent.last_seen > self.ENTITY_TIMEOUT_S:
                del self._entities[eid]
                self._detection_history.pop(eid, None)

    def get_entity(self, entity_id: str) -> Optional[EntityState]:
        return self._entities.get(entity_id)

    def get_primary_entity(self) -> Optional[EntityState]:
        """Entity mit hoechster Familiarity (= wahrscheinlich der Besitzer)."""
        if not self._entities:
            return None
        return max(self._entities.values(), key=lambda e: e.familiarity)

    def get_status(self) -> Dict:
        return {eid: {"familiarity": round(e.familiarity, 3),
                       "stability": round(e.stability_score, 3),
                       "motion": round(e.motion_level, 3),
                       "seen_ago": round(time.time() - e.last_seen, 1),
                       "detections": e.detection_count}
                for eid, e in self._entities.items()}


# =============================================================================
# Attention Map — 8x8 Spatial Grid
# =============================================================================

class AttentionMap:
    """8x8 Spatial Grid das Aktivitaet/Bewegung im Bild trackt."""

    DECAY_RATE = 0.95  # Pro Tick: Werte langsam abklingen lassen

    def __init__(self, grid_w: int = 8, grid_h: int = 8):
        self._w = grid_w
        self._h = grid_h
        self.activity = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.motion = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.novelty = np.zeros((grid_h, grid_w), dtype=np.float32)
        self._prev_activity = np.zeros((grid_h, grid_w), dtype=np.float32)

    def update(self, bboxes: List[Tuple], labels: List[str]):
        """Attention-Map mit aktuellen Detections aktualisieren.

        Args:
            bboxes: Liste von (x1,y1,x2,y2) normalisiert [0-1]
            labels: Passende Labels zu den BBoxes
        """
        # Decay: alles langsam abklingen
        self.activity *= self.DECAY_RATE
        self.motion *= self.DECAY_RATE
        self.novelty *= self.DECAY_RATE

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            # BBox auf Grid mappen
            gx1 = max(0, int(x1 * self._w))
            gy1 = max(0, int(y1 * self._h))
            gx2 = min(self._w, int(x2 * self._w) + 1)
            gy2 = min(self._h, int(y2 * self._h) + 1)

            # Aktivitaet erhoehen
            weight = 1.0 if label == "person" else 0.5
            self.activity[gy1:gy2, gx1:gx2] = np.minimum(
                1.0, self.activity[gy1:gy2, gx1:gx2] + weight * 0.1)

        # Motion = Aenderung der Aktivitaet
        diff = np.abs(self.activity - self._prev_activity)
        self.motion = np.maximum(self.motion, diff)

        # Novelty: hohe Aktivitaet wo vorher wenig war
        low_prev = self._prev_activity < 0.1
        self.novelty[low_prev] = np.maximum(
            self.novelty[low_prev], self.activity[low_prev])

        self._prev_activity = self.activity.copy()

    def get_hotspot(self) -> Tuple[int, int]:
        """Grid-Zelle mit hoechster Aktivitaet."""
        idx = np.argmax(self.activity)
        return (idx % self._w, idx // self._w)

    def get_total_activity(self) -> float:
        """Gesamtaktivitaet (0-1)."""
        return float(np.mean(self.activity))

    def get_status(self) -> Dict:
        return {
            "hotspot": self.get_hotspot(),
            "total_activity": round(self.get_total_activity(), 3),
            "total_motion": round(float(np.mean(self.motion)), 3),
            "total_novelty": round(float(np.mean(self.novelty)), 3),
        }


# =============================================================================
# Smoothed State — Geglaettete Signale fuer Scheduler
# =============================================================================

class SmoothedState:
    """Glaettet schnell wechselnde Perception-Signale.

    Verhindert Scheduler-Flattern (NAH→IDLE→NAH) durch temporale
    Mehrheitsentscheidung statt Einzel-Frame-Reaktion.
    """

    def __init__(self, window: int = 10):
        self._window = window
        self._person_history: deque = deque(maxlen=window)
        self._face_history: deque = deque(maxlen=window)
        self._height_history: deque = deque(maxlen=window)

    def push(self, person_count: int, face_detected: bool,
             bbox_height_pct: float):
        """Neuen Frame-State in die Historie schieben."""
        self._person_history.append(person_count)
        self._face_history.append(face_detected)
        self._height_history.append(bbox_height_pct)

    @property
    def person_count(self) -> int:
        """Geglaetteter Person-Count (Median)."""
        if not self._person_history:
            return 0
        vals = sorted(self._person_history)
        return vals[len(vals) // 2]

    @property
    def face_detected(self) -> bool:
        """Face erkannt in Mehrheit der letzten Frames?"""
        if not self._face_history:
            return False
        return sum(self._face_history) > len(self._face_history) // 2

    @property
    def bbox_height_pct(self) -> float:
        """Geglaettete BBox-Hoehe (gleitender Durchschnitt)."""
        if not self._height_history:
            return 0.0
        return sum(self._height_history) / len(self._height_history)


# =============================================================================
# PerceptionMemory — Hauptklasse (Singleton)
# =============================================================================

class PerceptionMemory:
    """NPU Memory System — vereint alle temporalen Komponenten.

    Nutzung:
        mem = get_perception_memory()
        mem.tick(detections, pframe)  # Jeden Frame aufrufen
        smoothed = mem.get_smoothed_scheduler_input()
        entity = mem.entity_tracker.get_primary_entity()
    """

    def __init__(self):
        self.entity_tracker = EntityTracker()
        self.attention_map = AttentionMap()
        self.smoothed_state = SmoothedState(window=10)
        self._lock = threading.Lock()
        self._tick_count = 0
        _logger.info("[PerceptionMemory] Initialisiert (Entity+Attention+Smoothing)")

    def tick(self, detections: List[Dict], face_id: Optional[str] = None,
             face_similarity: float = 0.0, face_embedding: Optional[np.ndarray] = None,
             face_bbox: Optional[Tuple] = None,
             person_count: int = 0, face_detected: bool = False,
             bbox_height_pct: float = 0.0):
        """Einmal pro Frame aufrufen — aktualisiert alle Subsysteme.

        Args:
            detections: Liste der Detection-Dicts aus _on_buffer
            face_id: Erkannte Face-ID (oder None)
            face_similarity: ArcFace Similarity
            face_embedding: ArcFace Embedding (512d)
            face_bbox: Face BBox (x1,y1,x2,y2)
            person_count: Anzahl erkannter Personen
            face_detected: SCRFD hat Gesicht gefunden
            bbox_height_pct: Groesste Person-BBox Hoehe
        """
        with self._lock:
            self._tick_count += 1

            # 1. Entity Tracker
            self.entity_tracker.update(
                face_id=face_id,
                similarity=face_similarity,
                embedding=face_embedding,
                bbox=face_bbox,
            )

            # 2. Attention Map
            bboxes = [d["bbox"] for d in detections if "bbox" in d]
            labels = [d["class"] for d in detections if "class" in d]
            self.attention_map.update(bboxes, labels)

            # 3. Smoothed State (fuer Scheduler)
            self.smoothed_state.push(
                person_count=person_count,
                face_detected=face_detected,
                bbox_height_pct=bbox_height_pct,
            )

    def get_smoothed_scheduler_input(self) -> Dict:
        """Geglaettete Werte fuer den ModelScheduler.tick().

        Statt Einzel-Frame-Werte → temporale Mehrheit.
        Verhindert Scheduler-Flattern.
        """
        with self._lock:
            return {
                "person_count": self.smoothed_state.person_count,
                "face_detected": self.smoothed_state.face_detected,
                "bbox_height_pct": self.smoothed_state.bbox_height_pct,
            }

    def get_status(self) -> Dict:
        """Voller Status fuer Panel/Debug."""
        with self._lock:
            return {
                "tick_count": self._tick_count,
                "entities": self.entity_tracker.get_status(),
                "attention": self.attention_map.get_status(),
                "smoothed": {
                    "person_count": self.smoothed_state.person_count,
                    "face_detected": self.smoothed_state.face_detected,
                    "bbox_height_pct": round(self.smoothed_state.bbox_height_pct, 3),
                },
            }


# Singleton
_instance: Optional[PerceptionMemory] = None

def get_perception_memory() -> PerceptionMemory:
    """Singleton-Zugriff auf das PerceptionMemory System."""
    global _instance
    if _instance is None:
        _instance = PerceptionMemory()
    return _instance
