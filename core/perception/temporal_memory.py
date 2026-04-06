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

    Window=30 (~1.5s bei 20 FPS) — genuegend Traegheit gegen Ausreisser.
    BBox-Hoehe: Median statt Mean, Null-Werte (kein Person-Detection) ignoriert.
    """

    def __init__(self, window: int = 30):
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
        """Geglaettete BBox-Hoehe (Median, Null-Werte ignoriert).

        Frames ohne Person-Detection (height=0.0) werden NICHT gezaehlt,
        weil sie den Durchschnitt verzerren und Szenario-Oszillation verursachen.
        Median ist robuster gegenueber Ausreissern als Mean.
        """
        if not self._height_history:
            return 0.0
        # Nur Frames MIT Person-Detection (height > 0) zaehlen
        valid = [h for h in self._height_history if h > 0.01]
        if not valid:
            return 0.0  # Keine Person in den letzten 30 Frames → 0
        valid.sort()
        return valid[len(valid) // 2]  # Median


# =============================================================================
# Routine Tracker — Tageszeit-Muster lernen (Gate 6)
# =============================================================================

class RoutineTracker:
    """Lernt Tageszeit-Muster und erkennt Anomalien.

    Speichert pro Stunde (0-23) typische Werte:
    - presence_rate: Wie oft ist jemand da (0-1)
    - avg_motion: Durchschnittliche Bewegung (0-1)
    - dominant_face: Haeufigste erkannte Person
    - avg_tension: Durchschnittliche System-Spannung

    Lernt ueber Tage/Wochen und erkennt Abweichungen.
    """

    LEARN_RATE = 0.01   # EMA-Faktor pro Stunden-Update (~100 Tage bis konvergiert)
    ANOMALY_THRESHOLD = 0.4  # Abweichung ab der Anomalie gemeldet wird

    def __init__(self, persist_path: str = "/mnt/moloch-data/memory/routines.json"):
        self._persist_path = persist_path
        # Pro Stunde (0-23): gelerntes Profil
        self._hourly_profile: Dict[int, Dict] = {}
        # Aktuelle Stunden-Akkumulatoren
        self._current_hour: int = -1
        self._hour_samples: int = 0
        self._hour_presence_sum: float = 0.0
        self._hour_motion_sum: float = 0.0
        self._hour_face_counts: Dict[str, int] = {}
        self._load()

    def _load(self):
        """Profil von Disk laden."""
        try:
            import json
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            # Keys sind Strings in JSON → zu int konvertieren
            self._hourly_profile = {int(k): v for k, v in data.items()}
            _logger.info(f"[RoutineTracker] {len(self._hourly_profile)} Stunden-Profile geladen")
        except FileNotFoundError:
            _logger.info("[RoutineTracker] Kein Profil vorhanden — starte leer")
        except Exception as e:
            _logger.warning(f"[RoutineTracker] Laden fehlgeschlagen: {e}")

    def _save(self):
        """Profil auf Disk speichern."""
        try:
            import json, os
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, "w") as f:
                json.dump(self._hourly_profile, f, indent=2)
        except Exception as e:
            _logger.warning(f"[RoutineTracker] Speichern fehlgeschlagen: {e}")

    def update(self, person_detected: bool, motion_level: float,
               face_id: Optional[str]):
        """Pro Frame aufrufen — akkumuliert Daten fuer aktuelle Stunde."""
        import datetime
        hour = datetime.datetime.now().hour

        # Stunden-Wechsel: altes Profil lernen, Akkumulator reset
        if hour != self._current_hour:
            if self._current_hour >= 0 and self._hour_samples > 0:
                self._learn_hour(self._current_hour)
            self._current_hour = hour
            self._hour_samples = 0
            self._hour_presence_sum = 0.0
            self._hour_motion_sum = 0.0
            self._hour_face_counts = {}

        self._hour_samples += 1
        self._hour_presence_sum += 1.0 if person_detected else 0.0
        self._hour_motion_sum += min(1.0, motion_level)
        if face_id:
            self._hour_face_counts[face_id] = self._hour_face_counts.get(face_id, 0) + 1

    def _learn_hour(self, hour: int):
        """EMA-Update des Stunden-Profils."""
        if self._hour_samples == 0:
            return
        presence_rate = self._hour_presence_sum / self._hour_samples
        avg_motion = self._hour_motion_sum / self._hour_samples
        dominant_face = max(self._hour_face_counts, key=self._hour_face_counts.get) \
            if self._hour_face_counts else None

        lr = self.LEARN_RATE
        if hour not in self._hourly_profile:
            # Erster Datenpunkt: direkt uebernehmen
            self._hourly_profile[hour] = {
                "presence_rate": presence_rate,
                "avg_motion": avg_motion,
                "dominant_face": dominant_face,
                "sample_days": 1,
            }
        else:
            p = self._hourly_profile[hour]
            p["presence_rate"] = (1 - lr) * p["presence_rate"] + lr * presence_rate
            p["avg_motion"] = (1 - lr) * p["avg_motion"] + lr * avg_motion
            if dominant_face:
                p["dominant_face"] = dominant_face
            p["sample_days"] = p.get("sample_days", 0) + 1

        self._save()

    def get_expected(self, hour: int = -1) -> Optional[Dict]:
        """Erwartetes Profil fuer eine Stunde (default: jetzt)."""
        if hour < 0:
            import datetime
            hour = datetime.datetime.now().hour
        return self._hourly_profile.get(hour)

    def check_anomaly(self, person_detected: bool, motion_level: float) -> Optional[str]:
        """Pruefen ob aktueller State vom gelernten Profil abweicht.

        Returns: Anomalie-Beschreibung oder None wenn normal.
        """
        import datetime
        hour = datetime.datetime.now().hour
        expected = self._hourly_profile.get(hour)
        if not expected or expected.get("sample_days", 0) < 3:
            return None  # Zu wenig Daten zum Vergleichen

        # Presence-Anomalie: normalerweise da, jetzt weg (oder umgekehrt)
        exp_presence = expected["presence_rate"]
        actual_presence = 1.0 if person_detected else 0.0
        if abs(actual_presence - exp_presence) > self.ANOMALY_THRESHOLD:
            if actual_presence > exp_presence:
                return f"Unerwartet: Person um {hour}:00 (normal: {exp_presence:.0%} Anwesenheit)"
            else:
                return f"Unerwartet: Niemand um {hour}:00 (normal: {exp_presence:.0%} Anwesenheit)"

        # Motion-Anomalie: normalerweise ruhig, jetzt hektisch
        exp_motion = expected["avg_motion"]
        if motion_level > exp_motion + self.ANOMALY_THRESHOLD:
            return f"Ungewoehnlich viel Bewegung um {hour}:00 (normal: {exp_motion:.2f}, jetzt: {motion_level:.2f})"

        return None

    def get_status(self) -> Dict:
        import datetime
        hour = datetime.datetime.now().hour
        expected = self.get_expected(hour)
        return {
            "current_hour": hour,
            "profiled_hours": len(self._hourly_profile),
            "expected": expected,
            "sample_days": expected.get("sample_days", 0) if expected else 0,
        }


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
        self.routine_tracker = RoutineTracker()  # Gate 6: Tageszeit-Muster
        self._lock = threading.Lock()
        self._tick_count = 0
        self._last_anomaly: Optional[str] = None
        self._anomaly_cooldown: float = 0.0  # Nicht jedes Frame melden
        _logger.info("[PerceptionMemory] Initialisiert (Entity+Attention+Smoothing+Routines)")

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

            # 4. Routine Tracker (Gate 6: Tageszeit-Muster lernen)
            primary = self.entity_tracker.get_primary_entity()
            motion = primary.motion_level if primary else 0.0
            self.routine_tracker.update(
                person_detected=person_count > 0,
                motion_level=motion,
                face_id=face_id,
            )

            # Anomalie-Check (max 1x pro 30 Sekunden)
            now = time.time()
            if now - self._anomaly_cooldown > 30.0:
                anomaly = self.routine_tracker.check_anomaly(
                    person_detected=person_count > 0,
                    motion_level=motion,
                )
                if anomaly and anomaly != self._last_anomaly:
                    self._last_anomaly = anomaly
                    self._anomaly_cooldown = now
                    _logger.info(f"[ROUTINE-ANOMALIE] {anomaly}")

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
                "routine": self.routine_tracker.get_status(),
                "last_anomaly": self._last_anomaly,
            }


# Singleton
_instance: Optional[PerceptionMemory] = None

def get_perception_memory() -> PerceptionMemory:
    """Singleton-Zugriff auf das PerceptionMemory System."""
    global _instance
    if _instance is None:
        _instance = PerceptionMemory()
    return _instance
