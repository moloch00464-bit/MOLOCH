#!/usr/bin/env python3
"""Teachen - Lernt Markus in verschiedenen Bedingungen.

Sammelt automatisch Snapshots von Markus bei:
- Verschiedenen Winkeln (Kopfpose)
- Verschiedenen Lichtverhältnissen
- Verschiedenen Entfernungen

Real-Time Learning: Jedes Teach-Foto wird sofort als Embedding
in die Face-DB aufgenommen (in-memory + Disk). Qualitaetscheck
stellt sicher dass nur echte Matches gelernt werden.

Max 1 Snapshot alle 15 Sekunden. Läuft im Hintergrund, Toggle via Panel.
"""
import os
import time
import json
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Storage: Person-Unterordner unter ~/moloch/media/snapshots/ (Galerie-kompatibel)
TEACH_DIR = Path("/mnt/moloch-data/Teachen")
SNAPSHOTS_DIR = Path("~/moloch/media/snapshots").expanduser()

# Qualitaetscheck: Mindest-Similarity zum bestehenden Basis-Embedding
LEARN_MIN_SIMILARITY = 0.4

# Max gelernte Embeddings pro Person (verhindert DB-Bloat)
MAX_LEARNED_PER_PERSON = 50

# Label-Format fuer gelernte Embeddings: "Name#learn_TIMESTAMP"
LEARN_SEPARATOR = '#'


class Teachen:
    """Lernt Markus zuverlaessig zu erkennen durch Sampling."""

    def __init__(self):
        self.enabled = False
        self.last_snapshot_time = 0
        self.snapshot_interval = 15  # Sekunden

        # Tracking: Welche Bedingungen wurden schon gesehen
        # Key: (angle_bucket, light_bucket, distance_bucket)
        # Value: timestamp wann zuletzt gesehen
        self.seen_conditions: Dict[Tuple[int, int, int], float] = {}
        self.condition_cooldown = 300  # 5min bevor gleiche Bedingung wieder

        # Face-DB Referenz (wird vom Service injiziert)
        self._face_db: Optional[Dict[str, np.ndarray]] = None
        self._face_db_path: Optional[str] = None
        self._save_lock = threading.Lock()

        # Stats: Gelernte Embeddings seit Start
        self.embeddings_learned = 0
        self.embeddings_rejected = 0

    def set_face_db(self, face_db: Dict[str, np.ndarray], db_path: str):
        """Injiziere Face-DB Referenz vom Service."""
        self._face_db = face_db
        self._face_db_path = db_path
        learned = sum(1 for k in face_db if LEARN_SEPARATOR in k)
        logger.info(f"[Teachen] Face-DB verbunden ({len(face_db)} Eintraege, {learned} gelernt)")

    def enable(self):
        """Aktiviere Teachen."""
        self.enabled = True
        logger.info("[Teachen] AKTIVIERT")

    def disable(self):
        """Deaktiviere Teachen."""
        self.enabled = False
        logger.info("[Teachen] DEAKTIVIERT")

    def toggle(self):
        """Toggle Teachen."""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled

    def _estimate_lighting(self, face_crop: np.ndarray) -> int:
        """Schätze Lichtverhältnis (0=dunkel, 1=normal, 2=hell)."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:
            return 0
        elif mean_brightness < 160:
            return 1
        else:
            return 2

    def _estimate_distance(self, bbox_height: float, frame_height: int) -> int:
        """Schätze Entfernung (0=fern, 1=mittel, 2=nah)."""
        ratio = bbox_height / frame_height
        if ratio < 0.15:
            return 0
        elif ratio < 0.35:
            return 1
        else:
            return 2

    def _estimate_angle(self, head_pose: Optional[Dict]) -> int:
        """Schätze Kopfwinkel-Kategorie (0=frontal, 1=leicht seitlich, 2=stark seitlich)."""
        if not head_pose:
            return 0
        yaw = abs(head_pose.get("yaw", 0))
        if yaw < 15:
            return 0
        elif yaw < 40:
            return 1
        else:
            return 2

    def _is_new_condition(self, angle: int, light: int, distance: int) -> bool:
        """Prüfe ob diese Bedingung neu/interessant ist."""
        condition = (angle, light, distance)
        now = time.time()
        if condition not in self.seen_conditions:
            return True
        last_seen = self.seen_conditions[condition]
        if now - last_seen > self.condition_cooldown:
            return True
        return False

    def _check_embedding_quality(self, name: str, embedding: np.ndarray) -> float:
        """Pruefe ob Embedding wirklich zu dieser Person gehoert."""
        if self._face_db is None:
            return 0.0
        base_emb = self._face_db.get(name)
        if base_emb is None:
            return 0.0
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        ref_norm = base_emb / (np.linalg.norm(base_emb) + 1e-10)
        sim = float(np.dot(emb_norm, ref_norm))
        return sim

    def _count_learned_for_person(self, name: str) -> int:
        """Zaehle wie viele gelernte Embeddings fuer diese Person existieren."""
        if self._face_db is None:
            return 0
        prefix = f"{name}{LEARN_SEPARATOR}"
        return sum(1 for k in self._face_db if k.startswith(prefix))

    def _learn_embedding(self, name: str, embedding: np.ndarray,
                         angle: int, light: int, distance: int,
                         confidence: float) -> bool:
        """Embedding in Face-DB aufnehmen (in-memory + Disk)."""
        if self._face_db is None:
            return False
        sim = self._check_embedding_quality(name, embedding)
        if sim < LEARN_MIN_SIMILARITY:
            self.embeddings_rejected += 1
            logger.info(f"[TEACHEN] Embedding ABGELEHNT fuer {name} "
                        f"(sim={sim:.3f} < {LEARN_MIN_SIMILARITY})")
            return False
        learned_count = self._count_learned_for_person(name)
        if learned_count >= MAX_LEARNED_PER_PERSON:
            logger.debug(f"[TEACHEN] Max Embeddings erreicht fuer {name}")
            return False
        learn_key = f"{name}{LEARN_SEPARATOR}learn_{int(time.time())}"
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        self._face_db[learn_key] = emb_norm.astype(np.float32)
        threading.Thread(target=self._save_face_db_to_disk, daemon=True).start()
        self.embeddings_learned += 1
        angle_names = {0: "frontal", 1: "seitlich", 2: "stark_seitlich"}
        light_names = {0: "dunkel", 1: "normal", 2: "hell"}
        dist_names = {0: "fern", 1: "mittel", 2: "nah"}
        logger.info(f"[TEACHEN] Neues Embedding fuer {name} gelernt! "
                    f"(Winkel={angle_names.get(angle, '?')}, "
                    f"Licht={light_names.get(light, '?')}, "
                    f"Distanz={dist_names.get(distance, '?')}, "
                    f"sim={sim:.3f}, conf={confidence:.2f}, "
                    f"gesamt={learned_count + 1}/{MAX_LEARNED_PER_PERSON})")
        return True

    def _save_face_db_to_disk(self):
        """Face-DB als JSON auf Disk schreiben (thread-safe)."""
        if not self._face_db or not self._face_db_path:
            return
        with self._save_lock:
            try:
                data = {}
                for name, emb in self._face_db.items():
                    if isinstance(emb, np.ndarray):
                        data[name] = emb.tolist()
                    else:
                        data[name] = list(emb)
                tmp_path = self._face_db_path + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                os.replace(tmp_path, self._face_db_path)
                logger.debug(f"[TEACHEN] Face-DB gespeichert ({len(data)} Eintraege)")
            except Exception as e:
                logger.error(f"[TEACHEN] Face-DB Speichern fehlgeschlagen: {e}")

    def _save_snapshot(self, face_crop: np.ndarray, name: str, confidence: float,
                       angle: int, light: int, distance: int,
                       head_pose: Optional[Dict] = None,
                       full_frame: Optional[np.ndarray] = None,
                       embedding: Optional[np.ndarray] = None):
        """Speichere Snapshot mit Metadaten + lerne Embedding."""
        learned = False
        try:
            person_dir = SNAPSHOTS_DIR / name
            person_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_c{int(confidence*100)}_a{angle}_l{light}_d{distance}.jpg"
            filepath = person_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if full_frame is not None:
                full_filename = f"{timestamp}_c{int(confidence*100)}_a{angle}_l{light}_d{distance}_full.jpg"
                cv2.imwrite(str(person_dir / full_filename), cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            if embedding is not None:
                learned = self._learn_embedding(name, embedding, angle, light, distance, confidence)
            meta = {
                "timestamp": time.time(),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": name,
                "confidence": round(confidence, 3),
                "angle": angle,
                "lighting": light,
                "distance": distance,
                "head_pose": head_pose,
                "has_full_frame": full_frame is not None,
                "embedding_learned": learned,
            }
            with open(filepath.with_suffix(".json"), 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info(f"[Teachen] Snapshot: {filename}"
                        + (" + full_frame" if full_frame is not None else "")
                        + (" + LEARNED" if learned else ""))
        except Exception as e:
            logger.error(f"[Teachen] Save failed: {e}")

    def maybe_snapshot(self, face_crop: np.ndarray, name: str, confidence: float,
                       bbox: Tuple[float, float, float, float], frame_height: int,
                       head_pose: Optional[Dict] = None,
                       full_frame: Optional[np.ndarray] = None,
                       embedding: Optional[np.ndarray] = None) -> bool:
        """Prüfe ob Snapshot sinnvoll ist und speichere ggf. Gibt True zurück wenn gespeichert."""
        if not self.enabled:
            return False
        now = time.time()
        if now - self.last_snapshot_time < self.snapshot_interval:
            return False
        light = self._estimate_lighting(face_crop)
        bbox_height = bbox[3] - bbox[1]
        distance = self._estimate_distance(bbox_height, frame_height)
        angle = self._estimate_angle(head_pose)

        # Teach-Modus: Markus ist alleine — jede Erkennung = Markus
        _SKIP_NAMES = {"unknown_maybe", "Unbekannt", "Keine DB"}
        if name in _SKIP_NAMES:
            logger.info(f"[Teachen] '{name}' -> 'markus' (Teach-Modus, alleine)")
            name = "markus"
            confidence = max(confidence, 0.60)

        if confidence <= 0.30:
            return False

        save_it = False
        reason = ""
        if self._is_new_condition(angle, light, distance):
            save_it = True
            reason = f"neue Bedingung fuer {name}"
        elif confidence >= 0.50:
            save_it = True
            reason = f"periodischer Teach-Snapshot fuer {name}"

        if save_it:
            self._save_snapshot(face_crop, name, confidence, angle, light, distance,
                                head_pose, full_frame=full_frame, embedding=embedding)
            self.last_snapshot_time = now
            self.seen_conditions[(angle, light, distance)] = now
            logger.info(f"[Teachen] Snapshot gespeichert: {reason}, "
                        f"angle={angle}, light={light}, dist={distance}, conf={confidence:.2f}")
            return True
        return False

    def maybe_snapshot_tappas(self, frame: np.ndarray,
                              face_bbox_px: Tuple[float, float, float, float],
                              name: str, confidence: float) -> bool:
        """Snapshot aus TAPPAS-Frame (kein separater face_crop noetig).

        Wird im TAPPAS-Perception-Loop aufgerufen (PFrame-Daten).
        Extrahiert face_crop automatisch aus frame + bbox.
        Gibt True zurueck wenn Snapshot gespeichert wurde.
        """
        if not self.enabled:
            return False
        if frame is None or face_bbox_px is None:
            return False
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox_px]
        # 50% Margin wie in inference_engine
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * 0.5), int(bh * 0.5)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(fw, x2 + mx)
        cy2 = min(fh, y2 + my)
        face_crop = frame[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            return False
        return self.maybe_snapshot(
            face_crop=face_crop,
            name=name or "Unbekannt",
            confidence=confidence,
            bbox=(float(cx1), float(cy1), float(cx2), float(cy2)),
            frame_height=fh,
            full_frame=frame,
        )

    def get_stats(self) -> Dict:
        """Stats für Display."""
        snapshots_today = 0
        if SNAPSHOTS_DIR.exists():
            today = time.strftime("%Y-%m-%d")
            for p in SNAPSHOTS_DIR.iterdir():
                if p.is_dir():
                    snapshots_today += len(list(p.glob(f"{today}*.jpg")))
        learned_total = 0
        if self._face_db:
            learned_total = sum(1 for k in self._face_db if LEARN_SEPARATOR in k)
        return {
            "enabled": self.enabled,
            "snapshots_today": snapshots_today,
            "conditions_seen": len(self.seen_conditions),
            "next_snapshot_in": max(0, int(self.snapshot_interval - (
                time.time() - self.last_snapshot_time))),
            "embeddings_learned": self.embeddings_learned,
            "embeddings_rejected": self.embeddings_rejected,
            "embeddings_in_db": learned_total,
        }


# Singleton
_teachen: Optional[Teachen] = None


def get_teachen() -> Teachen:
    """Get/create Singleton."""
    global _teachen
    if _teachen is None:
        _teachen = Teachen()
    return _teachen


# Rueckwaertskompatibilitaet fuer alten Code der noch daily_learner importiert
DailyLearner = Teachen
get_daily_learner = get_teachen
