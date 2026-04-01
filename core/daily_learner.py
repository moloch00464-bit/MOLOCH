#!/usr/bin/env python3
"""Daily Learner - Lernt Markus in verschiedenen Bedingungen.

Sammelt automatisch Snapshots von Markus bei:
- Verschiedenen Winkeln (Kopfpose)
- Verschiedenen Lichtverhältnissen
- Verschiedenen Entfernungen

Real-Time Learning: Jedes Learner-Foto wird sofort als Embedding
in die Face-DB aufgenommen (in-memory + Disk). Qualitaetscheck
stellt sicher dass nur echte Matches gelernt werden.

Max 1 Snapshot/Minute. Läuft im Hintergrund, Toggle via Panel.
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
DAILY_DIR = Path("/mnt/moloch-data/Teachen")  # Backup-Pfad (NTFS, historisch)
SNAPSHOTS_DIR = Path("~/moloch/media/snapshots").expanduser()

# Qualitaetscheck: Mindest-Similarity zum bestehenden Basis-Embedding
LEARN_MIN_SIMILARITY = 0.4

# Max gelernte Embeddings pro Person (verhindert DB-Bloat)
MAX_LEARNED_PER_PERSON = 50

# Label-Format fuer gelernte Embeddings: "Name#learn_TIMESTAMP"
LEARN_SEPARATOR = '#'


class DailyLearner:
    """Lernt Markus zuverlässig zu erkennen durch tägliches Sampling."""

    def __init__(self):
        self.enabled = False
        self.last_snapshot_time = 0
        self.snapshot_interval = 15  # Sekunden (war 60 - zu langsam fuer Teach)

        # Tracking: Welche Bedingungen wurden schon gesehen
        # Key: (angle_bucket, light_bucket, distance_bucket)
        # Value: timestamp wann zuletzt gesehen
        self.seen_conditions: Dict[Tuple[int, int, int], float] = {}
        self.condition_cooldown = 300  # 5min bevor gleiche Bedingung wieder (war 3600/1h)

        # Face-DB Referenz (wird vom Service injiziert)
        self._face_db: Optional[Dict[str, np.ndarray]] = None
        self._face_db_path: Optional[str] = None
        self._save_lock = threading.Lock()

        # Stats: Gelernte Embeddings seit Start
        self.embeddings_learned = 0
        self.embeddings_rejected = 0

    def set_face_db(self, face_db: Dict[str, np.ndarray], db_path: str):
        """Injiziere Face-DB Referenz vom Service.

        Args:
            face_db: Das Live-Dict {name: embedding} aus dem Service
            db_path: Pfad zur face_embeddings.json fuer Disk-Persistierung
        """
        self._face_db = face_db
        self._face_db_path = db_path
        learned = sum(1 for k in face_db if LEARN_SEPARATOR in k)
        logger.info(f"[DailyLearner] Face-DB verbunden ({len(face_db)} Eintraege, {learned} gelernt)")

    def enable(self):
        """Aktiviere Daily Learning."""
        self.enabled = True
        logger.info("[DailyLearner] AKTIVIERT")

    def disable(self):
        """Deaktiviere Daily Learning."""
        self.enabled = False
        logger.info("[DailyLearner] DEAKTIVIERT")

    def toggle(self):
        """Toggle Daily Learning."""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled

    def _estimate_lighting(self, face_crop: np.ndarray) -> int:
        """Schätze Lichtverhältnis (0=dunkel, 1=normal, 2=hell).

        Nutzt durchschnittliche Helligkeit des Gesichts.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 80:
            return 0  # Dunkel
        elif mean_brightness < 160:
            return 1  # Normal
        else:
            return 2  # Hell

    def _estimate_distance(self, bbox_height: float, frame_height: int) -> int:
        """Schätze Entfernung (0=fern, 1=mittel, 2=nah).

        Nutzt Gesichtsgröße relativ zum Frame.
        """
        ratio = bbox_height / frame_height

        if ratio < 0.15:
            return 0  # Fern (<15% des Frames)
        elif ratio < 0.35:
            return 1  # Mittel (15-35%)
        else:
            return 2  # Nah (>35%)

    def _estimate_angle(self, head_pose: Optional[Dict]) -> int:
        """Schätze Kopfwinkel-Kategorie (0=frontal, 1=leicht seitlich, 2=stark seitlich).

        Nutzt Yaw (Drehung links/rechts) aus Head Pose.
        """
        if not head_pose:
            return 0  # Unbekannt -> Frontal annehmen

        yaw = abs(head_pose.get("yaw", 0))

        if yaw < 15:
            return 0  # Frontal (±15°)
        elif yaw < 40:
            return 1  # Leicht seitlich (15-40°)
        else:
            return 2  # Stark seitlich (>40°)

    def _is_new_condition(self, angle: int, light: int, distance: int) -> bool:
        """Prüfe ob diese Bedingung neu/interessant ist."""
        condition = (angle, light, distance)
        now = time.time()

        # Noch nie gesehen?
        if condition not in self.seen_conditions:
            return True

        # Lange her seit letztem Mal?
        last_seen = self.seen_conditions[condition]
        if now - last_seen > self.condition_cooldown:
            return True

        return False

    def _check_embedding_quality(self, name: str, embedding: np.ndarray) -> float:
        """Pruefe ob Embedding wirklich zu dieser Person gehoert.

        Vergleicht gegen das Basis-Embedding (ohne #learn Suffix).
        Returns: Beste Similarity zum Basis-Embedding, 0.0 wenn keine DB.
        """
        if self._face_db is None:
            return 0.0

        # Basis-Embedding fuer diese Person finden (ohne #learn Suffix)
        base_emb = self._face_db.get(name)
        if base_emb is None:
            return 0.0

        # Cosine Similarity
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

    def _learn_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        angle: int,
        light: int,
        distance: int,
        confidence: float
    ) -> bool:
        """Embedding in Face-DB aufnehmen (in-memory + Disk).

        Returns: True wenn gelernt, False wenn abgelehnt.
        """
        if self._face_db is None:
            return False

        # Qualitaetscheck: Ist es wirklich diese Person?
        sim = self._check_embedding_quality(name, embedding)
        if sim < LEARN_MIN_SIMILARITY:
            self.embeddings_rejected += 1
            logger.info(f"[LEARNER] Embedding ABGELEHNT fuer {name} "
                       f"(sim={sim:.3f} < {LEARN_MIN_SIMILARITY}, unsicher)")
            return False

        # Limit pruefen
        learned_count = self._count_learned_for_person(name)
        if learned_count >= MAX_LEARNED_PER_PERSON:
            logger.debug(f"[LEARNER] Max Embeddings erreicht fuer {name} ({MAX_LEARNED_PER_PERSON})")
            return False

        # Key generieren: "Markus#learn_1740000000"
        learn_key = f"{name}{LEARN_SEPARATOR}learn_{int(time.time())}"

        # Sicherstellen dass embedding normalisiert ist
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)

        # In-Memory: Dict-Update (thread-safe in CPython fuer einzelne Zuweisung)
        self._face_db[learn_key] = emb_norm.astype(np.float32)

        # Disk-Persistierung in separatem Thread (blockiert nicht die Inference Loop)
        threading.Thread(
            target=self._save_face_db_to_disk,
            daemon=True
        ).start()

        self.embeddings_learned += 1
        angle_names = {0: "frontal", 1: "seitlich", 2: "stark_seitlich"}
        light_names = {0: "dunkel", 1: "normal", 2: "hell"}
        dist_names = {0: "fern", 1: "mittel", 2: "nah"}

        logger.info(f"[LEARNER] Neues Embedding fuer {name} gelernt! "
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
                # Dict -> JSON-serialisierbar
                data = {}
                for name, emb in self._face_db.items():
                    if isinstance(emb, np.ndarray):
                        data[name] = emb.tolist()
                    else:
                        data[name] = list(emb)

                # Atomic write: erst temp, dann rename
                tmp_path = self._face_db_path + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                os.replace(tmp_path, self._face_db_path)

                logger.debug(f"[LEARNER] Face-DB gespeichert ({len(data)} Eintraege)")
            except Exception as e:
                logger.error(f"[LEARNER] Face-DB Speichern fehlgeschlagen: {e}")

    def _save_snapshot(
        self,
        face_crop: np.ndarray,
        name: str,
        confidence: float,
        angle: int,
        light: int,
        distance: int,
        head_pose: Optional[Dict] = None,
        full_frame: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None
    ):
        """Speichere Snapshot mit Metadaten + lerne Embedding.

        Speichert Face-Crop (50% Margin, Q95) und optional den vollen 1080p Frame.
        Wenn ein Embedding mitgegeben wird, wird es nach Qualitaetscheck in die
        Face-DB aufgenommen.
        """
        learned = False
        try:
            # Verzeichnis: ~/moloch/media/snapshots/PERSON/ (Galerie-kompatibel)
            person_dir = SNAPSHOTS_DIR / name
            person_dir.mkdir(parents=True, exist_ok=True)

            # Dateiname: YYYY-MM-DD_HH-MM-SS_conf_angle_light_dist.jpg
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_c{int(confidence*100)}_a{angle}_l{light}_d{distance}.jpg"
            filepath = person_dir / filename

            # Face-Crop via Real-ESRGAN x2 hochskalieren (NPU)
            try:
                from core.perception.super_res_worker import get_super_res
                face_save = get_super_res().upscale(face_crop)
            except Exception:
                face_save = face_crop
            cv2.imwrite(str(filepath), cv2.cvtColor(face_save, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Speichere Full-Frame als Referenzbild
            if full_frame is not None:
                full_filename = f"{timestamp}_c{int(confidence*100)}_a{angle}_l{light}_d{distance}_full.jpg"
                full_filepath = person_dir / full_filename
                cv2.imwrite(str(full_filepath), cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Real-Time Learning: Embedding in Face-DB aufnehmen
            if embedding is not None:
                learned = self._learn_embedding(name, embedding, angle, light, distance, confidence)

            # Speichere Metadaten als JSON
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
                "embedding_learned": learned
            }
            meta_path = filepath.with_suffix(".json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"[DailyLearner] Snapshot: {filename}" +
                       (" + full_frame" if full_frame is not None else "") +
                       (" + LEARNED" if learned else ""))

        except Exception as e:
            logger.error(f"[DailyLearner] Save failed: {e}")

    def maybe_snapshot(
        self,
        face_crop: np.ndarray,
        name: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        frame_height: int,
        head_pose: Optional[Dict] = None,
        full_frame: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """Prüfe ob Snapshot sinnvoll ist und speichere ggf.

        Args:
            face_crop: Face-Crop (BGR, 50% Margin)
            name: ArcFace Match-Name ("Markus" oder "Unbekannt")
            confidence: ArcFace Confidence
            bbox: (x1, y1, x2, y2) in Pixel
            frame_height: Frame-Höhe in Pixel
            head_pose: Optional Head Pose Dict mit yaw/pitch/roll
            full_frame: Optional voller 1080p Frame als Referenzbild
            embedding: Optional 512-dim ArcFace Embedding (fuer Real-Time Learning)

        Returns:
            True wenn Snapshot gespeichert wurde, sonst False
        """
        if not self.enabled:
            return False

        # Rate Limit: 1 Snapshot/Minute
        now = time.time()
        if now - self.last_snapshot_time < self.snapshot_interval:
            return False

        # Bedingungen schätzen
        light = self._estimate_lighting(face_crop)
        bbox_height = bbox[3] - bbox[1]
        distance = self._estimate_distance(bbox_height, frame_height)
        angle = self._estimate_angle(head_pose)

        # Teach-Modus: Markus ist alleine — JEDE Erkennung = Markus
        # Unbekannt/unknown_maybe/Keine DB werden zu "markus" umbenannt
        _SKIP_NAMES = {"unknown_maybe", "Unbekannt", "Keine DB"}
        if name in _SKIP_NAMES:
            # Teach aktiv: Jede Erkennung ist Markus (er ist alleine)
            logger.info(f"[DailyLearner] '{name}' -> 'markus' (Teach-Modus, alleine)")
            name = "markus"
            confidence = max(confidence, 0.60)  # Mindest-Confidence fuer Teach

        # Confidence-Filter: Nur sehr schlechte Erkennungen ablehnen
        if confidence <= 0.30:
            return False

        save_it = False
        reason = ""

        if self._is_new_condition(angle, light, distance):
            save_it = True
            reason = f"neue Bedingung fuer {name}"
        elif confidence >= 0.50:
            # Auch bei bekannter Bedingung: periodisch Snapshots fuer Vielfalt
            # (Rate Limit oben greift trotzdem - max 1 pro snapshot_interval)
            save_it = True
            reason = f"periodischer Teach-Snapshot fuer {name}"

        if save_it:
            self._save_snapshot(face_crop, name, confidence, angle, light, distance, head_pose,
                               full_frame=full_frame, embedding=embedding)
            self.last_snapshot_time = now

            # Markiere Bedingung als gesehen
            condition = (angle, light, distance)
            self.seen_conditions[condition] = now

            logger.info(f"[DailyLearner] Snapshot gespeichert: {reason}, "
                       f"angle={angle}, light={light}, dist={distance}, conf={confidence:.2f}")
            return True

        return False

    def get_stats(self) -> Dict:
        """Stats für Display."""
        snapshots_today = 0
        if SNAPSHOTS_DIR.exists():
            today = time.strftime("%Y-%m-%d")
            # Zähle Dateien von heute in allen Person-Unterordnern
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
            "next_snapshot_in": max(0, int(self.snapshot_interval - (time.time() - self.last_snapshot_time))),
            "embeddings_learned": self.embeddings_learned,
            "embeddings_rejected": self.embeddings_rejected,
            "embeddings_in_db": learned_total
        }


# Singleton
_daily_learner: Optional[DailyLearner] = None


def get_daily_learner() -> DailyLearner:
    """Get/create Singleton."""
    global _daily_learner
    if _daily_learner is None:
        _daily_learner = DailyLearner()
    return _daily_learner
