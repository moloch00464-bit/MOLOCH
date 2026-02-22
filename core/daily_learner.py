#!/usr/bin/env python3
"""Daily Learner - Lernt Markus in verschiedenen Bedingungen.

Sammelt automatisch Snapshots von Markus bei:
- Verschiedenen Winkeln (Kopfpose)
- Verschiedenen Lichtverhältnissen
- Verschiedenen Entfernungen

Max 1 Snapshot/Minute. Läuft im Hintergrund, Toggle via Panel.
"""
import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Storage auf moloch-data SSD (nicht System-SSD)
DAILY_DIR = Path("/mnt/moloch-data/daily")


class DailyLearner:
    """Lernt Markus zuverlässig zu erkennen durch tägliches Sampling."""

    def __init__(self):
        self.enabled = False
        self.last_snapshot_time = 0
        self.snapshot_interval = 60  # Sekunden

        # Tracking: Welche Bedingungen wurden schon gesehen
        # Key: (angle_bucket, light_bucket, distance_bucket)
        # Value: timestamp wann zuletzt gesehen
        self.seen_conditions: Dict[Tuple[int, int, int], float] = {}
        self.condition_cooldown = 3600  # 1h bevor gleiche Bedingung wieder interessant

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

    def _save_snapshot(
        self,
        face_crop: np.ndarray,
        name: str,
        confidence: float,
        angle: int,
        light: int,
        distance: int,
        head_pose: Optional[Dict] = None,
        full_frame: Optional[np.ndarray] = None
    ):
        """Speichere Snapshot mit Metadaten.

        Speichert Face-Crop (50% Margin, Q95) und optional den vollen 1080p Frame.
        """
        try:
            # Verzeichnis: /mnt/moloch-data/daily/YYYY-MM-DD/
            today = time.strftime("%Y-%m-%d")
            day_dir = DAILY_DIR / today
            day_dir.mkdir(parents=True, exist_ok=True)

            # Dateiname: HH-MM-SS_name_conf_angle_light_dist.jpg
            timestamp = time.strftime("%H-%M-%S")
            filename = f"{timestamp}_{name}_c{int(confidence*100)}_a{angle}_l{light}_d{distance}.jpg"
            filepath = day_dir / filename

            # Speichere Face-Crop (JPEG Quality 95)
            cv2.imwrite(str(filepath), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Speichere Full-Frame als Referenzbild
            if full_frame is not None:
                full_filename = f"{timestamp}_{name}_c{int(confidence*100)}_a{angle}_l{light}_d{distance}_full.jpg"
                full_filepath = day_dir / full_filename
                cv2.imwrite(str(full_filepath), full_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

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
                "has_full_frame": full_frame is not None
            }
            meta_path = filepath.with_suffix(".json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"[DailyLearner] Snapshot: {filename}" +
                       (" + full_frame" if full_frame is not None else ""))

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
        full_frame: Optional[np.ndarray] = None
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

        # Nur speichern bei echtem Match (kein unknown_maybe, Unbekannt, Keine DB)
        _SKIP_NAMES = {"unknown_maybe", "Unbekannt", "Keine DB"}
        if name in _SKIP_NAMES or confidence <= 0.5:
            return False

        save_it = False
        reason = ""

        if self._is_new_condition(angle, light, distance):
            save_it = True
            reason = f"neue Bedingung fuer {name}"

        if save_it:
            self._save_snapshot(face_crop, name, confidence, angle, light, distance, head_pose,
                               full_frame=full_frame)
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
        today = time.strftime("%Y-%m-%d")
        day_dir = DAILY_DIR / today

        snapshots_today = 0
        if day_dir.exists():
            snapshots_today = len(list(day_dir.glob("*.jpg")))

        return {
            "enabled": self.enabled,
            "snapshots_today": snapshots_today,
            "conditions_seen": len(self.seen_conditions),
            "next_snapshot_in": max(0, int(self.snapshot_interval - (time.time() - self.last_snapshot_time)))
        }


# Singleton
_daily_learner: Optional[DailyLearner] = None


def get_daily_learner() -> DailyLearner:
    """Get/create Singleton."""
    global _daily_learner
    if _daily_learner is None:
        _daily_learner = DailyLearner()
    return _daily_learner
