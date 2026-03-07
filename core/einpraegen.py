#!/usr/bin/env python3
"""
M.O.L.O.C.H. Einpraegen — Batch-Analyse fuer Face + Pose Enrollment.

Laeuft als Background-Thread (GUI darf NICHT einfrieren).
Sammelt JPGs aus snapshots/ und daily/, jagt sie durch SCRFD+ArcFace und Pose.

WICHTIG: Nutzt den ModelOrchestrator des Service — kein eigenes VDevice!
Hailo-10H erlaubt nur EIN VDevice gleichzeitig.

Ergebnisse:
  - Face Embeddings → ~/moloch/data/face_embeddings.json (erweitern)
  - Pose-Profile   → ~/moloch/data/pose_profiles.json (neu/erweitern)
"""

import os
import json
import time
import glob
import threading
import logging

import cv2
import numpy as np

from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_pose,
)

logger = logging.getLogger("Einpraegen")


def _letterbox_resize(frame, target_size=640):
    """Letterbox-Resize: Aspect-Ratio erhalten, schwarz auffuellen.

    Gleiche Logik wie TAPPAS hailocropper use-letterbox=true.
    Returns: (letterboxed_frame_rgb, scale, pad_x, pad_y)
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    # Auf target_size x target_size mit Schwarz auffuellen
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR → RGB
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas_rgb, scale, pad_x, pad_y


def _unletterbox_coords(x1n, y1n, x2n, y2n, target_size, scale, pad_x, pad_y, orig_w, orig_h):
    """Normalisierte BBox-Koordinaten aus Letterbox zurueck in Originalkoordinaten (Pixel).

    SCRFD gibt normalisierte Koordinaten bezogen auf 640x640 Letterbox-Bild.
    Diese muessen zurueck auf Originalbild gemappt werden.
    """
    # Normalisiert → Pixel im 640x640 Bild
    px1 = x1n * target_size
    py1 = y1n * target_size
    px2 = x2n * target_size
    py2 = y2n * target_size

    # Padding entfernen
    px1 = (px1 - pad_x) / scale
    py1 = (py1 - pad_y) / scale
    px2 = (px2 - pad_x) / scale
    py2 = (py2 - pad_y) / scale

    # Clamp auf Originalbild
    px1 = max(0, int(px1))
    py1 = max(0, int(py1))
    px2 = min(orig_w, int(px2))
    py2 = min(orig_h, int(py2))

    return px1, py1, px2, py2


# Pfade
SNAPSHOTS_DIR = os.path.expanduser("~/moloch/snapshots")
DAILY_DIR = "/mnt/moloch-data/Teachen"
FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")
POSE_DB_PATH = os.path.expanduser("~/moloch/data/pose_profiles.json")

# Thresholds
MIN_FACE_SIZE = 20       # Pixel, kleiner wird uebersprungen
ARCFACE_MIN_SIM = 0.4    # Mindest-Aehnlichkeit zu bestehendem Markus
SCRFD_CONF = 0.40
SCRFD_NMS = 0.40


def _load_existing_face_db() -> dict:
    """Bestehende Face-Embeddings laden."""
    if not os.path.exists(FACE_DB_PATH):
        return {}
    try:
        with open(FACE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Face-DB laden fehlgeschlagen: {e}")
        return {}


def _save_face_db(db: dict):
    """Face-Embeddings atomar speichern."""
    os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
    tmp = FACE_DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=1, ensure_ascii=False)
    os.replace(tmp, FACE_DB_PATH)


def _load_existing_pose_db() -> dict:
    """Bestehende Pose-Profile laden."""
    if not os.path.exists(POSE_DB_PATH):
        return {}
    try:
        with open(POSE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Pose-DB laden fehlgeschlagen: {e}")
        return {}


def _save_pose_db(db: dict):
    """Pose-Profile atomar speichern."""
    os.makedirs(os.path.dirname(POSE_DB_PATH), exist_ok=True)
    tmp = POSE_DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=1, ensure_ascii=False)
    os.replace(tmp, POSE_DB_PATH)


def _collect_images() -> list:
    """Alle JPGs aus snapshots/ und daily/ sammeln."""
    images = []
    # Snapshots
    if os.path.isdir(SNAPSHOTS_DIR):
        for p in glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg")):
            images.append(p)
    # Daily (alle Unterordner)
    if os.path.isdir(DAILY_DIR):
        for p in glob.glob(os.path.join(DAILY_DIR, "**", "*.jpg"), recursive=True):
            images.append(p)
    return sorted(images)


class Einpraegen:
    """Batch-Analyse: Bilder durch NPU jagen via ModelOrchestrator."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._progress = ""       # z.B. "Face 14/87"
        self._done = False
        self._lock = threading.Lock()
        self._orchestrator = None
        # Statistik
        self._stats = {
            "total": 0,
            "faces_found": 0,
            "faces_saved": 0,
            "faces_skipped_small": 0,
            "faces_skipped_unsicher": 0,
            "poses_saved": 0,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def progress(self) -> str:
        with self._lock:
            return self._progress

    @property
    def is_done(self) -> bool:
        return self._done

    def start(self, orchestrator=None):
        """Einpraegen starten (Background-Thread).

        Args:
            orchestrator: ModelOrchestrator vom Service (shared VDevice).
        """
        if self._running:
            logger.warning("[LERNE] Laeuft bereits!")
            return
        if orchestrator is None:
            logger.error("[LERNE] Kein Orchestrator — Einpraegen nicht moeglich!")
            return
        self._orchestrator = orchestrator
        self._done = False
        self._running = True
        # Statistik zuruecksetzen
        for k in self._stats:
            self._stats[k] = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="Einpraegen")
        self._thread.start()

    def _update_progress(self, phase: str, current: int, total: int):
        """Fortschritt aktualisieren (thread-safe)."""
        with self._lock:
            self._progress = f"{phase} {current}/{total}"

    def _run(self):
        """Haupt-Batch-Schleife. Laeuft im Background-Thread.

        Nutzt ModelOrchestrator.configure()/run() statt eigenes VDevice.
        """
        t_start = time.monotonic()
        models_we_configured = []
        already_active = set()

        try:
            images = _collect_images()
            total = len(images)
            if total == 0:
                logger.info("[LERNE] Gesichter quelle=batch bilder_total=0")
                self._update_progress("", 0, 0)
                return
            self._stats["total"] = total
            logger.info(f"[LERNE] Gesichter quelle=batch bilder_total={total}")

            # Merken welche Modelle schon aktiv waren
            already_active = set(self._orchestrator.active_models)

            # ================================================================
            # PHASE 1: SCRFD + ArcFace (Gesichtserkennung)
            # ================================================================
            logger.info("[LERNE] Phase 1: SCRFD + ArcFace")

            # Modelle konfigurieren falls noetig
            for model_name in ("scrfd", "arcface"):
                if model_name not in already_active:
                    logger.info(f"[LERNE] Konfiguriere {model_name} temporaer")
                    self._orchestrator.configure(model_name)
                    models_we_configured.append(model_name)

            # Pruefen ob Modelle tatsaechlich verfuegbar
            if "scrfd" not in self._orchestrator.active_models:
                logger.error("[LERNE] SCRFD konnte nicht konfiguriert werden!")
                return
            if "arcface" not in self._orchestrator.active_models:
                logger.error("[LERNE] ArcFace konnte nicht konfiguriert werden!")
                return

            face_db = _load_existing_face_db()

            # Referenz-Embedding fuer Markus (Qualitaetscheck)
            markus_ref = None
            if "Markus" in face_db:
                markus_ref = np.array(face_db["Markus"], dtype=np.float32)
                norm = np.linalg.norm(markus_ref)
                if norm > 0:
                    markus_ref = markus_ref / norm

            # Gesammelte Face-Info pro Bild (fuer Phase 2)
            image_faces = {}

            for idx, img_path in enumerate(images):
                if not self._running:
                    break
                self._update_progress("Face", idx + 1, total)

                try:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    fh, fw = frame.shape[:2]

                    # SCRFD: Letterbox 640x640 (gleich wie TAPPAS Pipeline)
                    input_rgb, lb_scale, lb_pad_x, lb_pad_y = _letterbox_resize(frame, 640)
                    scrfd_outputs = self._orchestrator.run("scrfd", input_rgb)
                    if not scrfd_outputs:
                        continue

                    faces = decode_scrfd(scrfd_outputs, 640, SCRFD_CONF, SCRFD_NMS)
                    if not faces:
                        continue

                    image_faces[img_path] = []

                    for (x1n, y1n, x2n, y2n, conf) in faces:
                        # Letterbox-Koordinaten zurueck auf Originalbild
                        x1, y1, x2, y2 = _unletterbox_coords(
                            x1n, y1n, x2n, y2n, 640,
                            lb_scale, lb_pad_x, lb_pad_y, fw, fh)
                        bw, bh = x2 - x1, y2 - y1

                        # Mindestgroesse
                        if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
                            self._stats["faces_skipped_small"] += 1
                            continue

                        self._stats["faces_found"] += 1

                        # 20% Margin
                        mx, my = int(bw * 0.2), int(bh * 0.2)
                        cx1 = max(0, x1 - mx)
                        cy1 = max(0, y1 - my)
                        cx2 = min(fw, x2 + mx)
                        cy2 = min(fh, y2 + my)

                        crop = frame[cy1:cy2, cx1:cx2]
                        crop_112 = cv2.resize(crop, (112, 112))
                        crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

                        # ArcFace Inference via Orchestrator
                        arcface_outputs = self._orchestrator.run("arcface", crop_rgb)
                        if not arcface_outputs:
                            continue
                        emb_key = list(arcface_outputs.keys())[0]
                        embedding = arcface_outputs[emb_key].flatten()
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                        # Qualitaetscheck gegen Markus-Referenz
                        name = "unknown"
                        sim = 0.0
                        if markus_ref is not None:
                            sim = float(np.dot(embedding, markus_ref))
                            if sim >= ARCFACE_MIN_SIM:
                                name = "Markus"
                            else:
                                self._stats["faces_skipped_unsicher"] += 1
                                continue
                        else:
                            # Keine Referenz → als Markus speichern (Ersteinrichtung)
                            name = "Markus"

                        # In DB speichern
                        ts = int(time.time())
                        key = f"{name}#einpraegen_{ts}_{idx}"
                        face_db[key] = embedding.tolist()
                        self._stats["faces_saved"] += 1

                        image_faces[img_path].append({
                            "bbox": [x1, y1, x2, y2],
                            "name": name,
                            "sim": round(sim, 3),
                        })

                except Exception as e:
                    logger.warning(f"[LERNE] Fehler bei {os.path.basename(img_path)}: {e}")
                    continue

            # Face-DB speichern
            _save_face_db(face_db)
            logger.info(f"[LERNE] Phase 1 fertig: gespeichert={self._stats['faces_saved']} "
                        f"zu_klein={self._stats['faces_skipped_small']} "
                        f"unsicher={self._stats['faces_skipped_unsicher']}")

            # ================================================================
            # PHASE 2: Pose (nur Bilder mit erkannten Gesichtern)
            # ================================================================
            face_images = [p for p in images if p in image_faces and image_faces[p]]
            if not face_images:
                logger.info("[LERNE] Keine Bilder mit Gesichtern fuer Pose-Analyse")
            else:
                logger.info(f"[LERNE] Phase 2: Pose fuer {len(face_images)} Bilder")

                if "pose" not in already_active:
                    logger.info("[LERNE] Konfiguriere pose temporaer")
                    self._orchestrator.configure("pose")
                    models_we_configured.append("pose")

                if "pose" not in self._orchestrator.active_models:
                    logger.warning("[LERNE] Pose konnte nicht konfiguriert werden, ueberspringe Phase 2")
                else:
                    pose_db = _load_existing_pose_db()

                    for idx, img_path in enumerate(face_images):
                        if not self._running:
                            break
                        self._update_progress("Pose", idx + 1, len(face_images))

                        try:
                            frame = cv2.imread(img_path)
                            if frame is None:
                                continue
                            fh, fw = frame.shape[:2]

                            # Pose: Letterbox 640x640 (gleich wie TAPPAS Pipeline)
                            input_rgb, _, _, _ = _letterbox_resize(frame, 640)
                            pose_outputs = self._orchestrator.run("pose", input_rgb)
                            if not pose_outputs:
                                continue

                            persons = decode_yolov8_pose(pose_outputs, 640, 640, conf_thresh=0.3)
                            if not persons:
                                continue

                            face_info = image_faces.get(img_path, [])

                            for person in persons:
                                bbox = person.get("bbox", [0, 0, 0, 0])
                                keypoints = person.get("keypoints", [])
                                if not keypoints:
                                    continue

                                # Face mit Person verknuepfen
                                matched_name = "unknown"
                                matched_sim = 0.0
                                px_center = (bbox[0] + bbox[2]) / 2 * fw
                                py_top = bbox[1] * fh

                                for fi in face_info:
                                    fb = fi["bbox"]
                                    fx_center = (fb[0] + fb[2]) / 2
                                    fy_center = (fb[1] + fb[3]) / 2
                                    if (abs(fx_center - px_center) < fw * 0.3
                                            and abs(fy_center - py_top) < fh * 0.3):
                                        matched_name = fi["name"]
                                        matched_sim = fi["sim"]
                                        break

                                if matched_name == "unknown":
                                    continue

                                # Pose-Profil speichern
                                ts = int(time.time())
                                profile = {
                                    "name": matched_name,
                                    "source": os.path.basename(img_path),
                                    "timestamp": ts,
                                    "bbox": [round(b, 4) for b in bbox],
                                    "keypoints": [[round(k, 4) for k in kp]
                                                   for kp in keypoints],
                                    "face_sim": matched_sim,
                                }
                                key = f"{matched_name}#pose_{ts}_{idx}"
                                pose_db[key] = profile
                                self._stats["poses_saved"] += 1

                        except Exception as e:
                            logger.warning(f"[LERNE] Pose-Fehler bei {os.path.basename(img_path)}: {e}")
                            continue

                    _save_pose_db(pose_db)
                    logger.info(f"[LERNE] Phase 2 fertig: poses={self._stats['poses_saved']}")

            # Fertig
            dauer = time.monotonic() - t_start
            logger.info(f"[LERNE] Gesichter quelle=batch bilder_total={total} "
                        f"verarbeitet={self._stats['faces_found']} "
                        f"gespeichert={self._stats['faces_saved']} "
                        f"poses={self._stats['poses_saved']} "
                        f"dauer={dauer:.1f}s")

        except Exception as e:
            logger.error(f"[LERNE] Batch-Analyse fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Temporaer konfigurierte Modelle wieder freigeben
            for name in models_we_configured:
                try:
                    self._orchestrator.unconfigure(name)
                    logger.info(f"[LERNE] {name} wieder freigegeben")
                except Exception:
                    pass
            self._running = False
            self._done = True

    def stop(self):
        """Einpraegen abbrechen."""
        self._running = False


# Singleton
_instance = None
_instance_lock = threading.Lock()


def get_einpraegen() -> Einpraegen:
    """Singleton Einpraegen Instanz."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Einpraegen()
    return _instance
