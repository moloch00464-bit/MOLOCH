#!/usr/bin/env python3
"""
M.O.L.O.C.H. Einpraegen — Batch-Analyse fuer Face + Pose Enrollment.

Laeuft als Background-Thread (GUI darf NICHT einfrieren).
Sammelt JPGs aus snapshots/ und daily/, jagt sie durch SCRFD+ArcFace und Pose.

Ergebnisse:
  - Face Embeddings → ~/moloch/data/face_embeddings.json (erweitern)
  - Pose-Profile   → ~/moloch/data/pose_profiles.json (neu/erweitern)

NPU Max-2 Regel: Erst SCRFD+ArcFace fuer alle Bilder, DANN SCRFD+Pose fuer alle.
"""

import os
import json
import time
import glob
import threading
import logging

import cv2
import numpy as np
from hailo_platform import HEF, VDevice, FormatType

logger = logging.getLogger("Einpraegen")

# Pfade
SNAPSHOTS_DIR = os.path.expanduser("~/moloch/snapshots")
DAILY_DIR = "/mnt/moloch-data/daily"
FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")
POSE_DB_PATH = os.path.expanduser("~/moloch/data/pose_profiles.json")
MODEL_DIR = "/mnt/moloch-data/hailo/models"

# Modell-Pfade
SCRFD_HEF = f"{MODEL_DIR}/scrfd_10g.hef"
ARCFACE_HEF = f"{MODEL_DIR}/arcface_mobilefacenet.hef"
POSE_HEF = f"{MODEL_DIR}/yolov8s_pose_h10.hef"

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
    """Batch-Analyse: Bilder durch NPU jagen, Embeddings + Pose speichern."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._progress = ""       # z.B. "14/87"
        self._done = False
        self._lock = threading.Lock()
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

    def start(self):
        """Einpraegen starten (Background-Thread)."""
        if self._running:
            logger.warning("[EINPRAEGEN] Laeuft bereits!")
            return
        self._done = False
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="Einpraegen")
        self._thread.start()

    def _update_progress(self, current: int, total: int):
        """Fortschritt aktualisieren (thread-safe)."""
        with self._lock:
            self._progress = f"{current}/{total}"

    def _run(self):
        """Haupt-Batch-Schleife. Laeuft im Background-Thread."""
        try:
            logger.info("[EINPRAEGEN] Starte Batch-Analyse...")
            images = _collect_images()
            total = len(images)
            if total == 0:
                logger.info("[EINPRAEGEN] Keine Bilder gefunden.")
                self._update_progress(0, 0)
                return
            self._stats["total"] = total
            logger.info(f"[EINPRAEGEN] {total} Bilder gesammelt")

            # NPU-Kontext aufbauen (eigenes VDevice fuer Batch)
            # PHASE 1: SCRFD + ArcFace
            logger.info("[EINPRAEGEN] Phase 1: SCRFD + ArcFace (Gesichter)")
            face_db = _load_existing_face_db()

            # Referenz-Embedding fuer Markus laden (fuer Qualitaetscheck)
            markus_ref = None
            if "Markus" in face_db:
                markus_ref = np.array(face_db["Markus"], dtype=np.float32)
                norm = np.linalg.norm(markus_ref)
                if norm > 0:
                    markus_ref = markus_ref / norm

            # Gesammelte Embeddings + Metadata pro Bild
            image_faces = {}  # {bild_pfad: [(face_bbox, embedding, name, sim)]}

            params = VDevice.create_params()
            vdevice = VDevice(params)
            try:
                # SCRFD laden
                scrfd_model = vdevice.create_infer_model(SCRFD_HEF)
                scrfd_model.input().set_format_type(FormatType.UINT8)
                scrfd_hef = HEF(SCRFD_HEF)
                scrfd_out_names = [o.name for o in scrfd_hef.get_output_vstream_infos()]
                for oname in scrfd_out_names:
                    scrfd_model.output(oname).set_format_type(FormatType.FLOAT32)

                # ArcFace laden
                arcface_model = vdevice.create_infer_model(ARCFACE_HEF)
                arcface_model.input().set_format_type(FormatType.UINT8)
                arcface_hef = HEF(ARCFACE_HEF)
                arcface_out_names = [o.name for o in arcface_hef.get_output_vstream_infos()]
                for oname in arcface_out_names:
                    arcface_model.output(oname).set_format_type(FormatType.FLOAT32)

                # Konfigurieren
                scrfd_ctx = scrfd_model.configure().__enter__()
                scrfd_bufs = {n: np.empty(scrfd_model.output(n).shape, dtype=np.float32) for n in scrfd_out_names}
                scrfd_bindings = scrfd_ctx.create_bindings(output_buffers=scrfd_bufs)

                arcface_ctx = arcface_model.configure().__enter__()
                arcface_bufs = {n: np.empty(arcface_model.output(n).shape, dtype=np.float32) for n in arcface_out_names}
                arcface_bindings = arcface_ctx.create_bindings(output_buffers=arcface_bufs)

                # Alle Bilder durchgehen
                from core.perception.hailo_postprocess import decode_scrfd, normalize_arcface, match_face

                for idx, img_path in enumerate(images):
                    if not self._running:
                        break
                    self._update_progress(idx + 1, total)

                    try:
                        frame = cv2.imread(img_path)
                        if frame is None:
                            continue
                        fh, fw = frame.shape[:2]

                        # SCRFD: 640x640 resize
                        input_640 = cv2.resize(frame, (640, 640))
                        input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)
                        scrfd_bindings.input().set_buffer(np.ascontiguousarray(input_rgb))
                        scrfd_ctx.run([scrfd_bindings], timeout=10000)
                        scrfd_outputs = {n: scrfd_bufs[n].copy() for n in scrfd_out_names}

                        faces = decode_scrfd(scrfd_outputs, 640, SCRFD_CONF, SCRFD_NMS)
                        if not faces:
                            continue

                        image_faces[img_path] = []

                        for (x1n, y1n, x2n, y2n, conf) in faces:
                            # Pixel-Koordinaten im Original
                            x1 = max(0, int(x1n * fw))
                            y1 = max(0, int(y1n * fh))
                            x2 = min(fw, int(x2n * fw))
                            y2 = min(fh, int(y2n * fh))
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

                            # ArcFace Inference
                            arcface_bindings.input().set_buffer(np.ascontiguousarray(crop_rgb))
                            arcface_ctx.run([arcface_bindings], timeout=10000)
                            emb_key = arcface_out_names[0]
                            embedding = arcface_bufs[emb_key].copy().flatten()
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm

                            # Qualitaetscheck gegen bestehende Markus-Referenz
                            name = "unknown"
                            sim = 0.0
                            if markus_ref is not None:
                                sim = float(np.dot(embedding, markus_ref))
                                if sim >= ARCFACE_MIN_SIM:
                                    name = "Markus"
                                else:
                                    self._stats["faces_skipped_unsicher"] += 1
                                    logger.debug(f"[EINPRAEGEN] {os.path.basename(img_path)}: "
                                                 f"Sim={sim:.3f} < {ARCFACE_MIN_SIM} → uebersprungen")
                                    continue
                            else:
                                # Keine Referenz → alles als Markus speichern (Ersteinrichtung)
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
                        logger.warning(f"[EINPRAEGEN] Fehler bei {img_path}: {e}")
                        continue

                # Face-DB speichern
                _save_face_db(face_db)
                logger.info(f"[EINPRAEGEN] Phase 1 fertig: {self._stats['faces_saved']} Faces gespeichert, "
                            f"{self._stats['faces_skipped_small']} zu klein, "
                            f"{self._stats['faces_skipped_unsicher']} unsicher")

            finally:
                # NPU freigeben (Phase 1)
                try:
                    scrfd_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                try:
                    arcface_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                del vdevice

            # PHASE 2: SCRFD + Pose (Skelett/Koerperhaltung)
            logger.info("[EINPRAEGEN] Phase 2: SCRFD + Pose (Koerperhaltung)")
            pose_db = _load_existing_pose_db()

            # Nur Bilder mit erkannten Gesichtern analysieren
            face_images = [p for p in images if p in image_faces and image_faces[p]]
            if not face_images:
                logger.info("[EINPRAEGEN] Keine Bilder mit Gesichtern fuer Pose-Analyse")
            else:
                params2 = VDevice.create_params()
                vdevice2 = VDevice(params2)
                try:
                    # Pose-Modell laden
                    pose_model = vdevice2.create_infer_model(POSE_HEF)
                    pose_model.input().set_format_type(FormatType.UINT8)
                    pose_hef = HEF(POSE_HEF)
                    pose_out_names = [o.name for o in pose_hef.get_output_vstream_infos()]
                    for oname in pose_out_names:
                        pose_model.output(oname).set_format_type(FormatType.FLOAT32)

                    pose_ctx = pose_model.configure().__enter__()
                    pose_bufs = {n: np.empty(pose_model.output(n).shape, dtype=np.float32) for n in pose_out_names}
                    pose_bindings = pose_ctx.create_bindings(output_buffers=pose_bufs)

                    from core.perception.hailo_postprocess import decode_yolov8_pose

                    for idx, img_path in enumerate(face_images):
                        if not self._running:
                            break
                        self._update_progress(idx + 1, len(face_images))

                        try:
                            frame = cv2.imread(img_path)
                            if frame is None:
                                continue
                            fh, fw = frame.shape[:2]

                            # Pose: 640x640 resize
                            input_640 = cv2.resize(frame, (640, 640))
                            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)
                            pose_bindings.input().set_buffer(np.ascontiguousarray(input_rgb))
                            pose_ctx.run([pose_bindings], timeout=10000)
                            pose_outputs = {n: pose_bufs[n].copy() for n in pose_out_names}

                            persons = decode_yolov8_pose(pose_outputs, 640, 640, conf_thresh=0.3)
                            if not persons:
                                continue

                            # Faces aus Phase 1
                            face_info = image_faces.get(img_path, [])

                            for person in persons:
                                bbox = person.get("bbox", [0, 0, 0, 0])
                                keypoints = person.get("keypoints", [])
                                if not keypoints:
                                    continue

                                # Face mit Person verknuepfen (naechstes Face-Center zu Person-BBox)
                                matched_name = "unknown"
                                matched_sim = 0.0
                                px_center = (bbox[0] + bbox[2]) / 2 * fw
                                py_top = bbox[1] * fh

                                for fi in face_info:
                                    fb = fi["bbox"]
                                    fx_center = (fb[0] + fb[2]) / 2
                                    fy_center = (fb[1] + fb[3]) / 2
                                    if abs(fx_center - px_center) < fw * 0.3 and abs(fy_center - py_top) < fh * 0.3:
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
                                    "keypoints": [[round(k, 4) for k in kp] for kp in keypoints],
                                    "face_sim": matched_sim,
                                }
                                key = f"{matched_name}#pose_{ts}_{idx}"
                                pose_db[key] = profile
                                self._stats["poses_saved"] += 1

                        except Exception as e:
                            logger.warning(f"[EINPRAEGEN] Pose-Fehler bei {img_path}: {e}")
                            continue

                    # Pose-DB speichern
                    _save_pose_db(pose_db)
                    logger.info(f"[EINPRAEGEN] Phase 2 fertig: {self._stats['poses_saved']} Pose-Profile gespeichert")

                finally:
                    try:
                        pose_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                    del vdevice2

            # Fertig
            logger.info(f"[EINPRAEGEN] Komplett: {self._stats}")

        except Exception as e:
            logger.error(f"[EINPRAEGEN] Batch-Analyse fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
        finally:
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
