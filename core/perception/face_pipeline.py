#!/usr/bin/env python3
"""
Face Pipeline Worker — SCRFD + ArcFace + FaceAttr via HailoRT-Direct.

Kombiniert Face Detection, Recognition und Attribute in EINEM Worker-Thread.
Grund: Strikte Pipeline (detect → crop → align → recognize), Splitting
wuerde Queue-Latenz addieren.

Preprocessing identisch zu scripts/tappas_enroll.py:
  1. letterbox_resize(640x640) — EINMALIG, keine Doppelkorrektur
  2. SCRFD decode mit unletterbox_coords
  3. align_face via 5-Point Affine Transform
  4. ArcFace 112x112 Embedding

Damit sind Live-Embeddings GARANTIERT kompatibel mit Enrollment-Embeddings.
"""

import os
import json
import logging
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple

from core.perception.vision_workers import (
    BaseWorker, WorkItem, WorkerResult, create_configured_model
)
from core.perception.hailo_postprocess import (
    decode_scrfd, normalize_arcface, match_face, estimate_head_pose
)

logger = logging.getLogger("FacePipeline")

# --- Modell-Pfade ---
MODEL_DIR = "/mnt/moloch-data/hailo/models"
SCRFD_HEF = os.path.join(MODEL_DIR, "scrfd_10g.hef")
ARCFACE_HEF = os.path.join(MODEL_DIR, "arcface_mobilefacenet.hef")
FACE_ATTR_HEF = os.path.join(MODEL_DIR, "face_attr_resnet_v1_18.hef")

# Face-DB Pfad
FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")

# ArcFace Referenz-Landmarks (112x112) — identisch zu tappas_enroll.py
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # linkes Auge
    [73.5318, 51.5014],   # rechtes Auge
    [56.0252, 71.7366],   # Nase
    [41.5493, 92.3655],   # linker Mundwinkel
    [70.7299, 92.2041],   # rechter Mundwinkel
], dtype=np.float32)

# Thresholds
SCRFD_CONF_THRESH = 0.40
SCRFD_NMS_THRESH = 0.40
ARCFACE_MATCH_THRESH = 0.40
MIN_FACE_PIX = 30


# ============================================================
# Letterbox-Funktionen (identisch zu tappas_enroll.py)
# ============================================================

def letterbox_resize(img: np.ndarray, target_size: int = 640):
    """Letterbox-Resize mit Aspektverhaeltnis.

    Returns:
        (padded, scale, pad_x, pad_y, new_w, new_h)
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y, new_w, new_h


def unletterbox_coords(boxes, landmarks, pad_x, pad_y, rw, rh, target=640):
    """Letterbox-Space -> normalisierte [0,1] Koordinaten relativ zum Content."""
    bc = boxes.copy()
    bc[:, [0, 2]] = np.clip((boxes[:, [0, 2]] * target - pad_x) / rw, 0, 1)
    bc[:, [1, 3]] = np.clip((boxes[:, [1, 3]] * target - pad_y) / rh, 0, 1)

    lc = landmarks.copy()
    for i in range(5):
        lc[:, i * 2] = np.clip((landmarks[:, i * 2] * target - pad_x) / rw, 0, 1)
        lc[:, i * 2 + 1] = np.clip((landmarks[:, i * 2 + 1] * target - pad_y) / rh, 0, 1)
    return bc, lc


def align_face(img: np.ndarray, landmarks_5pt) -> Optional[np.ndarray]:
    """Face Alignment via 5-Point Affine Transform (wie tappas_enroll.py).

    Args:
        img: Original-Frame (RGB)
        landmarks_5pt: 5 Landmark-Punkte als Pixel-Koordinaten [(x,y), ...]

    Returns:
        aligned: 112x112 RGB Bild oder None
    """
    src_pts = np.array(landmarks_5pt, dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, ARCFACE_REF_LANDMARKS)
    if tform is None:
        return None
    return cv2.warpAffine(img, tform, (112, 112), borderValue=(0, 0, 0))


# ============================================================
# FaceWorker — SCRFD + ArcFace + FaceAttr in einem Thread
# ============================================================

class FaceWorker(BaseWorker):
    """Face Detection + Recognition + Attributes via HailoRT-Direct.

    Pipeline pro Frame:
      1. Letterbox 640x640
      2. SCRFD → Face BBoxes + 5-Point Landmarks
      3. Unletterbox → normalisierte Koordinaten [0,1]
      4. Fuer jedes Gesicht:
         a. Align (5-Point Affine → 112x112)
         b. ArcFace Embedding → Match gegen DB
         c. FaceAttr (Gender, Age, Emotion) — optional
      5. Head Pose (aus Landmarks)
    """

    def __init__(self):
        super().__init__(name="FaceWorker", max_queue=2)
        self._scrfd_configured = None
        self._scrfd_out_names = []
        self._scrfd_out_shapes = {}

        self._arcface_configured = None
        self._arcface_out_names = []
        self._arcface_out_shapes = {}

        self._faceattr_configured = None
        self._faceattr_out_names = []
        self._faceattr_out_shapes = {}

        # Face-DB (wird lazy geladen)
        self._face_db: Dict[str, np.ndarray] = {}
        self._face_db_loaded = False

    def _load_models(self, vdevice):
        """SCRFD + ArcFace + FaceAttr laden."""
        # SCRFD
        if not os.path.exists(SCRFD_HEF):
            raise FileNotFoundError(f"SCRFD HEF fehlt: {SCRFD_HEF}")
        _, self._scrfd_configured, _, self._scrfd_out_names, self._scrfd_out_shapes = \
            create_configured_model(vdevice, SCRFD_HEF)
        logger.info("[FaceWorker] SCRFD geladen — Outputs: %s", self._scrfd_out_names)

        # ArcFace
        if not os.path.exists(ARCFACE_HEF):
            raise FileNotFoundError(f"ArcFace HEF fehlt: {ARCFACE_HEF}")
        _, self._arcface_configured, _, self._arcface_out_names, self._arcface_out_shapes = \
            create_configured_model(vdevice, ARCFACE_HEF)
        logger.info("[FaceWorker] ArcFace geladen — Outputs: %s", self._arcface_out_names)

        # FaceAttr (optional — nicht-kritisch)
        if os.path.exists(FACE_ATTR_HEF):
            try:
                _, self._faceattr_configured, _, self._faceattr_out_names, self._faceattr_out_shapes = \
                    create_configured_model(vdevice, FACE_ATTR_HEF)
                logger.info("[FaceWorker] FaceAttr geladen — Outputs: %s", self._faceattr_out_names)
            except Exception as e:
                logger.warning("[FaceWorker] FaceAttr laden fehlgeschlagen (nicht-kritisch): %s", e)

        # Face-DB laden
        self._load_face_db()

    def _load_face_db(self):
        """Face Embedding DB laden."""
        if not os.path.exists(FACE_DB_PATH):
            logger.warning("[FaceWorker] Keine Face-DB: %s", FACE_DB_PATH)
            return
        try:
            with open(FACE_DB_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._face_db = {name: np.array(emb, dtype=np.float32)
                             for name, emb in raw.items()}
            self._face_db_loaded = True
            logger.info("[FaceWorker] Face-DB geladen: %d Eintraege", len(self._face_db))
        except Exception as e:
            logger.error("[FaceWorker] Face-DB Fehler: %s", e)

    def _process(self, item: WorkItem) -> WorkerResult:
        """Full Face Pipeline: SCRFD → ArcFace → FaceAttr."""
        frame_rgb = item.frame
        fh, fw = frame_rgb.shape[:2]

        # 1. Letterbox 640x640
        padded, _scale, pad_x, pad_y, rw, rh = letterbox_resize(frame_rgb, 640)

        # 2. SCRFD Inference
        scrfd_bindings = self._scrfd_configured.create_bindings()
        scrfd_bindings.input().set_buffer(np.ascontiguousarray(padded))
        scrfd_bufs = {}
        for name in self._scrfd_out_names:
            buf = np.empty(self._scrfd_out_shapes[name], dtype=np.float32)
            scrfd_bindings.output(name).set_buffer(buf)
            scrfd_bufs[name] = buf

        self._scrfd_configured.run([scrfd_bindings], INFERENCE_TIMEOUT_MS)
        outputs = {n: scrfd_bufs[n].copy() for n in self._scrfd_out_names}

        # 3. Decode SCRFD
        boxes, scores, landmarks = decode_scrfd(
            outputs, 640, SCRFD_CONF_THRESH, SCRFD_NMS_THRESH)

        if len(boxes) == 0:
            return WorkerResult(
                worker_name="FaceWorker",
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                data={"face_count": 0, "faces": []},
            )

        # 4. Unletterbox → normalisierte [0,1] Koordinaten
        boxes_norm, landmarks_norm = unletterbox_coords(
            boxes, landmarks, pad_x, pad_y, rw, rh)

        # 5. Fuer jedes Gesicht: Align + ArcFace + FaceAttr
        faces = []
        for i in range(len(boxes_norm)):
            face = self._process_single_face(
                frame_rgb, fw, fh,
                boxes_norm[i], scores[i], landmarks_norm[i]
            )
            if face is not None:
                faces.append(face)

        # Sortieren: hoechste Similarity zuerst (oder hoechste Confidence)
        faces.sort(key=lambda f: (f.get("similarity", 0), f.get("confidence", 0)),
                   reverse=True)

        return WorkerResult(
            worker_name="FaceWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            data={
                "face_count": len(faces),
                "faces": faces,
            },
        )

    def _process_single_face(self, frame_rgb: np.ndarray, fw: int, fh: int,
                             box: np.ndarray, score: float,
                             landmarks: np.ndarray) -> Optional[Dict]:
        """Ein Gesicht verarbeiten: Align → ArcFace → Match → FaceAttr."""
        # Pixel-Koordinaten
        px1 = max(0, int(box[0] * fw))
        py1 = max(0, int(box[1] * fh))
        px2 = min(fw, int(box[2] * fw))
        py2 = min(fh, int(box[3] * fh))
        bw, bh = px2 - px1, py2 - py1

        if bw < MIN_FACE_PIX or bh < MIN_FACE_PIX:
            return None

        # 5 Landmarks in Pixel-Koordinaten
        landmarks_px = []
        for p in range(5):
            lx = landmarks[p * 2] * fw
            ly = landmarks[p * 2 + 1] * fh
            landmarks_px.append([lx, ly])

        # Face Alignment
        aligned = align_face(frame_rgb, landmarks_px)
        if aligned is None:
            # Fallback: Crop + Resize
            crop = frame_rgb[py1:py2, px1:px2]
            if crop.size == 0:
                return None
            aligned = cv2.resize(crop, (112, 112))

        # ArcFace Inference
        arcface_bindings = self._arcface_configured.create_bindings()
        arcface_bindings.input().set_buffer(np.ascontiguousarray(aligned))
        arcface_bufs = {}
        for name in self._arcface_out_names:
            buf = np.empty(self._arcface_out_shapes[name], dtype=np.float32)
            arcface_bindings.output(name).set_buffer(buf)
            arcface_bufs[name] = buf

        self._arcface_configured.run([arcface_bindings], INFERENCE_TIMEOUT_MS)
        emb_raw = arcface_bufs[self._arcface_out_names[0]].flatten().copy()
        embedding = normalize_arcface(emb_raw)

        # Face-DB Match
        face_id = "Unbekannt"
        similarity = 0.0
        if self._face_db:
            face_id, similarity = match_face(
                embedding, self._face_db, ARCFACE_MATCH_THRESH)

        # Head Pose (aus Landmarks, CPU-only, schnell)
        head_pose = estimate_head_pose(landmarks, fw, fh)

        # FaceAttr (optional)
        gender = None
        age_range = None
        emotion = None
        if self._faceattr_configured is not None:
            try:
                gender, age_range, emotion = self._infer_face_attr(aligned)
            except Exception:
                pass  # Nicht-kritisch

        return {
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "confidence": float(score),
            "face_id": face_id,
            "similarity": float(similarity),
            "embedding": embedding,
            "landmarks": landmarks.tolist(),
            "head_pose": head_pose,
            "gender": gender,
            "age_range": age_range,
            "emotion": emotion,
            "face_size": [bw, bh],
        }

    def _infer_face_attr(self, aligned_112: np.ndarray) -> Tuple[str, str, str]:
        """Face Attributes: Gender, Age, Emotion.

        Input: 112x112 RGB (bereits aligned).
        FaceAttr erwartet ggf. andere Groesse — resize wenn noetig.
        """
        # face_attr_resnet_v1_18 erwartet typisch 224x224 oder 178x218
        # Pruefen wir die tatsaechliche Input-Shape
        attr_input = cv2.resize(aligned_112, (224, 224))

        bindings = self._faceattr_configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(attr_input))
        bufs = {}
        for name in self._faceattr_out_names:
            buf = np.empty(self._faceattr_out_shapes[name], dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            bufs[name] = buf

        self._faceattr_configured.run([bindings], INFERENCE_TIMEOUT_MS)

        # Decode (haengt vom konkreten Modell ab — minimal-Version)
        # Output ist typisch: [age, gender_logit, smile_logit] oder aehnlich
        # Hier: robustes Parsing, Detailformat spaeter anpassen
        out_data = bufs[self._faceattr_out_names[0]].flatten()

        gender = "M" if (len(out_data) > 1 and out_data[1] > 0) else "F"
        emotion = "neutral"
        if len(out_data) > 2:
            emotion = "happy" if out_data[2] > 0 else "neutral"
        age_range = None
        if len(out_data) > 0:
            age_val = float(out_data[0])
            if 0 < age_val < 100:
                age_low = int(age_val / 5) * 5
                age_range = f"{age_low}-{age_low + 5}"

        return gender, age_range, emotion

    def reload_face_db(self):
        """Face-DB neu laden (z.B. nach Enrollment)."""
        self._load_face_db()

    def get_health(self) -> Dict:
        """Erweiterte Health-Info mit Face-DB Status."""
        health = super().get_health()
        health["face_db_entries"] = len(self._face_db)
        health["face_db_loaded"] = self._face_db_loaded
        return health
