#!/usr/bin/env python3
"""
Pose + ReID + Hand Worker — HailoRT-Direct, crash-isoliert.

Jeder Worker laeuft in eigenem Thread auf dem SHARED VDevice.
Kein GStreamer, kein C-Code, kein SEGV-Risiko.
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, Optional, List

from core.perception.vision_workers import (
    BaseWorker, WorkItem, WorkerResult, create_configured_model,
    INFERENCE_TIMEOUT_MS
)
from core.perception.hailo_postprocess import (
    decode_yolov8_pose, normalize_arcface, decode_hand_landmark
)

logger = logging.getLogger("PoseWorker")

# --- Modell-Pfade ---
MODEL_DIR = "/mnt/moloch-data/hailo/models"
POSE_HEF = os.path.join(MODEL_DIR, "yolov8s_pose_h10.hef")
REID_HEF = os.path.join(MODEL_DIR, "repvgg_a0_person_reid_512.hef")
HAND_HEF = os.path.join(MODEL_DIR, "hand_landmark_lite.hef")


def letterbox_resize(img: np.ndarray, target_size: int = 640):
    """Letterbox-Resize (identisch zu face_pipeline.py)."""
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


# ============================================================
# PoseWorker — YOLOv8s Pose via HailoRT-Direct
# ============================================================

class PoseWorker(BaseWorker):
    """Pose Estimation (17 Keypoints) via HailoRT-Direct.

    Vorher: SEGV wegen HAILO_LANDMARKS Race Condition in GStreamer C-Code.
    Jetzt: Pure Python numpy — kein C-Code, kein SEGV.
    """

    def __init__(self):
        super().__init__(name="PoseWorker", max_queue=2)
        self._pose_configured = None
        self._pose_out_names = []
        self._pose_out_shapes = {}

    def _load_models(self, vdevice):
        if not os.path.exists(POSE_HEF):
            raise FileNotFoundError(f"Pose HEF fehlt: {POSE_HEF}")
        _, self._pose_configured, _, self._pose_out_names, self._pose_out_shapes = \
            create_configured_model(vdevice, POSE_HEF)
        logger.info("[PoseWorker] YOLOv8s-Pose geladen — Outputs: %d", len(self._pose_out_names))

    def _process(self, item: WorkItem) -> WorkerResult:
        frame_rgb = item.frame
        fh, fw = frame_rgb.shape[:2]

        # Letterbox 640x640
        padded, _scale, pad_x, pad_y, rw, rh = letterbox_resize(frame_rgb, 640)

        # Inference
        bindings = self._pose_configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(padded))
        bufs = {}
        for name in self._pose_out_names:
            buf = np.empty(self._pose_out_shapes[name], dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            bufs[name] = buf

        self._pose_configured.run([bindings], INFERENCE_TIMEOUT_MS)
        outputs = {n: bufs[n].copy() for n in self._pose_out_names}

        # Decode
        poses = decode_yolov8_pose(outputs, 640, 640, conf_thresh=0.3)

        # Keypoints von Model-Space (640x640) auf normalisierte [0,1] Coords umrechnen
        for pose in poses:
            # BBox: model pixels → [0,1] (mit Unletterbox)
            bx = pose["bbox"]
            pose["bbox_norm"] = [
                max(0, min(1, (bx[0] - pad_x) / rw)),
                max(0, min(1, (bx[1] - pad_y) / rh)),
                max(0, min(1, (bx[2] - pad_x) / rw)),
                max(0, min(1, (bx[3] - pad_y) / rh)),
            ]
            # Keypoints: model pixels → [0,1]
            kpts = pose["keypoints"]
            for k in range(17):
                kpts[k, 0] = max(0, min(1, (kpts[k, 0] - pad_x) / rw))
                kpts[k, 1] = max(0, min(1, (kpts[k, 1] - pad_y) / rh))

        return WorkerResult(
            worker_name="PoseWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            data={
                "pose_count": len(poses),
                "poses": poses,
            },
        )


# ============================================================
# ReIDWorker — Person Re-Identification via HailoRT-Direct
# ============================================================

class ReIDWorker(BaseWorker):
    """Person ReID (512d Embedding) via HailoRT-Direct.

    Vorher: libre_id.so crashte wegen HAILO_LANDMARKS in Person-Detections.
    Jetzt: Python numpy Crop → HailoRT-Direct — kein C-Code, kein Crash.
    """

    def __init__(self):
        super().__init__(name="ReIDWorker", max_queue=2)
        self._reid_configured = None
        self._reid_out_names = []
        self._reid_out_shapes = {}

    def _load_models(self, vdevice):
        if not os.path.exists(REID_HEF):
            raise FileNotFoundError(f"ReID HEF fehlt: {REID_HEF}")
        _, self._reid_configured, _, self._reid_out_names, self._reid_out_shapes = \
            create_configured_model(vdevice, REID_HEF)
        logger.info("[ReIDWorker] RepVGG-A0 geladen — Outputs: %s", self._reid_out_names)

    def _process(self, item: WorkItem) -> WorkerResult:
        frame_rgb = item.frame
        fh, fw = frame_rgb.shape[:2]

        embeddings = []
        for det in item.detections:
            if det.get("class") != "person":
                continue
            bbox = det.get("bbox", [0, 0, 1, 1])
            # Crop aus Frame
            x1 = max(0, int(bbox[0] * fw))
            y1 = max(0, int(bbox[1] * fh))
            x2 = min(fw, int(bbox[2] * fw))
            y2 = min(fh, int(bbox[3] * fh))
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Resize auf ReID Input (256x128 typisch, pruefen wir shape)
            # RepVGG-A0 erwartet 256x128x3 (HxWxC)
            reid_input = cv2.resize(crop, (128, 256))

            bindings = self._reid_configured.create_bindings()
            bindings.input().set_buffer(np.ascontiguousarray(reid_input))
            bufs = {}
            for name in self._reid_out_names:
                buf = np.empty(self._reid_out_shapes[name], dtype=np.float32)
                bindings.output(name).set_buffer(buf)
                bufs[name] = buf

            self._reid_configured.run([bindings], INFERENCE_TIMEOUT_MS)
            emb = bufs[self._reid_out_names[0]].flatten().copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            embeddings.append({
                "bbox": bbox,
                "embedding": emb,
                "track_id": det.get("track_id"),
            })

        return WorkerResult(
            worker_name="ReIDWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            data={
                "reid_count": len(embeddings),
                "embeddings": embeddings,
            },
        )


# ============================================================
# HandWorker — Hand Landmark via HailoRT-Direct
# ============================================================

class HandWorker(BaseWorker):
    """Hand Landmark Detection (21 Keypoints) via HailoRT-Direct.

    Vorher: cv2::resize Assertion Crash bei Valve-Oeffnung.
    Jetzt: Python cv2.resize + HailoRT-Direct — kontrolliert, kein Crash.
    """

    def __init__(self):
        super().__init__(name="HandWorker", max_queue=2)
        self._hand_configured = None
        self._hand_out_names = []
        self._hand_out_shapes = {}

    def _load_models(self, vdevice):
        if not os.path.exists(HAND_HEF):
            raise FileNotFoundError(f"Hand HEF fehlt: {HAND_HEF}")
        _, self._hand_configured, _, self._hand_out_names, self._hand_out_shapes = \
            create_configured_model(vdevice, HAND_HEF)
        logger.info("[HandWorker] Hand Landmark Lite geladen — Outputs: %s", self._hand_out_names)

    def _process(self, item: WorkItem) -> WorkerResult:
        frame_rgb = item.frame

        # Full-Frame auf 224x224 skalieren (wie in alter Pipeline)
        hand_input = cv2.resize(frame_rgb, (224, 224))

        bindings = self._hand_configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(hand_input))
        bufs = {}
        for name in self._hand_out_names:
            buf = np.empty(self._hand_out_shapes[name], dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            bufs[name] = buf

        self._hand_configured.run([bindings], INFERENCE_TIMEOUT_MS)
        outputs = {n: bufs[n].copy() for n in self._hand_out_names}

        # Decode
        hand_result = decode_hand_landmark(outputs)

        return WorkerResult(
            worker_name="HandWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            data={
                "hand_detected": hand_result is not None,
                "hand": hand_result,
            },
        )
