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

    Nutzt Pose-Keypoints (Wrist = 9/10) um Hand-Crops auszuschneiden.
    hand_landmark_lite ist ein Crop-Modell (224x224) — Full-Frame funktioniert nicht.

    Fix 2026-04-03: Diagnose via Claude<->MOLOCH MCP-Gespraech.
    """

    # COCO Wrist-Keypoints
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    # Crop-Groesse relativ zur Frame-Hoehe (wie weit um Handgelenk herum)
    CROP_EXPAND = 0.20  # 20% der Frame-Hoehe als Crop-Radius (vorher 0.15 — zu klein)
    # Wrist liegt am unteren Rand der Hand — Crop nach oben verschieben
    # 0.5 = halber Radius nach oben, damit Hand zentriert im Crop liegt
    WRIST_Y_SHIFT = 0.5

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

    def _extract_wrist_crops(self, frame_rgb, detections):
        """Wrist-Keypoints aus Pose-Detections extrahieren, Crops ausschneiden.

        Returns:
            Liste von (crop_224, cx_norm, cy_norm, crop_radius_norm, side) Tupeln
        """
        fh, fw = frame_rgb.shape[:2]
        crops = []
        for det in detections:
            if det.get("class") != "pose":
                continue
            kpts = det.get("keypoints", [])
            if len(kpts) < 11:
                continue
            # Beide Handgelenke pruefen
            for idx, side in [(self.LEFT_WRIST, "L"), (self.RIGHT_WRIST, "R")]:
                kp = kpts[idx]
                if len(kp) < 3 or kp[2] < 0.15:
                    continue  # Confidence zu niedrig
                # Normalisierte Coords [0,1] → Pixel
                cx, cy = kp[0], kp[1]
                px, py = int(cx * fw), int(cy * fh)
                radius = int(self.CROP_EXPAND * fh)
                # Crop-Mitte nach oben verschieben (Wrist = unteres Drittel)
                # Hand liegt OBERHALB des Handgelenks → Crop muss hoeher sitzen
                py_center = py - int(radius * self.WRIST_Y_SHIFT)
                # Crop-Grenzen (mit Clipping)
                x1 = max(0, px - radius)
                y1 = max(0, py_center - radius)
                x2 = min(fw, px + radius)
                y2 = min(fh, py_center + radius)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue  # Crop zu klein
                crop = frame_rgb[y1:y2, x1:x2]
                crop_224 = cv2.resize(crop, (224, 224))
                # Normalisierte Crop-Position fuer Rueck-Mapping
                crops.append((crop_224, x1 / fw, y1 / fh,
                              (x2 - x1) / fw, (y2 - y1) / fh, side))
        return crops

    def _run_inference(self, crop_224):
        """Hand Landmark Inference auf einem 224x224 Crop."""
        bindings = self._hand_configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(crop_224))
        bufs = {}
        for name in self._hand_out_names:
            buf = np.empty(self._hand_out_shapes[name], dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            bufs[name] = buf
        self._hand_configured.run([bindings], INFERENCE_TIMEOUT_MS)
        return {n: bufs[n].copy() for n in self._hand_out_names}

    def _process(self, item: WorkItem) -> WorkerResult:
        frame_rgb = item.frame
        detections = item.detections or []

        # Wrist-Crops aus Pose-Keypoints extrahieren
        crops = self._extract_wrist_crops(frame_rgb, detections)

        hands = []
        for crop_224, crop_x, crop_y, crop_w, crop_h, side in crops:
            outputs = self._run_inference(crop_224)
            result = decode_hand_landmark(outputs, presence_thresh=0.3)
            if result is None:
                continue
            # Landmarks von Crop-Space [0,1] auf Frame-Space [0,1] mappen
            landmarks = result["landmarks"]  # (21, 3) normalisiert auf [0,1] im Crop
            mapped = []
            for i in range(21):
                lx = crop_x + landmarks[i, 0] * crop_w  # x: Crop-Offset + relativ
                ly = crop_y + landmarks[i, 1] * crop_h  # y: Crop-Offset + relativ
                lz = float(landmarks[i, 2])
                mapped.append([lx, ly, lz])
            # BBox um alle Landmarks (fuer Panel-Overlay)
            xs = [p[0] for p in mapped]
            ys = [p[1] for p in mapped]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            hands.append({
                "landmarks": mapped,
                "bbox": bbox,
                "handedness": side,
                "presence": result.get("presence", 0.0),
            })

        return WorkerResult(
            worker_name="HandWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            data={
                "hand_detected": len(hands) > 0,
                "hand": hands[0] if hands else None,
                "hands": hands,
                "hand_count": len(hands),
            },
        )
