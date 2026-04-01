#!/usr/bin/env python3
"""
YOLOWorldWorker — Zero-Shot Objekterkennung via YOLO-World v2s.

Erkennt beliebige Objekte per Sprachbeschreibung (Zero-Shot).
Standard-Klassen: 80 Home/Office-Objekte (pre-computed CLIP Embeddings).
Dynamisch: Klassen via IPC 'yolo_world_set_classes' aenderbar.

Inputs:
  input_layer1: [640, 640, 3] uint8  — Bild (Letterbox)
  input_layer2: [1, 80, 512] float32 — CLIP Text-Embeddings

Outputs: 6 YOLO DFL-Koepfe (3 Scales, je BBox + Cls)
FPS: ~45

Embeddings: /mnt/moloch-data/yolo_world/default_embeddings.npy
Klassen:    /mnt/moloch-data/yolo_world/default_classes.json
"""

import os
import json
import logging
import threading
import numpy as np
import cv2
from typing import Optional, List, Dict

from core.perception.vision_workers import (
    BaseWorker, WorkItem, WorkerResult, create_configured_model,
    INFERENCE_TIMEOUT_MS
)

logger = logging.getLogger("YOLOWorldWorker")

MODEL_DIR   = "/mnt/moloch-data/hailo/models"
YOLO_WORLD_HEF = os.path.join(MODEL_DIR, "yolo_world_v2s.hef")
EMBED_DIR   = "/mnt/moloch-data/yolo_world"
EMBED_FILE  = os.path.join(EMBED_DIR, "default_embeddings.npy")
CLASS_FILE  = os.path.join(EMBED_DIR, "default_classes.json")

NUM_CLASSES = 80
CONF_THRESH = 0.35
REG_MAX = 16   # DFL bins

# Strides der 3 YOLO-Scales fuer 640x640 Input
STRIDES = [8, 16, 32]
GRID_SIZES = [80, 40, 20]  # 640/8, 640/16, 640/32

# Output-Name-Mapping: bbox + cls per Scale
# Hailo-Outputs (alphabetisch sortiert nach Index):
# conv48  [80,80,64]  — bbox scale0
# normalization3 [80,80,80] — cls scale0
# conv60  [40,40,64]  — bbox scale1
# normalization5 [40,40,80] — cls scale1
# conv71  [20,20,64]  — bbox scale2
# normalization7 [20,20,80] — cls scale2
OUTPUT_SCALES = [
    ("yolo_world_v2s/conv48",        "yolo_world_v2s/normalization3",  8),
    ("yolo_world_v2s/conv60",        "yolo_world_v2s/normalization5", 16),
    ("yolo_world_v2s/conv71",        "yolo_world_v2s/normalization7", 32),
]


def _decode_yolo_world(outputs: Dict, conf_thresh: float, classes: List[str],
                        img_w: int = 640, img_h: int = 640):
    """DFL + Sigmoid Decoder fuer YOLO-World v2s 6-Output-Kopf."""
    reg_range = np.arange(REG_MAX, dtype=np.float32)
    results = []

    for bbox_name, cls_name, stride in OUTPUT_SCALES:
        bbox_raw = outputs.get(bbox_name)  # [H, W, 64]
        cls_raw  = outputs.get(cls_name)   # [H, W, 80]
        if bbox_raw is None or cls_raw is None:
            continue

        H, W = bbox_raw.shape[:2]
        N = H * W
        bbox_flat = bbox_raw.reshape(N, 4, REG_MAX)   # (N, 4, 16)
        cls_flat  = cls_raw.reshape(N, NUM_CLASSES)    # (N, 80)

        # Sigmoid fuer Klassen-Scores
        cls_scores = 1.0 / (1.0 + np.exp(-cls_flat))  # (N, 80)
        max_scores = cls_scores.max(axis=1)             # (N,)
        best_cls   = cls_scores.argmax(axis=1)          # (N,)

        mask = max_scores > conf_thresh
        if not mask.any():
            continue

        idx = np.where(mask)[0]

        # Gitter-Koordinaten
        gy = (idx // W).astype(np.float32)
        gx = (idx  % W).astype(np.float32)
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride

        # DFL Box decode
        box = bbox_flat[idx]  # (K, 4, 16)
        box_exp = np.exp(box - box.max(axis=-1, keepdims=True))
        box_soft = box_exp / box_exp.sum(axis=-1, keepdims=True)
        dist = (box_soft * reg_range).sum(axis=-1) * stride  # (K, 4)

        # x1y1x2y2 in Pixel
        x1 = np.clip(cx - dist[:, 0], 0, img_w)
        y1 = np.clip(cy - dist[:, 1], 0, img_h)
        x2 = np.clip(cx + dist[:, 2], 0, img_w)
        y2 = np.clip(cy + dist[:, 3], 0, img_h)

        for k in range(len(idx)):
            cls_id = int(best_cls[idx[k]])
            score  = float(max_scores[idx[k]])
            label  = classes[cls_id] if cls_id < len(classes) else f"cls_{cls_id}"
            if label == "__pad__":
                continue
            results.append({
                "label": label,
                "score": score,
                "bbox": [
                    float(x1[k]) / img_w,
                    float(y1[k]) / img_h,
                    float(x2[k]) / img_w,
                    float(y2[k]) / img_h,
                ],
            })

    # Einfaches NMS: absteigend nach Score sortieren
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:20]  # Max 20 Detektionen


class YOLOWorldWorker(BaseWorker):
    """Zero-Shot Objekterkennung — beliebige Klassen per Text-Abfrage."""

    def __init__(self):
        super().__init__(name="YOLOWorldWorker", max_queue=2)
        self._model = None
        self._out_names = []
        self._out_shapes = {}
        self._classes: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # [1, 80, 512] float32
        self._embed_lock = threading.Lock()

    def _load_models(self, vdevice):
        if not os.path.exists(YOLO_WORLD_HEF):
            raise FileNotFoundError(f"YOLO-World HEF fehlt: {YOLO_WORLD_HEF}")

        _, self._model, self._in_names, self._out_names, self._out_shapes = \
            create_configured_model(vdevice, YOLO_WORLD_HEF)
        logger.info("[YOLOWorldWorker] Modell geladen — Inputs: %s Outputs: %s",
                    self._in_names, self._out_names)

        self._load_default_embeddings()

    def _load_default_embeddings(self):
        """Pre-computed CLIP Embeddings laden."""
        if not os.path.exists(EMBED_FILE) or not os.path.exists(CLASS_FILE):
            logger.warning("[YOLOWorldWorker] Keine Default-Embeddings gefunden: %s", EMBED_DIR)
            return
        emb = np.load(EMBED_FILE).astype(np.float32)      # [1, 80, 512]
        cls = json.load(open(CLASS_FILE))
        with self._embed_lock:
            self._embeddings = emb
            self._classes = cls
        logger.info("[YOLOWorldWorker] %d Default-Klassen geladen", len(cls))

    def set_classes(self, classes: List[str], embeddings: np.ndarray):
        """Klassen und Embeddings zur Laufzeit ersetzen (thread-safe)."""
        with self._embed_lock:
            self._classes = classes
            self._embeddings = embeddings.astype(np.float32)
        logger.info("[YOLOWorldWorker] Klassen gewechselt: %s", classes[:5])

    def _process(self, item: WorkItem) -> WorkerResult:
        with self._embed_lock:
            emb = self._embeddings
            classes = list(self._classes)

        if emb is None:
            return WorkerResult(
                worker_name="YOLOWorldWorker",
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                success=False,
                data={"error": "Keine Embeddings geladen"}
            )

        frame_rgb = item.frame
        # Letterbox auf 640x640
        h, w = frame_rgb.shape[:2]
        scale = min(640 / w, 640 / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_rgb, (nw, nh))
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        px = (640 - nw) // 2
        py = (640 - nh) // 2
        padded[py:py+nh, px:px+nw] = resized
        img_input = np.ascontiguousarray(padded, dtype=np.uint8)

        # Bindings
        bindings = self._model.create_bindings()

        # Input 1: Bild
        bindings.input("yolo_world_v2s/input_layer1").set_buffer(img_input)
        # Input 2: Text-Embeddings
        bindings.input("yolo_world_v2s/input_layer2").set_buffer(
            np.ascontiguousarray(emb, dtype=np.float32)
        )

        bufs = {}
        for name in self._out_names:
            buf = np.empty(self._out_shapes[name], dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            bufs[name] = buf

        self._model.run([bindings], INFERENCE_TIMEOUT_MS)

        detections = _decode_yolo_world(bufs, CONF_THRESH, classes)

        if detections:
            top = detections[0]
            logger.info("[YOLOWorldWorker] Top: %s (%.2f)", top["label"], top["score"])

        return WorkerResult(
            worker_name="YOLOWorldWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            success=True,
            data={
                "detections": detections,
                "count": len(detections),
                "classes_active": len([c for c in classes if c != "__pad__"]),
            }
        )


_instance: Optional[YOLOWorldWorker] = None


def get_yolo_world_worker() -> YOLOWorldWorker:
    """Singleton-Getter."""
    global _instance
    if _instance is None:
        _instance = YOLOWorldWorker()
    return _instance
