#!/usr/bin/env python3
"""
Hailo-10H Postprocessing fuer alle 4 NPU-Modelle.

SCRFD: Anchor-basierte Face Detection (9 Output-Layer)
YOLOv8m: On-Chip NMS Person Detection (1 Output-Layer)
YOLOv8s Pose: DFL Box + Keypoint Decode (9 Output-Layer)
ArcFace: L2-Normalize Face Embedding (1 Output-Layer)

Author: M.O.L.O.C.H. System
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


# ============================================================
# SCRFD Face Detection
# ============================================================

# Layer-Namen fuer scrfd_10g.hef (3 Strides x 3 Heads)
SCRFD_BOXES  = ["scrfd_10g/conv42", "scrfd_10g/conv50", "scrfd_10g/conv57"]
SCRFD_SCORES = ["scrfd_10g/conv41", "scrfd_10g/conv49", "scrfd_10g/conv56"]
SCRFD_LANDMARKS = ["scrfd_10g/conv43", "scrfd_10g/conv51", "scrfd_10g/conv58"]
SCRFD_STRIDES = [8, 16, 32]
SCRFD_ANCHORS_PER_CELL = 2  # min_sizes: [[16,32],[64,128],[256,512]]


def decode_scrfd(outputs: Dict[str, np.ndarray], img_size: int = 640,
                 conf_thresh: float = 0.4, iou_thresh: float = 0.4
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode SCRFD raw outputs zu Boxes, Scores, Landmarks.

    Returns:
        boxes: (N, 4) normalized [0,1] xyxy
        scores: (N,) confidence
        landmarks: (N, 10) = 5 points x (x, y) normalized
    """
    all_boxes, all_scores, all_lms = [], [], []

    for i, stride in enumerate(SCRFD_STRIDES):
        scores = outputs[SCRFD_SCORES[i]]     # (H, W, 2)
        boxes = outputs[SCRFD_BOXES[i]]       # (H, W, 8)
        lms = outputs[SCRFD_LANDMARKS[i]]     # (H, W, 20)
        H, W = scores.shape[:2]

        # Anchor-Zentren (vectorized)
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        cx = (cols + 0.5) * stride / img_size  # (H, W)
        cy = (rows + 0.5) * stride / img_size
        sx = stride / img_size
        sy = stride / img_size

        for a in range(SCRFD_ANCHORS_PER_CELL):
            score_map = scores[:, :, a]
            mask = score_map > conf_thresh
            if not mask.any():
                continue

            sc = score_map[mask]
            cx_m, cy_m = cx[mask], cy[mask]

            # Box decode: offset * anchor_scale
            bi = a * 4
            b = boxes[:, :, bi:bi + 4][mask]  # (N, 4)
            x1 = cx_m - b[:, 0] * sx
            y1 = cy_m - b[:, 1] * sy
            x2 = cx_m + b[:, 2] * sx
            y2 = cy_m + b[:, 3] * sy

            all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
            all_scores.append(sc)

            # Landmark decode: 5 Punkte x 2 Coords
            li = a * 10
            lm = lms[:, :, li:li + 10][mask]  # (N, 10)
            decoded_lm = np.zeros_like(lm)
            for p in range(5):
                decoded_lm[:, p * 2] = cx_m + lm[:, p * 2] * sx
                decoded_lm[:, p * 2 + 1] = cy_m + lm[:, p * 2 + 1] * sy
            all_lms.append(decoded_lm)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 10))

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    landmarks = np.concatenate(all_lms)

    # NMS
    keep = _nms(boxes, scores, iou_thresh)
    return boxes[keep], scores[keep], landmarks[keep]


# ============================================================
# YOLOv8m Person Detection (On-Chip NMS)
# ============================================================

def decode_yolov8_nms(output: np.ndarray, class_id: int = 0,
                      conf_thresh: float = 0.5) -> List[Dict]:
    """Parse Hailo On-Chip NMS Output.

    Hailo NMS Format: per class block = float count + count x (y1, x1, y2, x2, score)
    Coordinates normalized [0,1].
    """
    detections = []
    num_classes = 80
    max_bboxes = 100
    entry_size = 5

    flat = output.flatten().astype(np.float32)
    block_size = 1 + max_bboxes * entry_size

    for cls in range(num_classes):
        if class_id >= 0 and cls != class_id:
            continue
        offset = cls * block_size
        if offset >= len(flat):
            break
        count = int(flat[offset])
        count = min(count, max_bboxes)
        for b in range(count):
            base = offset + 1 + b * entry_size
            if base + 5 > len(flat):
                break
            y1, x1, y2, x2, score = flat[base:base + 5]
            if score >= conf_thresh:
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "class": "person" if cls == 0 else f"class_{cls}",
                })
    return detections


# ============================================================
# YOLOv8s Pose (DFL Box + Keypoint Decode)
# ============================================================

# Layer-Namen fuer yolov8s_pose_h10.hef
# Reihenfolge: stride 32 (20x20), 16 (40x40), 8 (80x80)
POSE_BOX_LAYERS   = ["yolov8s_pose/conv70", "yolov8s_pose/conv57", "yolov8s_pose/conv43"]
POSE_SCORE_LAYERS = ["yolov8s_pose/conv71", "yolov8s_pose/conv58", "yolov8s_pose/conv44"]
POSE_KPT_LAYERS   = ["yolov8s_pose/conv72", "yolov8s_pose/conv59", "yolov8s_pose/conv45"]
POSE_STRIDES = [32, 16, 8]  # groesster zuerst
POSE_REG_MAX = 15

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Kopf
    (5, 6), (5, 7), (7, 9), (6, 8),       # Arme
    (8, 10), (5, 11), (6, 12),            # Torso
    (11, 12), (11, 13), (13, 15),         # Linkes Bein
    (12, 14), (14, 16),                   # Rechtes Bein
]


def decode_yolov8_pose(outputs: Dict[str, np.ndarray], img_h: int = 640, img_w: int = 640,
                       conf_thresh: float = 0.3, iou_thresh: float = 0.7,
                       max_det: int = 10) -> List[Dict]:
    """Decode YOLOv8s Pose outputs zu Boxes + Keypoints.

    Returns: Liste von Dicts mit 'bbox' (xyxy pixel), 'score', 'keypoints' (17,3: x,y,vis)
    """
    reg_max = POSE_REG_MAX
    reg_range = np.arange(reg_max + 1, dtype=np.float32)

    all_boxes = []
    all_scores = []
    all_kpts = []

    for i, stride in enumerate(POSE_STRIDES):
        box_raw = outputs[POSE_BOX_LAYERS[i]]    # (H, W, 64)
        score_raw = outputs[POSE_SCORE_LAYERS[i]] # (H, W, 1)
        kpt_raw = outputs[POSE_KPT_LAYERS[i]]    # (H, W, 51)

        H, W = box_raw.shape[:2]
        N = H * W

        # Grid-Zentren
        grid_x = (np.arange(W) + 0.5) * stride
        grid_y = (np.arange(H) + 0.5) * stride
        gx, gy = np.meshgrid(grid_x, grid_y)
        centers = np.stack([gx.flatten(), gy.flatten()], axis=1)  # (N, 2)
        center4 = np.concatenate([centers, centers], axis=1)      # (N, 4)

        # DFL Box decode
        box_flat = box_raw.reshape(N, 4, reg_max + 1)  # (N, 4, 16)
        # Softmax
        box_exp = np.exp(box_flat - np.max(box_flat, axis=-1, keepdims=True))
        box_soft = box_exp / np.sum(box_exp, axis=-1, keepdims=True)
        box_dist = np.sum(box_soft * reg_range, axis=-1) * stride  # (N, 4)
        # left, top negativ; right, bottom positiv
        box_dist[:, :2] *= -1
        decoded = center4 + box_dist  # (N, 4) = x1, y1, x2, y2

        # xywh fuer NMS
        cx = (decoded[:, 0] + decoded[:, 2]) / 2
        cy = (decoded[:, 1] + decoded[:, 3]) / 2
        w = decoded[:, 2] - decoded[:, 0]
        h = decoded[:, 3] - decoded[:, 1]

        # Score (sigmoid)
        score_flat = score_raw.reshape(N)
        score_sig = 1.0 / (1.0 + np.exp(-score_flat))

        # Keypoints decode
        kpt_flat = kpt_raw.reshape(N, 17, 3)
        kpt_decoded = kpt_flat.copy()
        kpt_decoded[:, :, :2] = kpt_decoded[:, :, :2] * 2.0
        kpt_decoded[:, :, 0] = stride * (kpt_decoded[:, :, 0] - 0.5) + centers[:, 0:1]
        kpt_decoded[:, :, 1] = stride * (kpt_decoded[:, :, 1] - 0.5) + centers[:, 1:2]
        # Visibility bleibt als Logit (sigmoid spaeter)

        all_boxes.append(np.stack([cx, cy, w, h], axis=1))
        all_scores.append(score_sig)
        all_kpts.append(kpt_decoded)

    boxes_xywh = np.concatenate(all_boxes)   # (total, 4)
    scores = np.concatenate(all_scores)      # (total,)
    kpts = np.concatenate(all_kpts)          # (total, 17, 3)

    # Confidence filter
    mask = scores > conf_thresh
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    kpts = kpts[mask]

    # Top-k
    if len(scores) > 300:
        topk = np.argsort(scores)[::-1][:300]
        boxes_xywh, scores, kpts = boxes_xywh[topk], scores[topk], kpts[topk]

    # xywh -> xyxy fuer NMS
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # NMS
    keep = _nms(boxes_xyxy, scores, iou_thresh)
    keep = keep[:max_det]

    results = []
    for idx in keep:
        kp = kpts[idx].copy()  # (17, 3)
        kp[:, 2] = 1.0 / (1.0 + np.exp(-kp[:, 2]))  # sigmoid visibility
        results.append({
            "bbox": boxes_xyxy[idx].tolist(),  # xyxy in model pixels (640x640)
            "score": float(scores[idx]),
            "keypoints": kp,  # (17, 3) x, y in model pixels, vis [0,1]
        })
    return results


# ============================================================
# ArcFace Face Embedding
# ============================================================

def normalize_arcface(embedding: np.ndarray) -> np.ndarray:
    """L2-Normalize ArcFace 512-dim embedding."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def match_face(embedding: np.ndarray, face_db: Dict[str, np.ndarray],
               threshold: float = 0.6) -> Tuple[str, float]:
    """Match embedding gegen Face-DB via Cosine-Similarity.

    Returns: (name, similarity) oder ("Unbekannt", 0.0)
    """
    best_name = "Unbekannt"
    best_sim = 0.0

    emb_norm = normalize_arcface(embedding)
    for name, ref_emb in face_db.items():
        sim = float(np.dot(emb_norm, ref_emb))
        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_sim >= threshold:
        return best_name, best_sim
    return "Unbekannt", best_sim


# ============================================================
# NMS (shared)
# ============================================================

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Standard IoU-basierte Non-Maximum Suppression.

    Args:
        boxes: (N, 4) xyxy format
        scores: (N,)
        iou_thresh: overlap threshold

    Returns: indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


# ============================================================
# Overlay Drawing (OpenCV, BGR)
# ============================================================

COLOR_FACE = (0, 255, 0)       # Gruen
COLOR_PERSON = (255, 120, 0)   # Blau (BGR)
COLOR_POSE_BOX = (0, 165, 255) # Orange (BGR)
COLOR_KEYPOINT = (255, 0, 255) # Magenta
COLOR_SKELETON = (0, 255, 255) # Gelb (BGR)
COLOR_NAME = (0, 255, 0)       # Gruen


def draw_faces(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
               landmarks: np.ndarray, scale_x: float, scale_y: float):
    """Zeichne Face-Boxen + Landmarks auf Frame (BGR)."""
    h, w = frame.shape[:2]
    for i in range(len(boxes)):
        x1 = int(boxes[i, 0] * w)
        y1 = int(boxes[i, 1] * h)
        x2 = int(boxes[i, 2] * w)
        y2 = int(boxes[i, 3] * h)
        conf = scores[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FACE, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE, 1)
        # 5 Landmarks
        lm = landmarks[i]
        for p in range(5):
            lx = int(lm[p * 2] * w)
            ly = int(lm[p * 2 + 1] * h)
            cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)


def draw_name(frame: np.ndarray, box: np.ndarray, name: str,
              similarity: float, h: int, w: int):
    """Zeichne Namen unter Face-Box."""
    x1 = int(box[0] * w)
    y2 = int(box[3] * h)
    label = f"{name} ({similarity:.0%})" if name != "Unbekannt" else "Unbekannt"
    color = COLOR_NAME if name != "Unbekannt" else (0, 0, 255)
    cv2.putText(frame, label, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_persons(frame: np.ndarray, detections: List[Dict],
                 scale_x: float, scale_y: float):
    """Zeichne Person-Boxen (normalized coords)."""
    h, w = frame.shape[:2]
    for det in detections:
        bx = det["bbox"]
        x1 = int(bx[0] * w)
        y1 = int(bx[1] * h)
        x2 = int(bx[2] * w)
        y2 = int(bx[3] * h)
        conf = det["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 1)


def draw_poses(frame: np.ndarray, poses: List[Dict],
               scale_x: float, scale_y: float, joint_thresh: float = 0.3):
    """Zeichne Pose-Boxen + Skeleton + Keypoints."""
    h, w = frame.shape[:2]
    for pose in poses:
        # Box (in model pixels 640x640 -> frame pixels)
        bx = pose["bbox"]
        x1 = int(bx[0] * scale_x)
        y1 = int(bx[1] * scale_y)
        x2 = int(bx[2] * scale_x)
        y2 = int(bx[3] * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_POSE_BOX, 2)
        cv2.putText(frame, f"Pose {pose['score']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_POSE_BOX, 1)

        # Keypoints
        kpts = pose["keypoints"]  # (17, 3) in model pixels
        pts = []
        for ki in range(17):
            kx = int(kpts[ki, 0] * scale_x)
            ky = int(kpts[ki, 1] * scale_y)
            vis = kpts[ki, 2]
            pts.append((kx, ky, vis))
            if vis > joint_thresh:
                cv2.circle(frame, (kx, ky), 4, COLOR_KEYPOINT, -1)

        # Skeleton
        for j0, j1 in SKELETON_PAIRS:
            if pts[j0][2] > joint_thresh and pts[j1][2] > joint_thresh:
                cv2.line(frame, (pts[j0][0], pts[j0][1]),
                         (pts[j1][0], pts[j1][1]), COLOR_SKELETON, 2)
