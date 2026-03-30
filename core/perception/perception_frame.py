#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Frame
==============================
Zentraler Wahrnehmungsframe pro Inference-Tick.

Alle Modell-Outputs fliessen hier zusammen. Core Integrator und andere
Consumer lesen NUR diesen Frame — kein direkter Zugriff auf einzelne Modelle.

Ein PerceptionFrame entsteht pro Inference-Durchlauf (~15-30x/s).
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class PerceptionFrame:
    """Ein einzelner Wahrnehmungsframe — Aggregat aller Modell-Outputs."""

    # Zeitstempel (monoton, time.time())
    timestamp: float = 0.0

    # === Person Detection (YOLOv8m) ===
    person_detected: bool = False
    person_count: int = 0
    # Distanz-Schaetzung aus BBox-Groesse (Anteil am Frame)
    distance: str = "none"  # "close" | "medium" | "far" | "none"
    distance_ratio: float = 0.0  # BBox-Flaeche / Frame-Flaeche (0.0-1.0)

    # === Face Detection (SCRFD) ===
    face_detected: bool = False
    face_count: int = 0
    face_confidence: float = 0.0  # Beste Face-Confidence (0.0-1.0)
    face_bbox: Optional[tuple] = None  # (x1, y1, x2, y2) normalisiert

    # === Face Recognition (ArcFace) ===
    face_id: Optional[str] = None  # "markus" | "unknown" | None
    face_similarity: float = 0.0  # Cosine-Similarity (0.0-1.0)

    # === Face Attributes (NPU ResNet) ===
    gender: Optional[str] = None  # "M" | "F" | None
    age_range: Optional[str] = None  # "38-43" | None
    emotion: Optional[str] = None  # "neutral" | "happy" | "angry" | None

    # === Pose Estimation (YOLOv8s Pose) ===
    pose_count: int = 0
    pose_energy: float = 0.0  # Berechnet aus Keypoint-Bewegung (0.0-1.0)

    # === Hand/Gesture Detection ===
    hand_detected: bool = False
    hand_gesture: Optional[str] = None  # "thumbs_up" | "wave" | None

    # === Action Inference (Temporal Pose Buffer) ===
    person_action: Optional[str] = None  # "stehend" | "gehend" | "sitzend" | "winkend" | "zeigend" | None

    # === Head Pose ===
    head_pitch: Optional[float] = None  # Kopfneigung vertikal
    head_yaw: Optional[float] = None  # Kopfdrehung horizontal

    # === Object Detection (YOLOv8m Nicht-Personen) ===
    objects: List[Dict] = field(default_factory=list)
    # [{"class": "couch", "confidence": 0.87}, ...]

    # === OCR (PaddleOCR — Text im Raum) ===
    ocr_texts: List[str] = field(default_factory=list)  # Erkannte Texte

    # === Perception Router ===
    person_bbox_height: float = 0.0  # Groesste Person-BBox Hoehe (0.0-1.0)
    scenario: str = "IDLE"  # Aktuelles Szenario (IDLE/FERN/MITTEL/NAH/RUECKEN/MULTI/NACHT)

    # === Meta ===
    inference_ms: float = 0.0  # Gesamte Inference-Zeit in ms
    active_models: List[str] = field(default_factory=list)

    # === Abgeleitete Flags (Convenience) ===
    @property
    def markus_recognized(self) -> bool:
        return self.face_id is not None and self.face_id.lower() == "markus"

    @property
    def unknown_face(self) -> bool:
        return self.face_detected and (self.face_id is None or self.face_id.lower() in ("unknown", "unbekannt"))

    @property
    def anyone_present(self) -> bool:
        return self.person_detected or self.face_detected

    def to_dict(self) -> Dict:
        """Kompaktes Dict fuer JSON-Export / IPC."""
        return {
            "timestamp": self.timestamp,
            "person_detected": self.person_detected,
            "person_count": self.person_count,
            "distance": self.distance,
            "face_detected": self.face_detected,
            "face_count": self.face_count,
            "face_id": self.face_id,
            "face_confidence": round(self.face_confidence, 3),
            "face_similarity": round(self.face_similarity, 3),
            "gender": self.gender,
            "age_range": self.age_range,
            "emotion": self.emotion,
            "pose_count": self.pose_count,
            "pose_energy": round(self.pose_energy, 3),
            "hand_detected": self.hand_detected,
            "hand_gesture": self.hand_gesture,
            "person_action": self.person_action,
            "head_pitch": round(self.head_pitch, 1) if self.head_pitch is not None else None,
            "head_yaw": round(self.head_yaw, 1) if self.head_yaw is not None else None,
            "objects": self.objects,
            "ocr_texts": self.ocr_texts,
            "person_bbox_height": round(self.person_bbox_height, 3),
            "scenario": self.scenario,
            "active_models": self.active_models,
            "inference_ms": round(self.inference_ms, 1),
        }


def estimate_distance(bbox_area_ratio: float) -> str:
    """Distanz aus BBox-Flaechenanteil schaetzen.

    Args:
        bbox_area_ratio: BBox-Flaeche / Frame-Flaeche (0.0-1.0)

    Returns:
        "close" (>15%), "medium" (5-15%), "far" (<5%), "none" (0)
    """
    if bbox_area_ratio <= 0:
        return "none"
    if bbox_area_ratio > 0.15:
        return "close"
    if bbox_area_ratio > 0.05:
        return "medium"
    return "far"
