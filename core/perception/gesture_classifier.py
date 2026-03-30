#!/usr/bin/env python3
"""
M.O.L.O.C.H. Gesture Classifier
=================================
Klassifiziert Hand-Gesten aus 21 COCO-Keypoints (hand_landmark_lite.hef Output).

Keypoint-Index (MediaPipe Hand):
    0=Handwurzel, 1-4=Daumen (Spitze=4), 5-8=Zeigefinger (Spitze=8),
    9-12=Mittelfinger (Spitze=12), 13-16=Ringfinger (Spitze=16),
    17-20=Kleiner Finger (Spitze=20)

Verwendung:
    from core.perception.gesture_classifier import classify_hand_gesture
    gesture = classify_hand_gesture(keypoints)  # keypoints: list[tuple(x, y, z)]

Hinweis: Hand-Valve ist aktuell wegen cv2::resize Crash deaktiviert (bekannter Bug).
Diese Klasse ist fertig implementiert und wird aktiv sobald der Pipeline-Fix erfolgt.
"""

import math
from typing import Optional, List, Tuple


# Keypoint-Indizes (MediaPipe Hand Landmark)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

THUMB_MCP = 2
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

THUMB_PIP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

INDEX_DIP = 7
MIDDLE_DIP = 11
RING_DIP = 15
PINKY_DIP = 19


def _dist(a: Tuple, b: Tuple) -> float:
    """Euklidischer Abstand zwischen zwei Keypoints (x, y)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _is_finger_extended(kp: List[Tuple], tip: int, pip: int, mcp: int) -> bool:
    """Prueft ob ein Finger gestreckt ist (Spitze weiter von Handwurzel als PIP)."""
    return _dist(kp[tip], kp[WRIST]) > _dist(kp[pip], kp[WRIST])


def _is_thumb_extended(kp: List[Tuple]) -> bool:
    """Daumen-Streckung: Spitze weiter vom MCP als IP-Gelenk."""
    return _dist(kp[THUMB_TIP], kp[THUMB_MCP]) > _dist(kp[THUMB_PIP], kp[THUMB_MCP])


def classify_hand_gesture(keypoints: List[Tuple]) -> Optional[str]:
    """Klassifiziert eine Handgeste aus 21 Keypoints.

    Args:
        keypoints: Liste von 21 (x, y, z) oder (x, y) Tupeln (normalisiert 0.0-1.0)

    Returns:
        Gesten-Label: "thumbs_up" | "open_hand" | "point" | "fist" | "wave" | None
    """
    if not keypoints or len(keypoints) < 21:
        return None

    kp = keypoints  # Alias

    # Finger-Streckungsstatus
    thumb_ext = _is_thumb_extended(kp)
    index_ext = _is_finger_extended(kp, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_ext = _is_finger_extended(kp, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_ext = _is_finger_extended(kp, RING_TIP, RING_PIP, RING_MCP)
    pinky_ext = _is_finger_extended(kp, PINKY_TIP, PINKY_PIP, PINKY_MCP)

    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # --- Gesten-Erkennung ---

    # Daumen hoch: Daumen gestreckt, alle anderen gebeugt, Handwurzel unten
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        # Daumen-Spitze muss hoeher als Handwurzel sein (kleineres y = hoeher)
        if kp[THUMB_TIP][1] < kp[WRIST][1]:
            return "thumbs_up"

    # Offene Hand (Stopp): alle 4 Finger gestreckt
    if extended_count >= 4:
        return "open_hand"

    # Zeigefinger (Point): nur Zeigefinger gestreckt
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "point"

    # Peace/V: Zeige- und Mittelfinger gestreckt, Rest gebeugt
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"

    # Faust: alle Finger gebeugt
    if extended_count == 0 and not thumb_ext:
        return "fist"

    return None


def keypoints_from_hailo_landmarks(landmarks_obj) -> Optional[List[Tuple]]:
    """Extrahiert Keypoints aus einem Hailo HAILO_LANDMARKS Objekt.

    Args:
        landmarks_obj: hailo.HailoLandmarks Objekt aus der GStreamer Pipeline

    Returns:
        Liste von (x, y, z) Tupeln oder None bei Fehler
    """
    try:
        points = landmarks_obj.get_points()
        return [(p.x(), p.y(), p.z() if hasattr(p, 'z') else 0.0) for p in points]
    except Exception:
        return None
