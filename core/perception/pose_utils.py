#!/usr/bin/env python3
"""
pose_utils — Posture-Klassifikator aus 17 COCO-Keypoints.

Nutzt YOLO-Pose-Output (kpts: Liste von 17 (x, y, conf) normalisiert [0,1]).
Liefert kombinierten Posture-String wie "stehend_arme_verschraenkt".

COCO Keypoint-Indizes (0-basiert):
  0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
  5=left_shoulder, 6=right_shoulder
  7=left_elbow, 8=right_elbow
  9=left_wrist, 10=right_wrist
  11=left_hip, 12=right_hip
  13=left_knee, 14=right_knee
  15=left_ankle, 16=right_ankle

Aufruf:
  from core.perception.pose_utils import classify_posture
  posture = classify_posture(kpts_list)  # "stehend_arme_oben" o.ae.
"""

from typing import List, Optional, Tuple

# Keypoint-Indizes
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14

CONF_MIN = 0.3  # Minimum confidence pro Keypoint


def _kp(keypoints, idx: int) -> Optional[Tuple[float, float, float]]:
    """Holt Keypoint mit Confidence-Check. Liefert (x, y, conf) oder None."""
    if keypoints is None or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    if kp is None or len(kp) < 3:
        return None
    try:
        x = float(kp[0]); y = float(kp[1]); c = float(kp[2])
    except (TypeError, ValueError):
        return None
    if c < CONF_MIN:
        return None
    return (x, y, c)


def _avg_y(*pts) -> Optional[float]:
    """Mittelwert der y-Koordinaten aller nicht-None Punkte."""
    ys = [p[1] for p in pts if p is not None]
    return sum(ys) / len(ys) if ys else None


def _avg_x(*pts) -> Optional[float]:
    """Mittelwert der x-Koordinaten aller nicht-None Punkte."""
    xs = [p[0] for p in pts if p is not None]
    return sum(xs) / len(xs) if xs else None


def classify_posture(keypoints) -> str:
    """Klassifiziert Posture aus 17 COCO-Keypoints.

    Args:
        keypoints: Liste von 17 (x, y, conf) Tupeln/Listen, normalisiert [0,1].

    Returns:
        Posture-String z.B. "stehend_arme_verschraenkt" oder "unbekannt".
    """
    if not keypoints:
        return "unbekannt"

    # Sichere Keypoints (mit conf >= CONF_MIN)
    nose = _kp(keypoints, NOSE)
    sh_l = _kp(keypoints, LEFT_SHOULDER)
    sh_r = _kp(keypoints, RIGHT_SHOULDER)
    wr_l = _kp(keypoints, LEFT_WRIST)
    wr_r = _kp(keypoints, RIGHT_WRIST)
    hp_l = _kp(keypoints, LEFT_HIP)
    hp_r = _kp(keypoints, RIGHT_HIP)
    kn_l = _kp(keypoints, LEFT_KNEE)
    kn_r = _kp(keypoints, RIGHT_KNEE)

    valid_count = sum(1 for k in (nose, sh_l, sh_r, wr_l, wr_r,
                                  hp_l, hp_r, kn_l, kn_r) if k is not None)
    if valid_count < 4:
        return "unbekannt"

    parts: List[str] = []

    # 1. sitzend vs. stehend (Knie-Y vs. Hueft-Y)
    hip_y = _avg_y(hp_l, hp_r)
    knee_y = _avg_y(kn_l, kn_r)
    if hip_y is not None and knee_y is not None:
        # Knie deutlich tiefer im Bild (groesserer y) als Hueften -> stehend
        if knee_y - hip_y > 0.1:
            parts.append("stehend")
        else:
            parts.append("sitzend")
    else:
        # Beine nicht sichtbar — defensive Annahme: sitzend
        parts.append("sitzend")

    # 2. arme_verschraenkt (Wrists kreuzen Body-Mittellinie)
    center_x = _avg_x(sh_l, sh_r)
    if center_x is not None and wr_l is not None and wr_r is not None:
        # Linkes Handgelenk auf rechter Bildhaelfte (x > center) UND
        # rechtes Handgelenk auf linker Bildhaelfte (x < center)
        if wr_l[0] > center_x and wr_r[0] < center_x:
            parts.append("arme_verschraenkt")

    # 3. arme_oben (Wrist-Y kleiner als Shoulder-Y)
    sh_y = _avg_y(sh_l, sh_r)
    if sh_y is not None:
        wrist_ys = [w[1] for w in (wr_l, wr_r) if w is not None]
        if wrist_ys and min(wrist_ys) < sh_y - 0.02:
            parts.append("arme_oben")

    # 4. abgewandt (Schultern zu nah zusammen oder nur eine sichtbar)
    if sh_l is None or sh_r is None:
        parts.append("abgewandt")
    else:
        shoulder_dx = abs(sh_l[0] - sh_r[0])
        if shoulder_dx < 0.08:
            parts.append("abgewandt")

    # 5. gestikulierend (Wrists auf sehr unterschiedlicher Hoehe)
    if wr_l is not None and wr_r is not None:
        if abs(wr_l[1] - wr_r[1]) > 0.2:
            parts.append("gestikulierend")

    return "_".join(parts) if parts else "unbekannt"
