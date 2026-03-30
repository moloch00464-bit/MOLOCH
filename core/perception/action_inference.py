#!/usr/bin/env python3
"""
M.O.L.O.C.H. Action Inference
================================
Erkennt menschliche Aktionen aus einem Ringpuffer von Pose-Keypoints.
Kein dediziertes ML-Modell noetig — Threshold-basierte Regeln auf 17 COCO-Keypoints.

COCO Keypoint-Index:
    0=Nase, 1=AugeL, 2=AugeR, 3=OhrL, 4=OhrR
    5=SchulterL, 6=SchulterR, 7=EllbogenL, 8=EllbogenR
    9=HandgelenkL, 10=HandgelenkR
    11=HuefteL, 12=HuefteR, 13=KnieL, 14=KnieR
    15=KnoechelL, 16=KnoechelR

Verwendung:
    infer = get_action_inferrer()
    action = infer.update(keypoints)   # -> "stehend" | "gehend" | etc.
"""

import logging
import time
from collections import deque
from typing import Optional, List, Tuple

logger = logging.getLogger("ActionInference")

# COCO Keypoint-Indizes
SCHULTER_L, SCHULTER_R = 5, 6
ELLBOGEN_L, ELLBOGEN_R = 7, 8
HANDGELENK_L, HANDGELENK_R = 9, 10
HUEFTE_L, HUEFTE_R = 11, 12
KNIE_L, KNIE_R = 13, 14
KNOECHEL_L, KNOECHEL_R = 15, 16

# Mindest-Konfidenz damit ein Keypoint als gueltig gilt
MIN_CONF = 0.4

# Puffer-Groesse: 30 Frames bei 20 FPS = 1.5 Sekunden
BUFFER_SIZE = 30


def _kp(keypoints: List[Tuple], idx: int) -> Optional[Tuple]:
    """Gibt Keypoint zurueck wenn Konfidenz ausreichend, sonst None."""
    if idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    # kp kann (x, y) oder (x, y, conf) sein
    if len(kp) >= 3 and kp[2] < MIN_CONF:
        return None
    return kp


def _y(kp: Optional[Tuple]) -> Optional[float]:
    """Y-Koordinate eines Keypoints (None wenn ungueltig)."""
    return kp[1] if kp is not None else None


def _x(kp: Optional[Tuple]) -> Optional[float]:
    """X-Koordinate eines Keypoints (None wenn ungueltig)."""
    return kp[0] if kp is not None else None


class ActionInferrer:
    """Erkennt Aktionen aus einem Ringpuffer von Pose-Frames."""

    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._last_action: Optional[str] = None
        self._action_stable_count: int = 0
        # Mindest-Frames fuer stabile Erkennung
        self._stability_threshold = 5

    def update(self, keypoints: Optional[List[Tuple]]) -> Optional[str]:
        """Neuen Frame einpflegen und aktuelle Aktion ableiten.

        Args:
            keypoints: Liste von 17 COCO-Keypoints (x, y, conf) oder None

        Returns:
            Aktions-Label oder None
        """
        if keypoints is None or len(keypoints) < 13:
            return self._last_action

        self._buffer.append(keypoints)

        if len(self._buffer) < 8:
            # Zu wenig Frames fuer sichere Erkennung
            return None

        action = self._classify()

        # Stabilitaets-Filter: Aktion muss mehrfach aufeinander folgen
        if action == self._last_action:
            self._action_stable_count += 1
        else:
            self._action_stable_count = 1
            self._last_action = action

        if self._action_stable_count >= self._stability_threshold:
            return self._last_action

        return None

    def _classify(self) -> Optional[str]:
        """Klassifiziert Aktion aus aktuellem Puffer-Inhalt."""
        frames = list(self._buffer)
        current = frames[-1]

        # --- Koerpergroesse schaetzen (Schulter zu Knie) ---
        body_height = self._estimate_body_height(current)

        # --- Zeigen erkennen (Priorisiert wegen MOLOCH-Resonanz) ---
        if self._is_pointing(current, body_height):
            return "zeigend"

        # --- Winken erkennen ---
        if self._is_waving(frames, body_height):
            return "winkend"

        # --- Sitzen erkennen ---
        if self._is_sitting(current, body_height):
            return "sitzend"

        # --- Gehen erkennen ---
        if self._is_walking(frames):
            return "gehend"

        # --- Stehen als Default ---
        if self._is_standing(current, body_height):
            return "stehend"

        return None

    def _estimate_body_height(self, kps: List[Tuple]) -> float:
        """Koerpergroesse in Frame-Anteilen (Schulter bis Knoechel)."""
        schulter = _kp(kps, SCHULTER_L) or _kp(kps, SCHULTER_R)
        knoechel = _kp(kps, KNOECHEL_L) or _kp(kps, KNOECHEL_R)
        if schulter and knoechel:
            return abs(_y(knoechel) - _y(schulter))
        # Fallback: Schulter zu Huefte * 2
        huefte = _kp(kps, HUEFTE_L) or _kp(kps, HUEFTE_R)
        if schulter and huefte:
            return abs(_y(huefte) - _y(schulter)) * 2.2
        return 0.4  # Standardwert

    def _is_pointing(self, kps: List[Tuple], body_h: float) -> bool:
        """Zeige-Geste: Ein Arm gestreckt, Handgelenk deutlich vor/ueber Ellbogen."""
        for hand_idx, elbow_idx, shoulder_idx in [
            (HANDGELENK_L, ELLBOGEN_L, SCHULTER_L),
            (HANDGELENK_R, ELLBOGEN_R, SCHULTER_R),
        ]:
            hand = _kp(kps, hand_idx)
            elbow = _kp(kps, elbow_idx)
            shoulder = _kp(kps, shoulder_idx)
            if hand and elbow and shoulder:
                # Arm gestreckt wenn Hand-Ellbogen-Abstand > Ellbogen-Schulter-Abstand * 0.9
                hand_elbow = abs(_y(hand) - _y(elbow)) + abs(_x(hand) - _x(elbow))
                elbow_shoulder = abs(_y(elbow) - _y(shoulder)) + abs(_x(elbow) - _x(shoulder))
                if hand_elbow > elbow_shoulder * 0.85 and hand_elbow > body_h * 0.25:
                    return True
        return False

    def _is_waving(self, frames: List[List[Tuple]], body_h: float) -> bool:
        """Winken: Handgelenk ueber Schulter + horizontale Bewegung > 15% Frame-Breite."""
        if len(frames) < 10:
            return False

        wave_frames = 0
        for kps in frames[-15:]:
            for hand_idx, shoulder_idx in [(HANDGELENK_L, SCHULTER_L), (HANDGELENK_R, SCHULTER_R)]:
                hand = _kp(kps, hand_idx)
                shoulder = _kp(kps, shoulder_idx)
                if hand and shoulder and _y(hand) < _y(shoulder):
                    wave_frames += 1
                    break

        if wave_frames < 8:
            return False

        # Horizontale Bewegung des Handgelenks pruefen
        for hand_idx in [HANDGELENK_L, HANDGELENK_R]:
            xs = []
            for kps in frames[-15:]:
                h = _kp(kps, hand_idx)
                if h:
                    xs.append(_x(h))
            if len(xs) >= 8:
                x_range = max(xs) - min(xs)
                if x_range > 0.12:  # >12% Frame-Breite Bewegung
                    return True
        return False

    def _is_sitting(self, kps: List[Tuple], body_h: float) -> bool:
        """Sitzen: Huefte-y nahe Knie-y (< 35% Koerpergroesse Abstand)."""
        huefte = _kp(kps, HUEFTE_L) or _kp(kps, HUEFTE_R)
        knie = _kp(kps, KNIE_L) or _kp(kps, KNIE_R)
        if huefte and knie and body_h > 0:
            huefte_knie_dist = abs(_y(knie) - _y(huefte))
            return huefte_knie_dist < body_h * 0.3
        return False

    def _is_walking(self, frames: List[List[Tuple]]) -> bool:
        """Gehen: Knie wechseln alternierend links/rechts ueber mehrere Frames."""
        if len(frames) < 10:
            return False

        alternations = 0
        last_leading = None

        for kps in frames[-20:]:
            knie_l = _kp(kps, KNIE_L)
            knie_r = _kp(kps, KNIE_R)
            if knie_l and knie_r:
                # Welches Knie ist vorne (niedrigeres y = hoeher im Bild = vorne beim Gehen)
                leading = "L" if _y(knie_l) < _y(knie_r) else "R"
                if last_leading and leading != last_leading:
                    alternations += 1
                last_leading = leading

        return alternations >= 3

    def _is_standing(self, kps: List[Tuple], body_h: float) -> bool:
        """Stehen: Schulter deutlich ueber Huefte, Huefte ueber Knie."""
        schulter = _kp(kps, SCHULTER_L) or _kp(kps, SCHULTER_R)
        huefte = _kp(kps, HUEFTE_L) or _kp(kps, HUEFTE_R)
        knie = _kp(kps, KNIE_L) or _kp(kps, KNIE_R)

        if schulter and huefte and knie:
            # Erwartete vertikale Anordnung: Schulter (kl. y) < Huefte < Knie (gr. y)
            return _y(schulter) < _y(huefte) < _y(knie)
        return False

    def reset(self):
        """Puffer leeren (z.B. wenn Person aus Frame verschwindet)."""
        self._buffer.clear()
        self._last_action = None
        self._action_stable_count = 0


# --- Singleton ---
_inferrer: Optional[ActionInferrer] = None


def get_action_inferrer() -> ActionInferrer:
    """Singleton ActionInferrer."""
    global _inferrer
    if _inferrer is None:
        _inferrer = ActionInferrer()
    return _inferrer
