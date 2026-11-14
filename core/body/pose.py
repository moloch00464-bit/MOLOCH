"""Zielpose des Koerpers, Grenzen und Mapping-Konfiguration."""

import json
import math
import os
from dataclasses import dataclass, fields, replace
from typing import Dict

MAPPINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "body_mappings.json",
)

# Achsen, die geklemmt und fuer Geschwindigkeit/Deadband verglichen werden
AXES = ("head_pitch", "head_yaw", "head_roll", "head_z", "body_yaw", "antenna_l", "antenna_r")


@dataclass(frozen=True)
class BodyPose:
    """Zielpose. Winkel in Grad, head_z in mm, duration_s = Dauer der Bewegung.

    Vorzeichen (eigene Konvention des Koerpers, NICHT die der Sonoff-Kamera):
    head_pitch + = Kopf hoch, head_yaw + = nach links, body_yaw + = nach links.
    """
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    head_z: float = 0.0
    body_yaw: float = 0.0
    antenna_l: float = 0.0
    antenna_r: float = 0.0
    duration_s: float = 0.5

    def max_delta(self, other: "BodyPose") -> float:
        """Groesste Achsabweichung zu einer anderen Pose."""
        return max(abs(getattr(self, a) - getattr(other, a)) for a in AXES)


NEUTRAL = BodyPose()


def load_mappings(path: str = MAPPINGS_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_pose(pose: BodyPose, limits: Dict) -> BodyPose:
    """Jede Achse auf ihre Grenze kuerzen. Nicht-endliche Werte werden 0.

    Fehlt fuer eine Achse die Grenze, wird sie auf 0 gesetzt statt
    ungeprueft durchgereicht - lieber keine Bewegung als eine unbegrenzte.
    """
    values = {}
    for axis in AXES:
        val = float(getattr(pose, axis))
        lim = limits.get(axis)
        if not math.isfinite(val) or not lim or len(lim) != 2 or lim[0] > lim[1]:
            values[axis] = 0.0
        else:
            values[axis] = max(float(lim[0]), min(float(lim[1]), val))
    dur = float(pose.duration_s)
    if not math.isfinite(dur) or dur < 0:
        dur = 0.0
    return replace(pose, duration_s=dur, **values)


def pose_to_dict(pose: BodyPose) -> Dict[str, float]:
    return {f.name: round(getattr(pose, f.name), 2) for f in fields(pose)}
