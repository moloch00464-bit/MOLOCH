"""Gefuehl -> Zielpose. Reine Funktion: kein I/O, kein Zufall, keine Uhr.

Jede Ebene steuert einen eigenen Koerperkanal (Konzept Abschnitt 3):
    Seins-Zustand -> Grundhaltung      Zone      -> Amplitude / Tempo
    Mood          -> Antennen          Tension   -> Atmung + Unruhe
    Dominance     -> Kopfneigung       Wahrnehmung -> Blickrichtung
    Soma          -> Haltung + Daempfung (technischer Koerperzustand, Issue #5)
Die Pose ist danach noch NICHT geklemmt - das macht der Controller.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .pose import BodyPose

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class BodyState:
    """Momentaufnahme von Molochs Zustand, wie der Koerper sie braucht."""
    being_state: str = "idle"       # state_engine: idle/observing/engaged/...
    zone: str = "guardian"          # guardian/shadow/berserker
    mood: str = "calm"              # mood_engine: calm/focused/alert/...
    tension: float = 0.0            # 0..1
    dominance: float = 0.0          # -1..+1
    gaze_yaw: Optional[float] = None    # Blickziel in Grad, None = kein Ziel
    gaze_pitch: Optional[float] = None
    speaking: bool = False
    soma: str = "ok"                # core/awareness/soma.py: ok/strained/distressed/recovering/unknown


def _num(value, lo: float, hi: float) -> float:
    """Zahl in [lo, hi]; alles Unbrauchbare wird 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(lo, min(hi, v))


def compute_pose(state: BodyState, mappings: Dict, t: float) -> BodyPose:
    """Zielpose fuer den Zustand `state` zum Zeitpunkt `t` (Sekunden, monoton).

    Unbekannte Zustands-/Zonen-/Mood-Namen fallen auf idle/guardian/calm zurueck.
    """
    states, zones, moods = mappings["states"], mappings["zones"], mappings["moods"]
    st = states.get(state.being_state) or states["idle"]
    zn = zones.get(state.zone) or zones["guardian"]
    md = moods.get(state.mood) or moods["calm"]
    mod = mappings["modulation"]
    gz = mappings["gaze"]
    # Soma: fehlt der Abschnitt oder ist der Zustand unbekannt -> kein Einfluss.
    # Ein nicht gemessener Zustand wird nicht dargestellt (UNKNOWN != OK, aber auch != krank).
    sm = (mappings.get("soma") or {}).get(state.soma) or {}

    tension = _num(state.tension, 0.0, 1.0)
    dominance = _num(state.dominance, -1.0, 1.0)
    amp = float(zn.get("amplitude", 1.0)) * float(sm.get("amplitude", 1.0))

    # Grundhaltung + Zone + Dominance (dominant = Kopf hoch)
    pitch = st.get("head_pitch", 0.0) + zn.get("head_pitch", 0.0) + sm.get("head_pitch", 0.0)
    pitch += dominance * mod["dominance_pitch_deg"]

    # Atmung: Frequenz steigt mit Tension
    breath_hz = mod["breath_hz_base"] + tension * mod["breath_hz_per_tension"]
    head_z = st.get("head_z", 0.0) + mod["breath_mm"] * math.sin(TWO_PI * breath_hz * t)

    # Unruhe: quadratisch, erst ab ca. 0.5 Tension spuerbar
    noise = tension * tension * mod["noise_deg_max"] * amp
    roll = noise * math.sin(TWO_PI * 1.7 * t)
    yaw = noise * math.sin(TWO_PI * 1.1 * t + 1.0)
    body_yaw = st.get("body_yaw", 0.0)

    # Blick: eigenes Vorzeichen (gaze.yaw_sign), unabhaengig von der Sonoff
    gaze_scale = st.get("gaze_scale", 1.0)
    if state.gaze_yaw is not None and gaze_scale:
        total = (_num(state.gaze_yaw, -180.0, 180.0) * gz["yaw_sign"] * gz["yaw_scale"]
                 + gz["yaw_offset"]) * gaze_scale
        share = _num(gz["body_yaw_share"], 0.0, 1.0)
        body_yaw += total * share
        yaw += total * (1.0 - share)
    if state.gaze_pitch is not None and gaze_scale:
        pitch += (_num(state.gaze_pitch, -90.0, 90.0) * gz["pitch_scale"]
                  + gz["pitch_offset"]) * gaze_scale

    if state.speaking:
        pitch += 2.0 * math.sin(TWO_PI * 2.0 * t)

    # Antennen: Grundstellung aus Zustand, Mood verschiebt und wippt
    base = st.get("antenna", 0.0) + md.get("antenna_offset", 0.0) + sm.get("antenna_offset", 0.0)
    wiggle = md.get("wiggle_deg", 0.0) * amp
    phase = TWO_PI * md.get("wiggle_hz", 0.0) * t
    asym = st.get("antenna_asym", 0.0)
    antenna_l = base + asym + wiggle * math.sin(phase)
    antenna_r = base - asym + wiggle * math.sin(phase + 2.0)

    speed = max(0.1, float(zn.get("speed", 1.0)))
    return BodyPose(
        head_pitch=pitch, head_yaw=yaw, head_roll=roll, head_z=head_z,
        body_yaw=body_yaw, antenna_l=antenna_l, antenna_r=antenna_r,
        duration_s=0.5 / speed,
    )
