#!/usr/bin/env python3
"""face_attr_parser — Formatiert FaceWorker-Output als lesbaren String."""

from typing import Optional

_EMOTION_MAP = {
    "happy": "froehllich",
    "neutral": "neutral",
    "sad": "traurig",
    "angry": "erregt",
    "fearful": "aengstlich",
    "disgusted": "abgeneigt",
    "surprised": "ueberrascht",
}
_GENDER_MAP = {"M": "m", "F": "w"}


def format_face_attr(gender: Optional[str], age_range: Optional[str], emotion: Optional[str]) -> str:
    """Formatiert FaceWorker-Attribute als Prompt-lesbaren String.

    Returns: "m, ca.35-40, neutral" oder "?" bei fehlenden Daten.
    """
    if not any([gender, age_range, emotion]):
        return "?"
    parts = []
    if gender:
        parts.append(_GENDER_MAP.get(gender, gender.lower()))
    if age_range:
        parts.append(f"ca.{age_range}")
    if emotion:
        parts.append(_EMOTION_MAP.get(emotion, emotion))
    return ", ".join(parts) if parts else "?"


def format_face_attr_from_dict(face: dict) -> str:
    """Liest gender/age_range/emotion direkt aus face-Detection-Dict."""
    return format_face_attr(
        gender=face.get("gender"),
        age_range=face.get("age_range"),
        emotion=face.get("emotion"),
    )
