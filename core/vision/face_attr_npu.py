#!/usr/bin/env python3
"""
M.O.L.O.C.H. Face Attribute NPU Detector
==========================================
face_attr_resnet_v1_18.hef auf Hailo-10H NPU.
CelebA 40 Attribute -> Gender/Age/Emotion Ableitung.
~2926 FPS, ~7MB NPU-RAM.
"""
import numpy as np
from typing import Dict, Tuple

# CelebA 40 Attribute (Index -> Name)
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young"
]

# Model Input-Groesse
FACE_ATTR_W = 178
FACE_ATTR_H = 218


def parse_face_attributes(raw_output: np.ndarray) -> Dict[str, float]:
    """Parse NPU Output in Attribut-Scores.

    Unterstuetzt:
    - 80 Werte: 40 x 2 (neg/pos Paar pro Attribut)
    - 40 Werte: Direkte Scores pro Attribut
    """
    values = raw_output.flatten().astype(np.float32)
    attrs = {}

    if len(values) == 80:
        # 40 Attribute x 2 (neg/pos) -> Differenz als Score
        for i, name in enumerate(CELEBA_ATTRS):
            neg = values[2 * i]
            pos = values[2 * i + 1]
            attrs[name] = float(pos - neg)
    elif len(values) >= 40:
        for i, name in enumerate(CELEBA_ATTRS):
            attrs[name] = float(values[i])
    else:
        for i in range(len(values)):
            if i < len(CELEBA_ATTRS):
                attrs[CELEBA_ATTRS[i]] = float(values[i])

    return attrs


def derive_gender(attrs: Dict[str, float]) -> str:
    """Gender aus Male-Attribut: M oder F."""
    return "M" if attrs.get("Male", 0) > 0 else "F"


def derive_age_range(attrs: Dict[str, float]) -> str:
    """Grobe Altersschaetzung aus CelebA Attributen."""
    young = attrs.get("Young", 0) > 0
    gray_hair = attrs.get("Gray_Hair", 0) > 0
    bald = attrs.get("Bald", 0) > 0
    receding = attrs.get("Receding_Hairline", 0) > 0
    bags = attrs.get("Bags_Under_Eyes", 0) > 0
    double_chin = attrs.get("Double_Chin", 0) > 0

    aging_signs = sum([gray_hair, bald, receding, bags, double_chin])

    if not young and aging_signs >= 2:
        return "60+"
    elif not young and aging_signs >= 1:
        return "48-53"
    elif not young:
        return "38-43"
    elif aging_signs >= 1:
        return "25-32"
    else:
        return "15-20"


def derive_emotion(attrs: Dict[str, float]) -> str:
    """Emotion-Schaetzung aus CelebA Attributen."""
    smiling = attrs.get("Smiling", 0) > 0
    mouth_open = attrs.get("Mouth_Slightly_Open", 0) > 0
    narrow_eyes = attrs.get("Narrow_Eyes", 0) > 0

    if smiling:
        return "Happy"
    elif mouth_open and not smiling:
        return "Surprised"
    elif narrow_eyes:
        return "Angry"
    else:
        return "Neutral"


def analyze_face(raw_output: np.ndarray) -> Tuple[str, str, str]:
    """Vollanalyse: (gender, age_range, emotion) aus NPU Output."""
    attrs = parse_face_attributes(raw_output)
    return derive_gender(attrs), derive_age_range(attrs), derive_emotion(attrs)
