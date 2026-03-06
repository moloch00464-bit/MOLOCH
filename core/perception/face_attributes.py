#!/usr/bin/env python3
"""
FaceAttributes — Parser fuer face_attr_resnet_v1_18 Output (80 Werte, CelebA).

Wird von TappasPipeline._on_buffer aufgerufen um gender/smiling aus dem
Raw-Tensor-Output des 4. hailonet-Elements zu parsen.

CelebA 40-Attribut Layout (je 2 Ausgaenge: negativ/positiv):
  Index 20 = Male   (Positionen 40/41 im 80er Vektor)
  Index 31 = Smiling (Positionen 62/63 im 80er Vektor)
"""

import numpy as np

# CelebA Attribut-Indizes im 80er Output (je 2: negativ, positiv)
ATTR_MALE_NEG = 40
ATTR_MALE_POS = 41
ATTR_SMILING_NEG = 62
ATTR_SMILING_POS = 63


def _softmax_pair(neg_val: float, pos_val: float) -> float:
    """Softmax ueber ein Attribut-Paar, gibt Wahrscheinlichkeit fuer positiv zurueck."""
    vals = np.array([neg_val, pos_val], dtype=np.float32)
    e = np.exp(vals - np.max(vals))
    return float(e[1] / e.sum())


def parse_face_attributes(raw_output: np.ndarray) -> dict:
    """80er Output-Vektor in gender + smiling parsen.

    Args:
        raw_output: np.ndarray mit 80 Werten (float32 oder uint8)

    Returns:
        {"gender": "M"|"F", "smiling": True|False,
         "male_prob": float, "smiling_prob": float}
    """
    raw = raw_output.flatten().astype(np.float32)
    if len(raw) < 64:
        return {"gender": None, "smiling": False, "male_prob": 0.0, "smiling_prob": 0.0}

    male_prob = _softmax_pair(raw[ATTR_MALE_NEG], raw[ATTR_MALE_POS])
    smiling_prob = _softmax_pair(raw[ATTR_SMILING_NEG], raw[ATTR_SMILING_POS])

    return {
        "gender": "M" if male_prob > 0.5 else "F",
        "smiling": smiling_prob > 0.5,
        "male_prob": round(male_prob, 3),
        "smiling_prob": round(smiling_prob, 3),
    }
