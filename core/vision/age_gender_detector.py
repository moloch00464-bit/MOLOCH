#!/usr/bin/env python3
"""
M.O.L.O.C.H. Age & Gender Detector
====================================
Caffe GoogLeNet Modelle auf CPU (kein NPU) via cv2.dnn.
Input: Face-Crop (BGR, beliebige Groesse)
Output: (gender, age_range, confidence)
"""
import os
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.expanduser("~/moloch/models/age_gender")
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# Preprocessing mean (ImageNet-style)
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDERS = ["M", "F"]


class AgeGenderDetector:
    """Age and Gender Detection via Caffe GoogLeNet on CPU."""

    def __init__(self):
        self.available = False
        self.age_net = None
        self.gender_net = None

        for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]:
            if not os.path.exists(path):
                logger.warning(f"[AGE/GENDER] Model not found: {path}")
                return

        try:
            self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
            self.gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
            self.available = True
            logger.info("[AGE/GENDER] Caffe models loaded (age + gender)")
        except Exception as e:
            logger.error(f"[AGE/GENDER] Failed to load: {e}")

    def detect(self, face_crop_bgr: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """Detect age and gender from face crop.

        Args:
            face_crop_bgr: BGR face crop (any size)

        Returns:
            (gender, age_range, confidence) or (None, None, 0.0)
            gender: "M" or "F"
            age_range: "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"
        """
        if not self.available or face_crop_bgr is None:
            return None, None, 0.0

        try:
            blob = cv2.dnn.blobFromImage(
                face_crop_bgr, 1.0, (227, 227), MEAN_VALUES, swapRB=False)

            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()[0]
            gender_idx = gender_preds.argmax()
            gender = GENDERS[gender_idx]
            gender_conf = float(gender_preds[gender_idx])

            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()[0]
            age_idx = age_preds.argmax()
            age_range = AGE_BUCKETS[age_idx]
            age_conf = float(age_preds[age_idx])

            conf = min(gender_conf, age_conf)
            return gender, age_range, conf

        except Exception as e:
            logger.debug(f"[AGE/GENDER] Detection failed: {e}")
            return None, None, 0.0


# Singleton
_detector: Optional[AgeGenderDetector] = None


def get_age_gender_detector() -> AgeGenderDetector:
    """Get AgeGenderDetector singleton."""
    global _detector
    if _detector is None:
        _detector = AgeGenderDetector()
    return _detector
