#!/usr/bin/env python3
"""
M.O.L.O.C.H. Emotion Detector
==============================
FER+ ONNX Modell auf CPU (kein NPU).
Input: Face-Crop (BGR, beliebige Groesse)
Output: (emotion_label, confidence)
Labels: Happy, Sad, Angry, Neutral, Surprised
"""
import os
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.expanduser("~/moloch/models/emotion-ferplus-8.onnx")

# FER+ 8 classes
_LABELS_8 = ["Neutral", "Happy", "Surprised", "Sad", "Angry", "Disgust", "Fear", "Contempt"]


class EmotionDetector:
    """Emotion Detection via FER+ ONNX on CPU."""

    def __init__(self, model_path: str = None):
        self.available = False
        self.session = None
        self.input_name = None

        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"[EMOTION] Model not found: {path}")
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.available = True
            logger.info(f"[EMOTION] FER+ loaded ({os.path.getsize(path) // 1024}KB)")
        except Exception as e:
            logger.error(f"[EMOTION] Failed to load: {e}")

    def detect(self, face_crop_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect emotion from face crop.

        Args:
            face_crop_bgr: BGR face crop (any size)

        Returns:
            (emotion_label, confidence) or (None, 0.0)
        """
        if not self.available or face_crop_bgr is None:
            return None, 0.0

        try:
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)).astype(np.float32)
            blob = resized.reshape(1, 1, 64, 64)

            logits = self.session.run(None, {self.input_name: blob})[0][0]

            # Softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()

            # Map 8 -> 5 (merge related emotions)
            mapped = {
                "Happy": float(probs[1]),
                "Sad": float(probs[3]),
                "Angry": float(probs[4] + probs[5]),
                "Neutral": float(probs[0] + probs[7]),
                "Surprised": float(probs[2] + probs[6]),
            }
            best = max(mapped, key=mapped.get)
            return best, mapped[best]

        except Exception as e:
            logger.debug(f"[EMOTION] Detection failed: {e}")
            return None, 0.0


# Singleton
_detector: Optional[EmotionDetector] = None


def get_emotion_detector() -> EmotionDetector:
    """Get EmotionDetector singleton."""
    global _detector
    if _detector is None:
        _detector = EmotionDetector()
    return _detector
