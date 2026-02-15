#!/usr/bin/env python3
"""Patch: Age + Gender Detection einbauen (Caffe auf CPU via cv2.dnn)."""
import sys
import os

# =====================================================
# STEP 1: Create age_gender_detector.py
# =====================================================
age_gender_module = '''\
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
'''

age_gender_path = "/home/molochzuhause/moloch/core/vision/age_gender_detector.py"
with open(age_gender_path, "w") as f:
    f.write(age_gender_module)
print(f"STEP 1: Created {age_gender_path}")

# =====================================================
# STEP 2: Patch moloch_service.py
# =====================================================
svc_path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(svc_path, "r") as f:
    code = f.read()

changes = 0

# PATCH 2a: Add age_gender detector init after emotion detector init
old_emotion_init = """        # Emotion Detection (CPU, kein NPU)
        self._emotion_detector = None
        try:
            from core.vision.emotion_detector import get_emotion_detector
            self._emotion_detector = get_emotion_detector()
            if self._emotion_detector and self._emotion_detector.available:
                logger.info("[INIT] Emotion Detection bereit (FER+ CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Emotion Detection nicht verfuegbar: {e}")"""

new_emotion_init = """        # Emotion Detection (CPU, kein NPU)
        self._emotion_detector = None
        try:
            from core.vision.emotion_detector import get_emotion_detector
            self._emotion_detector = get_emotion_detector()
            if self._emotion_detector and self._emotion_detector.available:
                logger.info("[INIT] Emotion Detection bereit (FER+ CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Emotion Detection nicht verfuegbar: {e}")

        # Age + Gender Detection (CPU, kein NPU)
        self._age_gender_detector = None
        try:
            from core.vision.age_gender_detector import get_age_gender_detector
            self._age_gender_detector = get_age_gender_detector()
            if self._age_gender_detector and self._age_gender_detector.available:
                logger.info("[INIT] Age+Gender Detection bereit (Caffe CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Age+Gender Detection nicht verfuegbar: {e}")"""

if old_emotion_init in code:
    code = code.replace(old_emotion_init, new_emotion_init, 1)
    changes += 1
    print("PATCH 2a: Age+Gender detector init added")
else:
    print("ERROR: Could not find emotion detector init block")
    sys.exit(1)

# PATCH 2b: Add age_gender detection after emotion detection + extend draw_name and _write_face_state calls
old_emotion_block = """                            # Emotion Detection (CPU)
                            emotion = None
                            if self._emotion_detector and crop is not None:
                                try:
                                    emotion, _ = self._emotion_detector.detect(crop)
                                except Exception:
                                    pass

                            draw_name(annotated, box, name, sim, fh, fw, emotion=emotion)
                            self._write_face_state(name, sim, len(face_boxes), emotion=emotion)"""

new_emotion_block = """                            # Emotion Detection (CPU)
                            emotion = None
                            if self._emotion_detector and crop is not None:
                                try:
                                    emotion, _ = self._emotion_detector.detect(crop)
                                except Exception:
                                    pass

                            # Age + Gender Detection (CPU)
                            gender, age_range = None, None
                            if self._age_gender_detector and crop is not None:
                                try:
                                    gender, age_range, _ = self._age_gender_detector.detect(crop)
                                except Exception:
                                    pass

                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range)
                            self._write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range)"""

if old_emotion_block in code:
    code = code.replace(old_emotion_block, new_emotion_block, 1)
    changes += 1
    print("PATCH 2b: Age+Gender detection added after emotion")
else:
    print("ERROR: Could not find emotion detection block")
    sys.exit(1)

# PATCH 2c: Extend _write_face_state signature
old_write_state = """    def _write_face_state(self, name, similarity, person_count, emotion=None):
        \"\"\"Schreibe Face-Recognition-State fuer IPC mit push_to_talk.\"\"\"
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "emotion": emotion,
                "timestamp": time.time(),
                "source": "moloch_service"
            }"""

new_write_state = """    def _write_face_state(self, name, similarity, person_count, emotion=None, gender=None, age_range=None):
        \"\"\"Schreibe Face-Recognition-State fuer IPC mit push_to_talk.\"\"\"
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "emotion": emotion,
                "gender": gender,
                "age_range": age_range,
                "timestamp": time.time(),
                "source": "moloch_service"
            }"""

if old_write_state in code:
    code = code.replace(old_write_state, new_write_state, 1)
    changes += 1
    print("PATCH 2c: _write_face_state extended with gender + age_range")
else:
    print("ERROR: Could not find _write_face_state")
    sys.exit(1)

with open(svc_path, "w") as f:
    f.write(code)
print(f"Service patched: {changes} changes")

# =====================================================
# STEP 3: Patch hailo_postprocess.py - draw_name()
# =====================================================
pp_path = "/home/molochzuhause/moloch/core/perception/hailo_postprocess.py"
with open(pp_path, "r") as f:
    pp_code = f.read()

pp_changes = 0

old_draw_name = """def draw_name(frame: np.ndarray, box: np.ndarray, name: str,
              similarity: float, h: int, w: int, emotion: str = None):
    \"\"\"Zeichne Namen + Emotion unter Face-Box.\"\"\"
    x1 = int(box[0] * w)
    y2 = int(box[3] * h)
    label = f"{name} ({similarity:.0%})" if name != "Unbekannt" else "Unbekannt"
    if emotion:
        label += f" [{emotion}]"
    color = COLOR_NAME if name != "Unbekannt" else (0, 0, 255)
    cv2.putText(frame, label, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)"""

new_draw_name = """def draw_name(frame: np.ndarray, box: np.ndarray, name: str,
              similarity: float, h: int, w: int, emotion: str = None,
              gender: str = None, age_range: str = None):
    \"\"\"Zeichne Namen + Emotion + Age/Gender unter Face-Box.\"\"\"
    x1 = int(box[0] * w)
    y2 = int(box[3] * h)
    label = f"{name} ({similarity:.0%})" if name != "Unbekannt" else "Unbekannt"
    if emotion:
        label += f" [{emotion}]"
    if gender and age_range:
        label += f" {gender}/{age_range}"
    color = COLOR_NAME if name != "Unbekannt" else (0, 0, 255)
    cv2.putText(frame, label, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)"""

if old_draw_name in pp_code:
    pp_code = pp_code.replace(old_draw_name, new_draw_name, 1)
    pp_changes += 1
    print("PATCH 3: draw_name() extended with gender + age_range")
else:
    print("ERROR: Could not find draw_name function")
    sys.exit(1)

with open(pp_path, "w") as f:
    f.write(pp_code)
print(f"Postprocess patched: {pp_changes} changes")

# =====================================================
# STEP 4: Patch unified_panel.py - age/gender in chat context
# =====================================================
panel_path = "/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py"
with open(panel_path, "r") as f:
    panel_code = f.read()

panel_changes = 0

old_vision_context = '''                    if name and name not in ("Unbekannt", "Keine DB"):
                        emotion = face_state.get("emotion", "")
                        emo_str = f", Emotion: {emotion}" if emotion else ""
                        message_content = (
                            f"[Vision: Ich sehe {name} ({sim:.0%}){emo_str}]\\n\\n"
                            f"Markus sagt: {user_text}")'''

new_vision_context = '''                    if name and name not in ("Unbekannt", "Keine DB"):
                        emotion = face_state.get("emotion", "")
                        emo_str = f", Emotion: {emotion}" if emotion else ""
                        gender = face_state.get("gender", "")
                        age_range = face_state.get("age_range", "")
                        ag_str = f", {gender}/{age_range}" if gender and age_range else ""
                        message_content = (
                            f"[Vision: Ich sehe {name} ({sim:.0%}){emo_str}{ag_str}]\\n\\n"
                            f"Markus sagt: {user_text}")'''

if old_vision_context in panel_code:
    panel_code = panel_code.replace(old_vision_context, new_vision_context, 1)
    panel_changes += 1
    print("PATCH 4: Age+Gender added to Claude chat context")
else:
    print("WARNING: Could not find vision context block in panel (checking...)")
    # Debug
    idx = panel_code.find('emo_str = f", Emotion:')
    if idx >= 0:
        print(f"  Found emo_str at pos {idx}")
        print(f"  Context: {panel_code[idx-50:idx+200]}")
    else:
        print("  emo_str NOT found at all")

with open(panel_path, "w") as f:
    f.write(panel_code)
print(f"Panel patched: {panel_changes} changes")

# =====================================================
# Summary
# =====================================================
print(f"\nDone! Age+Gender Detection eingebaut:")
print(f"  1. core/vision/age_gender_detector.py (NEU)")
print(f"  2. moloch_service.py: {changes} patches")
print(f"  3. hailo_postprocess.py: {pp_changes} patches")
print(f"  4. unified_panel.py: {panel_changes} patches")
print(f"\nIm Kamerabild: 'Markus (84%) [Happy] M/38-43'")
print(f"In face_state.json: gender: 'M', age_range: '38-43'")
print(f"In Claude-Chat: '[Vision: Ich sehe Markus (84%), Emotion: Happy, M/38-43]'")
