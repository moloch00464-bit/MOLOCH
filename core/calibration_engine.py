#!/usr/bin/env python3
"""
M.O.L.O.C.H. Calibration Engine
=================================
Bilderbuch-Kalibrierung: Referenzbilder durch NPU-Pipeline,
Vergleich mit Ground Truth, Erkennungsraten berechnen.

Phase 1: Emotionen (SCRFD + FER+ CPU)
Phase 2: Gesten (Pose NPU + GestureDetector)

Laeuft als Thread im MolochService. Ergebnisse via Observer ans Panel.
"""
import os
import time
import json
import random
import logging
import threading
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("CalibrationEngine")

# Pfade
EMOTIONS_DIR = "/mnt/moloch-data/reference/emotions"
GESTURES_DIR = "/mnt/moloch-data/reference/gestures_fullbody"
RESULTS_PATH = os.path.expanduser("~/moloch/data/calibration_results.json")
CAL_EVENT_SHM = "/dev/shm/moloch_cal_event.json"

# Emotion-Mapping: FER2013 Ordnername -> FER+ 5-Klassen
# fear -> Sad (naechste verwandte), disgust -> Angry (naechste verwandte)
EMOTION_MAP = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "neutral": "Neutral",
    "surprised": "Surprised",
    "fear": "Sad",
    "disgust": "Angry",
}

# Gesten-Mapping: HaGRID Ordnername -> GestureType.value (oder None fuer "nur Hand erkannt")
GESTURE_MAP = {
    "thumbs_up": "thumbs_up",
    "peace": None,  # kein GestureType-Match
    "open_hand": "hand_raised_left",  # Annaeherung
    "fist": None,  # kein GestureType-Match
    "pointing": "pointing_right",  # Annaeherung
    "wave": "wave_right",  # Annaeherung
}

# Gender + Age Pfade (FairFace 224x224 Face-Crops)
GENDER_DIR = "/mnt/moloch-data/reference/gender"
AGE_DIR = "/mnt/moloch-data/reference/age"
EMOTIONS_HD_DIR = "/mnt/moloch-data/reference/emotions_hd"

# Gender-Mapping: Ordnername -> Detector-Output ("M"/"F")
GENDER_MAP = {"male": "M", "female": "F"}

# Age-Mapping: MOLOCH-Klasse -> passende Caffe AGE_BUCKETS
# Caffe: ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
AGE_BUCKET_MAP = {
    "kind": ["0-2", "4-6", "8-12"],
    "jung": ["15-20", "25-32"],
    "mittel": ["38-43", "48-53"],
    "alt": ["60+"],
}


class CalibrationEngine:
    """Bilderbuch-Kalibrierung fuer Emotionen und Gesten."""

    def __init__(self, service):
        self.service = service
        self._running = False
        self._paused = False
        self._stop_requested = False
        self._speed = 3
        self._phase = None
        self._results = []
        self._category_stats = {}
        self._total_images = 0
        self._processed = 0
        self._model_swaps = 0
        self._swap_times = []
        self._start_time = 0
        self._lock = threading.Lock()

    def _write_cal_event(self, event_type, data):
        """Event auch als SHM-Datei fuer ServiceProxy schreiben."""
        try:
            import json as _json
            payload = {"event": event_type, "data": data, "ts": time.time()}
            tmp = CAL_EVENT_SHM + ".tmp"
            with open(tmp, "w") as f:
                _json.dump(payload, f)
            os.rename(tmp, CAL_EVENT_SHM)
        except Exception:
            pass

    def start(self, phase: str, speed: int = 3):
        """Starte Kalibrierung (aufgerufen in Thread)."""
        self._phase = phase
        self._speed = max(1, min(10, speed))
        self._running = True
        self._paused = False
        self._stop_requested = False
        self._results = []
        self._category_stats = {}
        self._processed = 0
        self._start_time = time.time()

        logger.info(f"[CAL] Start: phase={phase}, speed={speed}")
        self.service._calibration_active = True

        # Tracker + Autonomie pausieren waehrend Kalibrierung
        self._prev_autonomous = getattr(self.service, "_autonomous_mode", False)
        if self._prev_autonomous:
            try:
                self.service._autonomous_mode = False
                if self.service._tracker:
                    self.service._tracker.stop()
                logger.info("[CAL] Tracker pausiert")
            except Exception as e:
                logger.warning(f"[CAL] Tracker pause failed: {e}")

        # Wrap _notify: Alle calibration_* Events auch via SHM senden
        self._orig_notify = self.service._notify
        def _cal_notify(event, data):
            self._orig_notify(event, data)
            if event.startswith("calibration"):
                self._write_cal_event(event, data)
        self.service._notify = _cal_notify

        self.service._notify("calibration_status", {
            "status": "running", "phase": phase})

        try:
            if phase == "emotions":
                self._run_emotions()
            elif phase == "emotions_hd":
                self._run_emotions_hd()
            elif phase == "gestures":
                self._run_gestures()
            elif phase == "gender":
                self._run_gender()
            elif phase == "age":
                self._run_age()
            else:
                logger.error(f"[CAL] Unbekannte Phase: {phase}")
                return
        except Exception as e:
            logger.error(f"[CAL] Fehler: {e}", exc_info=True)
        finally:
            self._finish()

    def pause(self):
        """Pausieren/Fortsetzen."""
        self._paused = not self._paused
        state = "paused" if self._paused else "running"
        logger.info(f"[CAL] {state}")
        _s = {"status": state, "phase": self._phase}
        self.service._notify("calibration_status", _s)

    def stop(self):
        """Abbrechen."""
        logger.info("[CAL] Stop requested")
        self._stop_requested = True

    # =========================================================================
    # Phase 1: Emotionen
    # =========================================================================

    def _run_emotions(self):
        """Emotionen kalibrieren: FER+ direkt auf FER2013 Face-Crops.

        FER2013 Bilder sind 48x48 Face-Crops - SCRFD ueberspringen,
        FER+ Emotion Detector direkt auf das Bild anwenden.
        """
        if not os.path.isdir(EMOTIONS_DIR):
            logger.error(f"[CAL] Emotions dir fehlt: {EMOTIONS_DIR}")
            return

        # Alle Bilder sammeln
        image_list = []
        for category in sorted(os.listdir(EMOTIONS_DIR)):
            cat_dir = os.path.join(EMOTIONS_DIR, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_list.append((category, os.path.join(cat_dir, fname)))

        self._total_images = len(image_list)
        logger.info(f"[CAL] Emotionen: {self._total_images} Bilder in "
                     f"{len(set(c for c,_ in image_list))} Kategorien (FER+ direkt)")

        # Emotion Detector laden (CPU - kein NPU noetig!)
        from core.vision.emotion_detector import get_emotion_detector
        emo_det = get_emotion_detector()
        if not emo_det.available:
            logger.error("[CAL] FER+ Emotion Detector nicht verfuegbar!")
            return

        for category, img_path in image_list:
            if self._stop_requested:
                break
            while self._paused and not self._stop_requested:
                time.sleep(0.1)
            if self._stop_requested:
                break

            fname = os.path.basename(img_path)
            expected = EMOTION_MAP.get(category, category)

            # Bild laden (FER2013 = 48x48 Grayscale Face-Crop)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # FER+ direkt auf das Face-Crop anwenden (kein SCRFD noetig!)
            detected, confidence = emo_det.detect(img)
            correct = (detected == expected) if detected else False

            # Annotiertes Bild erstellen
            annotated = np.zeros((480, 640, 3), dtype=np.uint8)
            annotated[:] = (10, 10, 20)  # Dunkler Hintergrund

            # Face vergroessert anzeigen (glatt skaliert)
            display = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
            # Zentriert platzieren
            y_off, x_off = 80, 160
            annotated[y_off:y_off+320, x_off:x_off+320] = display

            # Farbiger Rahmen
            color = (0, 255, 0) if correct else (0, 0, 255)
            cv2.rectangle(annotated, (x_off-2, y_off-2),
                          (x_off+322, y_off+322), color, 2)

            status = "OK" if correct else "FALSCH"

            # Banner: BILDERBUCH MODUS
            cv2.rectangle(annotated, (0, 0), (640, 35), (30, 30, 60), -1)
            cv2.putText(annotated, "BILDERBUCH: Emotionen",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            pct = (self._processed + 1) / self._total_images * 100
            cv2.putText(annotated, f"{self._processed+1}/{self._total_images} ({pct:.0f}%)",
                        (430, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # Ground Truth (links)
            cv2.putText(annotated, f"Soll: {expected}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Ergebnis (links unten)
            cv2.putText(annotated, f"Erkannt: {detected or '---'}",
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"Konfidenz: {confidence:.0%}",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # Status (rechts oben)
            cv2.putText(annotated, status,
                        (550, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Kategorie-Stats (rechts unten)
            cat_stats = self._category_stats.get(category, {})
            cat_total = cat_stats.get("total", 0) + 1
            cat_correct = cat_stats.get("correct", 0) + (1 if correct else 0)
            cat_rate = cat_correct / cat_total if cat_total > 0 else 0
            rate_color = (0, 255, 100) if cat_rate >= 0.7 else ((0, 180, 255) if cat_rate >= 0.5 else (0, 0, 255))
            cv2.putText(annotated, f"{category}: {cat_rate:.0%}",
                        (500, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2)
            cv2.putText(annotated, f"({cat_correct}/{cat_total})",
                        (520, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

            # Frame via shm ans Panel senden
            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)

            # Statistik aktualisieren
            self._processed += 1
            self._update_stats(category, expected, detected, confidence, correct)

            # Ergebnis ans Panel senden (Observer + SHM fuer Remote)
            _result = {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "emotions",
            }
            self.service._notify("calibration_result", _result)

            # Tempo
            time.sleep(1.0 / self._speed)

    # =========================================================================
    # Phase 2: Gesten
    # =========================================================================

    def _run_gestures(self):
        """Gesten kalibrieren: Pose(NPU) + GestureDetector(CPU)."""
        if not os.path.isdir(GESTURES_DIR):
            logger.error(f"[CAL] Gestures dir fehlt: {GESTURES_DIR}")
            _err = {"status": "error", "message": f"Ordner fehlt: {GESTURES_DIR}"}
            self.service._notify("calibration_status", _err)
            return

        # Alle Bilder sammeln
        image_list = []
        for category in sorted(os.listdir(GESTURES_DIR)):
            cat_dir = os.path.join(GESTURES_DIR, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_list.append((category, os.path.join(cat_dir, fname)))

        self._total_images = len(image_list)
        logger.info(f"[CAL] Gesten: {self._total_images} Bilder in {len(set(c for c,_ in image_list))} Kategorien")

        # Pose-Modell sicherstellen
        self._ensure_model("pose")

        # GestureDetector laden
        from core.vision.gesture_detector import GestureDetector, KeypointPosition
        from core.perception.hailo_postprocess import decode_yolov8_pose

        gesture_det = GestureDetector()

        for category, img_path in image_list:
            if self._stop_requested:
                break
            while self._paused and not self._stop_requested:
                time.sleep(0.1)
            if self._stop_requested:
                break

            fname = os.path.basename(img_path)
            expected_gesture = GESTURE_MAP.get(category)

            # Bild laden
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize auf 640x640 fuer Pose
            input_640 = cv2.resize(img, (640, 640))
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

            # Pose Detection
            detected = None
            confidence = 0.0
            person_found = False
            hand_found = False

            outputs = self.service._run_model("pose", input_rgb)
            if outputs:
                poses = decode_yolov8_pose(
                    outputs, img_h=640, img_w=640,
                    conf_thresh=0.3, iou_thresh=0.7)

                if poses:
                    person_found = True
                    # Beste Pose nehmen
                    pose = poses[0]
                    kpts = pose["keypoints"]  # (17, 3)

                    # Keypoints in GestureDetector-Format
                    kp_list = []
                    for i in range(17):
                        kp_list.append(KeypointPosition(
                            x=float(kpts[i, 0]) / 640.0,
                            y=float(kpts[i, 1]) / 640.0,
                            confidence=float(kpts[i, 2]),
                            visible=float(kpts[i, 2]) > 0.3
                        ))

                    # Wrist sichtbar = Hand gefunden
                    for wi in (9, 10):
                        if kpts[wi, 2] > 0.3:
                            hand_found = True
                            break

                    # Geste erkennen
                    gesture = gesture_det.detect(kp_list)
                    if gesture:
                        detected = gesture.type.value
                        confidence = gesture.confidence

            # Vergleich
            if expected_gesture:
                # Exakter Match oder verwandte Gesten
                correct = detected is not None and (
                    detected == expected_gesture
                    or (expected_gesture.startswith("wave") and detected and "wave" in detected)
                    or (expected_gesture.startswith("pointing") and detected and "pointing" in detected)
                    or (expected_gesture.startswith("hand_raised") and detected and "hand_raised" in detected)
                )
            else:
                # Kein GestureType-Match: "korrekt" wenn Hand erkannt
                correct = hand_found

            # Annotiertes Bild
            annotated = cv2.resize(img, (640, 480))
            color = (0, 255, 0) if correct else (0, 0, 255)
            status_text = "OK" if correct else "FALSCH"

            cv2.putText(annotated, f"GT: {category}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            det_text = detected if detected else ("Hand" if hand_found else "---")
            cv2.putText(annotated, f"Erkannt: {det_text} ({confidence:.0%})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, status_text,
                        (560, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if not person_found:
                cv2.putText(annotated, "KEINE PERSON",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Frame via shm ans Panel senden
            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)

            # Statistik
            self._processed += 1
            self._update_stats(category, expected_gesture or f"{category}(hand)",
                               detected or ("hand" if hand_found else None),
                               confidence, correct)

            # Ergebnis ans Panel
            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected_gesture or f"{category}(hand)",
                "detected": det_text,
                "confidence": round(confidence, 3),
                "correct": correct,
                "person_found": person_found,
                "hand_found": hand_found,
                "progress": (self._processed, self._total_images),
                "phase": "gestures",
            })

            time.sleep(1.0 / self._speed)

    # =========================================================================
    # Phase 3: Gender (FairFace -> Caffe GoogLeNet)
    # =========================================================================

    def _run_gender(self):
        """Gender kalibrieren: Caffe GoogLeNet direkt auf FairFace Face-Crops."""
        if not os.path.isdir(GENDER_DIR):
            logger.error(f"[CAL] Gender dir fehlt: {GENDER_DIR}")
            self.service._notify("calibration_status", {
                "status": "error", "message": f"Ordner fehlt: {GENDER_DIR}"})
            return

        image_list = self._collect_images(GENDER_DIR)
        self._total_images = len(image_list)
        logger.info(f"[CAL] Gender: {self._total_images} Bilder")

        from core.vision.age_gender_detector import get_age_gender_detector
        ag_det = get_age_gender_detector()
        if not ag_det.available:
            logger.error("[CAL] Age/Gender Detector nicht verfuegbar!")
            self.service._notify("calibration_status", {
                "status": "error", "message": "Age/Gender Detector fehlt"})
            return

        for category, img_path in image_list:
            if self._stop_requested:
                break
            while self._paused and not self._stop_requested:
                time.sleep(0.1)
            if self._stop_requested:
                break

            fname = os.path.basename(img_path)
            expected = GENDER_MAP.get(category, category)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Caffe direkt auf Face-Crop (kein SCRFD noetig!)
            detected_gender, detected_age, confidence = ag_det.detect(img)
            correct = (detected_gender == expected) if detected_gender else False

            # Annotiertes Bild
            annotated = self._make_annotated(
                img, "Gender", category, expected,
                detected_gender or "---", confidence, correct,
                extra_info=f"Age: {detected_age or '?'}")

            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)

            self._processed += 1
            self._update_stats(category, expected, detected_gender, confidence, correct)

            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected_gender or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "progress": (self._processed, self._total_images),
                "phase": "gender",
            })

            time.sleep(1.0 / self._speed)

    # =========================================================================
    # Phase 4: Age (FairFace -> Caffe GoogLeNet)
    # =========================================================================

    def _run_age(self):
        """Alter kalibrieren: Caffe GoogLeNet direkt auf FairFace Face-Crops."""
        if not os.path.isdir(AGE_DIR):
            logger.error(f"[CAL] Age dir fehlt: {AGE_DIR}")
            self.service._notify("calibration_status", {
                "status": "error", "message": f"Ordner fehlt: {AGE_DIR}"})
            return

        image_list = self._collect_images(AGE_DIR)
        self._total_images = len(image_list)
        logger.info(f"[CAL] Alter: {self._total_images} Bilder")

        from core.vision.age_gender_detector import get_age_gender_detector
        ag_det = get_age_gender_detector()
        if not ag_det.available:
            logger.error("[CAL] Age/Gender Detector nicht verfuegbar!")
            self.service._notify("calibration_status", {
                "status": "error", "message": "Age/Gender Detector fehlt"})
            return

        for category, img_path in image_list:
            if self._stop_requested:
                break
            while self._paused and not self._stop_requested:
                time.sleep(0.1)
            if self._stop_requested:
                break

            fname = os.path.basename(img_path)
            valid_buckets = AGE_BUCKET_MAP.get(category, [])

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Caffe direkt auf Face-Crop
            detected_gender, detected_age, confidence = ag_det.detect(img)
            correct = (detected_age in valid_buckets) if detected_age else False

            # Annotiertes Bild
            annotated = self._make_annotated(
                img, "Alter", category, "/".join(valid_buckets),
                detected_age or "---", confidence, correct,
                extra_info=f"Gender: {detected_gender or '?'}")

            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)

            self._processed += 1
            self._update_stats(category, category, detected_age, confidence, correct)

            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": "/".join(valid_buckets),
                "detected": detected_age or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "progress": (self._processed, self._total_images),
                "phase": "age",
            })

            time.sleep(1.0 / self._speed)

    # =========================================================================
    # Phase 5: Emotionen HD (SCRFD + FER+ auf echten Portraets)
    # =========================================================================

    def _run_emotions_hd(self):
        """Emotionen HD: SCRFD Face Detection + FER+ auf grossen Bildern."""
        if not os.path.isdir(EMOTIONS_HD_DIR):
            logger.error(f"[CAL] Emotions HD dir fehlt: {EMOTIONS_HD_DIR}")
            self.service._notify("calibration_status", {
                "status": "error", "message": f"Ordner fehlt: {EMOTIONS_HD_DIR}"})
            return

        image_list = self._collect_images(EMOTIONS_HD_DIR)
        self._total_images = len(image_list)
        logger.info(f"[CAL] Emotionen HD: {self._total_images} Bilder (SCRFD + FER+)")

        # SCRFD auf NPU laden
        self._ensure_model("scrfd")

        from core.vision.emotion_detector import get_emotion_detector
        from core.perception.hailo_postprocess import decode_scrfd
        emo_det = get_emotion_detector()
        if not emo_det.available:
            logger.error("[CAL] FER+ Emotion Detector nicht verfuegbar!")
            return

        for category, img_path in image_list:
            if self._stop_requested:
                break
            while self._paused and not self._stop_requested:
                time.sleep(0.1)
            if self._stop_requested:
                break

            fname = os.path.basename(img_path)
            expected = EMOTION_MAP.get(category, category)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Face-Crop auf 640x640 Canvas platzieren (nicht strecken!)
            # FairFace = 224x224 Face-Crop -> in Mitte mit dunklem Rand
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)  # Dunkler Hintergrund
            # Face auf ~250x250 skalieren (realistisches Verhaeltnis)
            face_size = min(280, max(200, img.shape[0] * 640 // 480))
            face_resized = cv2.resize(img, (face_size, face_size))
            y_off = (640 - face_size) // 2
            x_off = (640 - face_size) // 2
            canvas[y_off:y_off+face_size, x_off:x_off+face_size] = face_resized
            input_640 = canvas
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

            # Face Detection via NPU
            detected = None
            confidence = 0.0
            face_found = False

            outputs = self.service._run_model("scrfd", input_rgb)
            if outputs:
                boxes, scores, landmarks = decode_scrfd(
                    outputs, img_size=640, conf_thresh=0.3)

                if len(boxes) > 0:
                    face_found = True
                    # Original-Bild IST der Face-Crop (FairFace 224x224)
                    # -> FER+ direkt auf Originalbild anwenden
                    detected, confidence = emo_det.detect(img)

            correct = (detected == expected) if detected else False

            # Annotiertes Bild
            annotated = self._make_annotated(
                img, "Emotionen HD", category, expected,
                detected or "---", confidence, correct,
                extra_info="SCRFD+FER+" if face_found else "KEIN FACE!")

            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)

            self._processed += 1
            self._update_stats(category, expected, detected, confidence, correct)

            _result = {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": face_found,
                "progress": (self._processed, self._total_images),
                "phase": "emotions_hd",
            }
            self.service._notify("calibration_result", _result)

            time.sleep(1.0 / self._speed)

    # =========================================================================
    # Shared Helpers
    # =========================================================================

    def _collect_images(self, base_dir):
        """Bilder aus Unterordnern sammeln."""
        image_list = []
        for category in sorted(os.listdir(base_dir)):
            cat_dir = os.path.join(base_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_list.append((category, os.path.join(cat_dir, fname)))
        return image_list

    def _make_annotated(self, img, phase_name, category, expected,
                        detected, confidence, correct, extra_info=""):
        """Einheitliches annotiertes Bild fuer alle Phasen."""
        annotated = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated[:] = (10, 10, 20)

        # Bild vergroessert anzeigen
        h, w = img.shape[:2]
        scale = min(320 / w, 320 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        y_off = 80 + (320 - new_h) // 2
        x_off = 160 + (320 - new_w) // 2
        annotated[y_off:y_off+new_h, x_off:x_off+new_w] = display

        color = (0, 255, 0) if correct else (0, 0, 255)
        status = "OK" if correct else "FALSCH"

        # Rahmen
        cv2.rectangle(annotated, (x_off-2, y_off-2),
                      (x_off+new_w+2, y_off+new_h+2), color, 2)

        # Banner
        cv2.rectangle(annotated, (0, 0), (640, 35), (30, 30, 60), -1)
        cv2.putText(annotated, f"BILDERBUCH: {phase_name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        pct = (self._processed + 1) / self._total_images * 100
        cv2.putText(annotated, f"{self._processed+1}/{self._total_images} ({pct:.0f}%)",
                    (430, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Infos
        cv2.putText(annotated, f"Soll: {expected}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated, f"Erkannt: {detected}",
                    (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Konfidenz: {confidence:.0%}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(annotated, status,
                    (550, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if extra_info:
            cv2.putText(annotated, extra_info,
                        (400, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        # Kategorie-Stats
        cat_stats = self._category_stats.get(category, {})
        cat_total = cat_stats.get("total", 0) + 1
        cat_correct = cat_stats.get("correct", 0) + (1 if correct else 0)
        cat_rate = cat_correct / cat_total if cat_total > 0 else 0
        rate_color = (0, 255, 100) if cat_rate >= 0.7 else (
            (0, 180, 255) if cat_rate >= 0.5 else (0, 0, 255))
        cv2.putText(annotated, f"{category}: {cat_rate:.0%} ({cat_correct}/{cat_total})",
                    (430, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rate_color, 1)

        return annotated

    # =========================================================================
    # Hilfsmethoden
    # =========================================================================

    def _ensure_model(self, name: str):
        """Modell auf NPU laden wenn nicht aktiv."""
        with self.service._ctx_lock:
            active = name in self.service._active_ctx
        if not active:
            t0 = time.time()
            self.service._configure_model(name)
            dt = time.time() - t0
            self._model_swaps += 1
            self._swap_times.append(dt * 1000)
            logger.info(f"[CAL] Modell {name} geladen ({dt*1000:.0f}ms)")

    def _update_stats(self, category: str, expected, detected, confidence: float, correct: bool):
        """Kategorie-Statistik aktualisieren."""
        if category not in self._category_stats:
            self._category_stats[category] = {
                "total": 0, "correct": 0, "conf_sum": 0.0,
                "no_detect": 0, "confusion": {}
            }
        s = self._category_stats[category]
        s["total"] += 1
        if correct:
            s["correct"] += 1
        if detected:
            s["conf_sum"] += confidence
        else:
            s["no_detect"] += 1

        # Confusion Tracking
        if not correct and detected:
            s["confusion"][detected] = s["confusion"].get(detected, 0) + 1

    def _finish(self):
        """Kalibrierung abschliessen, Ergebnisse speichern."""
        self._running = False
        self.service._calibration_active = False
        duration = time.time() - self._start_time

        # Zusammenfassung
        total_correct = sum(s["correct"] for s in self._category_stats.values())
        total_all = sum(s["total"] for s in self._category_stats.values())
        overall_rate = total_correct / total_all if total_all > 0 else 0

        categories = {}
        for cat, s in self._category_stats.items():
            rate = s["correct"] / s["total"] if s["total"] > 0 else 0
            avg_conf = s["conf_sum"] / (s["total"] - s["no_detect"]) if (s["total"] - s["no_detect"]) > 0 else 0
            categories[cat] = {
                "total": s["total"],
                "correct": s["correct"],
                "rate": round(rate, 3),
                "avg_conf": round(avg_conf, 3),
                "no_detect": s["no_detect"],
                "confusion": s["confusion"],
            }

        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phase": self._phase,
            "duration_seconds": round(duration, 1),
            "total": total_all,
            "correct": total_correct,
            "rate": round(overall_rate, 3),
            "categories": categories,
            "model_swaps": self._model_swaps,
            "avg_swap_time_ms": round(sum(self._swap_times) / len(self._swap_times), 1) if self._swap_times else 0,
            "stopped_early": self._stop_requested,
        }

        # Speichern
        try:
            os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
            # Bestehende Ergebnisse laden
            existing = {}
            if os.path.exists(RESULTS_PATH):
                with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)

            if "phases" not in existing:
                existing["phases"] = {}
            existing["phases"][self._phase] = result
            existing["last_run"] = result["timestamp"]

            import tempfile as _tf
            with _tf.NamedTemporaryFile("w", dir=os.path.dirname(RESULTS_PATH),
                                        delete=False, suffix=".tmp",
                                        encoding="utf-8") as tf:
                json.dump(existing, tf, indent=2, ensure_ascii=False)
                _tmp = tf.name
            os.replace(_tmp, RESULTS_PATH)
            logger.info(f"[CAL] Ergebnisse gespeichert: {RESULTS_PATH}")
        except Exception as e:
            logger.error(f"[CAL] Speichern fehlgeschlagen: {e}")

        # Schwellenwert-Anpassung
        self._adjust_thresholds(categories)

        # Panel benachrichtigen
        self.service._notify("calibration_status", {
            "status": "finished",
            "phase": self._phase,
            "total": total_all,
            "correct": total_correct,
            "rate": overall_rate,
            "duration": duration,
        })

        # _notify Wrapper wiederherstellen
        if hasattr(self, "_orig_notify"):
            self.service._notify = self._orig_notify

        # Tracker wieder aktivieren wenn er vorher lief
        if getattr(self, "_prev_autonomous", False):
            try:
                self.service._autonomous_mode = True
                logger.info("[CAL] Tracker wieder aktiviert")
            except Exception:
                pass

        logger.info(f"[CAL] Fertig: {self._phase} - {total_correct}/{total_all} "
                     f"({overall_rate:.1%}) in {duration:.0f}s")

    def _adjust_thresholds(self, categories: Dict):
        """Schwellenwerte anpassen fuer Kategorien mit <70% Erkennungsrate."""
        weights_path = os.path.expanduser("~/moloch/config/perception_weights.json")
        try:
            data = {}
            if os.path.exists(weights_path):
                with open(weights_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            weights = data.get("weights", {})
            adjusted = False

            for cat, stats in categories.items():
                if stats["rate"] < 0.70 and stats["total"] >= 10:
                    # Score fuer das relevante Modell erhoehen
                    if self._phase == "emotions":
                        # SCRFD Score leicht erhoehen (Face Detection ist der Flaschenhals)
                        old = weights.get("scrfd", 0.0)
                        weights["scrfd"] = round(min(0.3, old + 0.02), 4)
                        adjusted = True
                    elif self._phase == "gestures":
                        old = weights.get("pose", 0.0)
                        weights["pose"] = round(min(0.3, old + 0.02), 4)
                        adjusted = True

            if adjusted:
                data["weights"] = weights
                data["calibration_adjusted"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                import tempfile as _tf2
                with _tf2.NamedTemporaryFile("w", dir=os.path.dirname(weights_path),
                                             delete=False, suffix=".tmp",
                                             encoding="utf-8") as tf:
                    json.dump(data, tf, indent=2)
                    _tmp2 = tf.name
                os.replace(_tmp2, weights_path)
                logger.info(f"[CAL] Weights angepasst: {weights}")
        except Exception as e:
            logger.warning(f"[CAL] Threshold-Anpassung fehlgeschlagen: {e}")

    def get_summary(self) -> Dict:
        """Aktueller Zustand fuer GUI."""
        return {
            "running": self._running,
            "paused": self._paused,
            "phase": self._phase,
            "processed": self._processed,
            "total": self._total_images,
            "categories": dict(self._category_stats),
        }
