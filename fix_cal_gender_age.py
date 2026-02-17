#!/usr/bin/env python3
"""Fix: Gender + Age Kalibrierungs-Phasen hinzufuegen.

1. calibration_engine.py: Pfade, Mappings, _run_gender(), _run_age(), start() erweitern
2. moloch_unified_panel.py: Radiobuttons fuer Gender + Alter
"""
import sys

# ============================================================
# TEIL 1: CalibrationEngine - Gender + Age Phasen
# ============================================================
cal = '/home/molochzuhause/moloch/core/calibration_engine.py'
with open(cal) as f:
    code = f.read()

fixes = 0

# FIX 1A: Pfade + Mappings oben hinzufuegen (nach GESTURE_MAP)
old_paths = '''GESTURE_MAP = {
    "thumbs_up": "thumbs_up",
    "peace": None,  # kein GestureType-Match
    "open_hand": "hand_raised_left",  # Annaeherung
    "fist": None,  # kein GestureType-Match
    "pointing": "pointing_right",  # Annaeherung
    "wave": "wave_right",  # Annaeherung
}'''

new_paths = '''GESTURE_MAP = {
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
}'''

if old_paths in code:
    code = code.replace(old_paths, new_paths)
    print('FIX 1A: Pfade + Mappings - OK')
    fixes += 1
else:
    print('FIX 1A: ANCHOR NOT FOUND!')

# FIX 1B: start() - Gender + Age Phasen routen
old_start_routes = '''            if phase == "emotions":
                self._run_emotions()
            elif phase == "gestures":
                self._run_gestures()
            else:
                logger.error(f"[CAL] Unbekannte Phase: {phase}")
                return'''

new_start_routes = '''            if phase == "emotions":
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
                return'''

if old_start_routes in code:
    code = code.replace(old_start_routes, new_start_routes)
    print('FIX 1B: start() Routen - OK')
    fixes += 1
else:
    print('FIX 1B: ANCHOR NOT FOUND!')

# FIX 1C: Gender + Age + Emotions HD Methoden (vor Hilfsmethoden)
old_helpers = '''    # =========================================================================
    # Hilfsmethoden
    # ========================================================================='''

new_methods_and_helpers = '''    # =========================================================================
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

            # Resize fuer SCRFD
            input_640 = cv2.resize(img, (640, 640))
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
                    # Groesstes Face nehmen
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best = areas.argmax()
                    x1, y1, x2, y2 = boxes[best].astype(int)

                    # Face crop aus Original-Groesse
                    h, w = img.shape[:2]
                    fx1 = int(x1 * w / 640)
                    fy1 = int(y1 * h / 640)
                    fx2 = int(x2 * w / 640)
                    fy2 = int(y2 * h / 640)
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2, fy2 = min(w, fx2), min(h, fy2)

                    if fx2 > fx1 + 10 and fy2 > fy1 + 10:
                        face_crop = img[fy1:fy2, fx1:fx2]
                        detected, confidence = emo_det.detect(face_crop)

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

            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": face_found,
                "progress": (self._processed, self._total_images),
                "phase": "emotions_hd",
            })

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
    # ========================================================================='''

if old_helpers in code:
    code = code.replace(old_helpers, new_methods_and_helpers)
    print('FIX 1C: Gender + Age + Emotions HD + Shared Helpers - OK')
    fixes += 1
else:
    print('FIX 1C: ANCHOR NOT FOUND!')

with open(cal, 'w') as f:
    f.write(code)

# Syntax check
compile(open(cal).read(), cal, 'exec')
print(f'\nCalibration Engine: {fixes}/3 Fixes. Syntax OK.')

# ============================================================
# TEIL 2: Panel - Radiobuttons fuer Gender + Alter
# ============================================================
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    pcode = f.read()

fixes2 = 0

# FIX 2A: Radiobuttons erweitern
old_radio = '''        tk.Radiobutton(phase_frame, text="Gesten", variable=self._cal_phase,
                        value="gestures", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)'''

new_radio = '''        tk.Radiobutton(phase_frame, text="Gesten", variable=self._cal_phase,
                        value="gestures", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)
        tk.Radiobutton(phase_frame, text="Emotionen HD", variable=self._cal_phase,
                        value="emotions_hd", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)
        tk.Radiobutton(phase_frame, text="Gender", variable=self._cal_phase,
                        value="gender", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)
        tk.Radiobutton(phase_frame, text="Alter", variable=self._cal_phase,
                        value="age", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)'''

if old_radio in pcode:
    pcode = pcode.replace(old_radio, new_radio)
    print('FIX 2A: Panel Radiobuttons - OK')
    fixes2 += 1
else:
    print('FIX 2A: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(pcode)

compile(open(panel).read(), panel, 'exec')
print(f'\nPanel: {fixes2}/1 Fixes. Syntax OK.')

total = fixes + fixes2
if total < 4:
    print(f'\n!!! INCOMPLETE: {total}/4 Fixes !!!')
    sys.exit(1)

print('\n=== GENDER + AGE CALIBRATION KOMPLETT ===')
print('Neue Phasen: Gender, Alter, Emotionen HD')
print('Panel: 5 Radiobuttons (Emotionen, Gesten, Emotionen HD, Gender, Alter)')
