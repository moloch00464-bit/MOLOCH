#!/usr/bin/env python3
"""Fix: Emotions-Kalibrierung FER+ direkt auf FER2013 Face-Crops.

FER2013 sind 48x48 Face-Crops -> SCRFD ueberspringen.
"""
path = "/home/molochzuhause/moloch/core/calibration_engine.py"
with open(path) as f:
    code = f.read()

# Finde Start und Ende der _run_emotions Methode
start_marker = "    def _run_emotions(self):"
end_marker = "    # =========================================================================\n    # Phase 2: Gesten"

start_idx = code.find(start_marker)
end_idx = code.find(end_marker)

if start_idx < 0 or end_idx < 0:
    print(f"ANCHOR NOT FOUND! start={start_idx}, end={end_idx}")
    import sys
    sys.exit(1)

new_method = '''    def _run_emotions(self):
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

            # Annotiertes Bild erstellen (vergroessert fuer Anzeige)
            display = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
            annotated = np.zeros((480, 640, 3), dtype=np.uint8)
            annotated[:, 80:560] = display  # Zentriert

            color = (0, 255, 0) if correct else (0, 0, 255)
            status = "OK" if correct else "FALSCH"

            cv2.putText(annotated, f"GT: {expected}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Erkannt: {detected or '---'} ({confidence:.0%})",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, status,
                        (560, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Kategorie-Counter
            cat_stats = self._category_stats.get(category, {})
            cat_total = cat_stats.get("total", 0) + 1
            cat_correct = cat_stats.get("correct", 0) + (1 if correct else 0)
            cat_rate = cat_correct / cat_total if cat_total > 0 else 0
            cv2.putText(annotated, f"{category}: {cat_rate:.0%} ({cat_correct}/{cat_total})",
                        (350, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Frame injizieren
            with self.service._frame_lock:
                self.service._annotated_frame = annotated

            # Statistik aktualisieren
            self._processed += 1
            self._update_stats(category, expected, detected, confidence, correct)

            # Ergebnis ans Panel senden
            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "emotions",
            })

            # Tempo
            time.sleep(1.0 / self._speed)

'''

code = code[:start_idx] + new_method + code[end_idx:]

with open(path, 'w') as f:
    f.write(code)

# Verify
compile(open(path).read(), path, 'exec')
print("FIX OK + Syntax OK")
print(f"_run_emotions ersetzt: FER+ direkt auf Face-Crops, kein SCRFD")
