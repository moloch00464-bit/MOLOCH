#!/usr/bin/env python3
"""Fix: Bessere Calibration-Anzeige.

1. INTER_CUBIC statt INTER_NEAREST (glatter)
2. BILDERBUCH MODUS Banner oben
3. Farbiger Rahmen um das Bild
"""
path = "/home/molochzuhause/moloch/core/calibration_engine.py"
with open(path) as f:
    code = f.read()

# Ersetze den Annotations-Block in _run_emotions
old_annotate = '''            # Annotiertes Bild erstellen (vergroessert fuer Anzeige)
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
                        (350, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)'''

new_annotate = '''            # Annotiertes Bild erstellen
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
                        (520, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)'''

if old_annotate in code:
    code = code.replace(old_annotate, new_annotate)
    print('FIX: Emotions-Anzeige verbessert - OK')
else:
    print('ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)

with open(path, 'w') as f:
    f.write(code)

compile(open(path).read(), path, 'exec')
print('Syntax OK')
print('\n=== DISPLAY FIX KOMPLETT ===')
