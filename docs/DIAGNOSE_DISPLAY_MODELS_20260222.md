# DIAGNOSE: Modelle & Bildgroessen — 2026-02-22

## 1. SERVICE & MODELLE

**Service:** `active` (laeuft)

| Modell | Aktiv | FPS | Bemerkung |
|--------|-------|-----|-----------|
| SCRFD (Face Detection) | JA | 25.7 | OK |
| ArcFace (Face Recognition) | NEIN (Status) / dynamisch | 5.8 | Wird bei Bedarf reingeswappt |
| YOLOv8m (Person Detection) | JA | 33.9 | OK |
| Hand Landmark | NEIN (Status) / dynamisch | 130.6 | Wird bei Occlusion reingeswappt |

**Modus:** Autonomous = true, Manual = false, Moloch hat Kontrolle

**Perception Engine:**
- 2 Slots aktiv, dynamisches Swapping funktioniert
- Scores: scrfd=1.07, arcface=0.55, yolov8m=0.72, hand=2.1
- Decision Count: 322.632 (laeuft seit langem stabil)
- Tension: 0.0

**Logs (letzte 5 Min):** Keine Errors/Exceptions. Modell-Swaps laufen sauber:
```
scrfd+yolov8m -> scrfd+hand_landmark (occlusion=True)
scrfd+hand_landmark -> scrfd+arcface (occlusion=False)
```
Autonomer Tracker funktioniert: TRACK -> FROZEN -> LOCKED Zyklus sichtbar.

**FAZIT:** Modelle laufen einwandfrei. Kein Problem hier.

---

## 2. PREVIEW-AUFLOESUNG (Panel)

**Code:** `core/gui/panel_preview.py`

| Einstellung | Wert |
|-------------|------|
| Default-Aufloesung | SD 640x360 |
| Verfuegbare Modi | SD 640x360, HD 800x450, HD+ 960x540, Full 1280x720 |
| MAX_CANVAS_W | 960 |
| MAX_CANVAS_H | 540 |

Der Preview zeigt den annotierten Frame (`_annotated_frame`), resized auf die gewaehlte Aufloesung.

**ABER:** Der Service liefert Frames in **640x480** (siehe unten), das Preview resized diese dann auf 640x360 (16:9 aus 4:3 = verzerrt!).

---

## 3. FRAME-PIPELINE IM SERVICE

**Code:** `core/moloch_service.py`

```
Kamera (1920x1080 RTSP)
  -> cv2.resize(frame, (640, 480))     # Zeile 387, PREVIEW_W/H
  -> _latest_frame = 640x480           # Zeile 417
  -> Inference: cv2.resize(frame, (640, 640))  # Zeile 726, fuer SCRFD/YOLO
  -> Face Crop: frame[y1:y2, x1:x2]   # Zeile 816, aus 640x480 Frame
  -> ArcFace: cv2.resize(crop, (112, 112))     # Zeile 817
```

**PROBLEM:** Die gesamte Pipeline arbeitet auf 640x480!
- Kamera liefert 1920x1080
- Service schrumpft SOFORT auf 640x480
- Face Crops kommen aus dem 640x480 Frame
- Snapshots (IPC) speichern den 640x480 Frame

---

## 4. DAILY LEARNER FOTO-GROESSE

**Code:** `core/daily_learner.py:143`
- Speichert `face_crop` direkt via `cv2.imwrite()`
- Kein Resize — die Groesse haengt von der Gesichts-BBox im 640x480 Frame ab

**Heutige Fotos (2026-02-22):**

| Datei | Groesse (px) | Dateigroesse |
|-------|-------------|--------------|
| 12-29-35_Markus_c61_a2_l1_d2.jpg | 172x282 | 21.8 KB |
| 12-36-38_Markus_c62_a2_l1_d1.jpg | 108x162 | 9.9 KB |

**FAZIT:** Face Crops sind winzig (108-172px breit) weil sie aus einem 640x480 Frame geschnitten werden. Bei 1920x1080 waeren sie ~3x groesser.

---

## 5. SNAPSHOT-GROESSE (Manuell via Panel)

**Code:** `core/moloch_service.py:2321-2338`
- Nimmt `_annotated_frame` (oder `_latest_frame` als Fallback)
- Speichert ohne Resize

**Letzte Snapshots:**

| Datei | Groesse (px) | Dateigroesse |
|-------|-------------|--------------|
| moloch_20260221_222308.jpg | 640x480 | 89 KB |
| moloch_20260221_222310.jpg | 640x480 | 85 KB |
| moloch_20260221_225506.jpg | 640x480 | 75 KB |
| moloch_20260221_225509.jpg | 640x480 | 83 KB |
| sonoff_cam_20260201_113547.jpg | 1920x1080 | 547 KB |

**FAZIT:** Aktuelle Snapshots nur 640x480. Der eine 1920x1080 Snapshot ist vom 01.02 (vermutlich direkt von der Kamera geholt, nicht ueber den Service).

---

## 6. MONITOR-AUFLOESUNG

```
HDMI-A-1: 1920x1080 (Full HD)
```

Der Monitor kann Full HD. Das Panel nutzt max 960x540 fuer die Preview.

---

## ZUSAMMENFASSUNG

| Aspekt | IST | SOLL | Problem? |
|--------|-----|------|----------|
| Service | Laeuft, kein Error | - | NEIN |
| Modelle | Swapping funktioniert | - | NEIN |
| FPS | 14-34 je Modell | - | NEIN |
| Kamera-Stream | 1920x1080 | - | OK |
| Service-Frame | 640x480 | 1920x1080? | JA - unnoetig runterskaliert |
| Preview-Display | 640x360 (aus 640x480) | - | JA - Seitenverhaeltnis falsch |
| Snapshots | 640x480 | 1920x1080 | JA - viel zu klein |
| Daily Learner Crops | 108-172px breit | 300-500px | JA - viel zu klein |
| Monitor | 1920x1080 | - | OK |

---

## VORGESCHLAGENE FIXES (NOCH NICHT AUSFUEHREN)

### Fix 1: Frame-Pipeline auf volle Aufloesung
**Datei:** `core/moloch_service.py`
- `_latest_frame` auf 1920x1080 belassen (kein Resize in Zeile 387)
- Inference-Resize (640x640 in Zeile 726) kann bleiben — das ist Modell-Input
- Face Crops waeren dann 3x groesser (aus 1080p statt 480p)
- Snapshots waeren automatisch 1080p

**Risiko:** Mehr RAM-Verbrauch (~6x pro Frame). Bei 2GB zram + 2GB swap knapp?
**Alternative:** Zwei Frames halten — 640x480 fuer Inference-Loop, 1080p fuer Snapshots/Crops.

### Fix 2: Preview Seitenverhaeltnis
**Datei:** `core/gui/panel_preview.py`
- PREVIEW_W/H im Service ist 640x480 (4:3)
- Panel-Aufloesung "SD 640x360" ist 16:9
- Entweder Service auf 640x360 (16:9) aendern, oder Panel auf 640x480 (4:3)
- Kamera liefert 1920x1080 (16:9), also waere 640x360 korrekt

### Fix 3: Snapshot in voller Aufloesung
**Datei:** `core/moloch_service.py`
- Bei Snapshot den Raw-Frame von der Kamera holen (1080p), nicht den resized Frame
- Oder separaten high-res Capture fuer Snapshots

### Fix 4: Daily Learner Crops vergroessern
**Datei:** `core/daily_learner.py` + `core/moloch_service.py`
- Wenn Fix 1 umgesetzt: Crops kommen automatisch groesser
- Optional: Minimum-Groesse erzwingen (z.B. min 200x200 per Upscale)
