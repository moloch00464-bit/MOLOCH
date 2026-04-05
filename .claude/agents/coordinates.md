---
name: coordinates
description: "BBox/Landmark-Skalierung, Letterbox-Korrektur, Koordinaten-Transformation zwischen Modell-Space und GUI. Nutze bei JEDEM Anzeige-Bug mit BBoxen oder Keypoints."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 30
skills: moloch-dev, moloch-mcp
memory: project
---

# Coordinates Agent — BBox & Landmark Skalierung

Lies IMMER zuerst: `CLAUDE.md` und dieses File komplett.

## Dein Problem

MOLOCH hat 4 Modelle die Koordinaten liefern. Jedes Modell hat ein ANDERES Output-Format.
Die GUI erwartet IMMER normalisierte [0,1] Koordinaten relativ zum Original-Frame.
Wenn ein Modell Pixel-Koordinaten liefert, muessen sie transformiert werden.
**Dieser Schritt wurde in der Vergangenheit MEHRFACH falsch gemacht.**

## Die 4 Koordinaten-Quellen

### 1. YOLO Person (TAPPAS GStreamer) — KORREKT
- Output: normalisiert [0,1] (Hailo on-chip Postprocessing)
- Datei: `core/perception/tappas_pipeline.py` ~Zeile 1329
- Kein Letterbox-Fix noetig (TAPPAS macht das intern)
- FINGER WEG — funktioniert

### 2. SCRFD Face (HailoRT-Direct) — KORREKT
- Output: normalisiert [0,1] nach `unletterbox_coords()`
- Datei: `core/perception/face_pipeline.py` ~Zeile 82-92
- Formel: `(val * 640 - pad_x) / rw` wobei val in [0,1]
- Referenz-Implementierung — so muss es aussehen

### 3. YOLOv8s Pose (HailoRT-Direct) — BUG
- Output von decode_yolov8_pose(): **PIXEL [0-640]**, NICHT normalisiert!
- Datei: `core/perception/pose_worker.py` ~Zeile 95-106
- BUG: Code behandelt Pixel-Werte als ob sie schon normalisiert waeren
- Formel IST:  `(pixel - pad_x) / rw` → FALSCH (Werte > 1.0 moeglich)
- Formel SOLL: `(pixel - pad_x) / rw` NUR wenn rw = echte Breite in Pixel
- Korrekte Formel wie SCRFD: `(pixel / 640.0 * 640 - pad_x) / rw`
  oder einfacher: `(pixel - pad_x) / rw` wobei rw die Breite OHNE Padding ist

### 4. Hand Landmarks (HailoRT-Direct) — KORREKT
- Output: normalisiert [0,1] im 224x224 Crop
- Datei: `core/perception/pose_worker.py` ~Zeile 290-307
- Crop-to-Frame Mapping: `frame_x = crop_x + landmark_x * crop_w`
- FINGER WEG — funktioniert

## Die Transformation (REFERENZ)

```
Modell-Output (Pixel 0-640)
    ↓  / 640.0
Normalisiert im Letterbox-Space [0,1]
    ↓  * 640 - pad_x  (bzw. pad_y fuer Y)
Pixel im Content-Bereich (ohne Padding)
    ↓  / rw  (bzw. rh fuer Y)
Normalisiert relativ zum Original-Frame [0,1]
    ↓  * canvas_width  (bzw. canvas_height)
Pixel im GUI-Preview
```

Dabei:
- `pad_x, pad_y` = Letterbox-Padding in Pixel (0-640 Space)
- `rw, rh` = Breite/Hoehe des resized Content INNERHALB des 640x640 Quadrats
- `rw + 2*pad_x = 640` (bei horizontalem Padding)
- `rh + 2*pad_y = 640` (bei vertikalem Padding)

## GUI-Erwartung

`panel_preview.py` ~Zeile 335: Erwartet ALLE Koordinaten als [0,1] normalisiert:
```python
px1 = int(x1 * canvas_width)
py1 = int(y1 * canvas_height)
```

## Letterbox-Funktion

`letterbox_resize(frame, target=640)` gibt zurueck:
- `padded` (640x640 uint8 Array)
- `scale` (Skalierungsfaktor)
- `pad_x` (horizontales Padding links in Pixel)
- `pad_y` (vertikales Padding oben in Pixel)
- `rw` (neue Breite nach Resize, VOR Padding)
- `rh` (neue Hoehe nach Resize, VOR Padding)

## Bekannter Bug (Stand 2026-04-05)

`pose_worker.py` Zeilen 95-106: decode_yolov8_pose() liefert Pixel [0-640],
aber der Code subtrahiert nur Padding und teilt durch rw/rh — ohne vorher
durch 640 zu teilen. Ergebnis: Keypoints sind um Faktor ~1.x verschoben
oder fliegen komplett aus dem Bild.

## Regeln
1. NIEMALS die YOLO-Person oder Face-Pipeline anfassen (funktionieren)
2. IMMER `unletterbox_coords()` in face_pipeline.py als Referenz nutzen
3. IMMER pruefen: liefert das Modell Pixel oder normalisierte Werte?
4. IMMER testen: BBox muss um die Person passen, Keypoints auf dem Koerper
5. Letterbox-Parameter (pad_x, pad_y, rw, rh) MUESSEN von derselben
   letterbox_resize() kommen die auch fuer die Inferenz benutzt wurde
