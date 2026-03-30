# Agent Handoff — Session vigorous-chebyshev
**Datum:** 2026-03-30 | **Sitzung:** vigorous-chebyshev | **Status:** CLEAN (39/39 PASS)

---

## Was diese Sitzung erledigt hat

### ✅ Voice-Unification (core/voice_pipeline.py)
- `_speak()` delegiert jetzt an `personality_engine.speak()`
- Chat-Panel und autonome Ausgaben klingen **identisch** (Pitch-Shift, Modus-Stimme Guardian/Shadow/Berserker, Micro-Jitter)
- `_speak_direct()` bleibt als Fallback
- **Commit:** `78b1039`

### ✅ YOLOv11m Upgrade (core/perception/tappas_pipeline.py)
- YOLO_HEF: `yolov8m_h10.hef` → `yolov11m_h10.hef`
- mAP: 50.2 → 52.7 (+2.5), FPS stabil ~20
- **Commit:** `7e30d07`

### ✅ Action Inference (core/perception/action_inference.py — NEU)
- Temporal Pose Buffer (30 Frames Ringpuffer)
- Erkennt: `stehend`, `gehend`, `sitzend`, `winkend`, `zeigend`
- Eingebunden in `_on_pre_overlay` (Keypoints aus HAILO_LANDMARKS)
- `perception_frame.person_action` Feld hinzugefügt
- **Commit:** `78b1039`

### ✅ Gesture Classifier (core/perception/gesture_classifier.py — NEU)
- 21-Keypoint MediaPipe Hand-Gesten-Klassifizierer
- Gesten: `thumbs_up`, `open_hand`, `point`, `peace`, `fist`
- Fertig implementiert — wartet auf Hand-Valve-Crash-Fix
- **Commit:** `78b1039`

### ✅ FaceAttr bestätigt (BEREITS FERTIG — nicht doppelt codiert)
- `face_attr_resnet_v1_18.hef` war schon vollständig verdrahtet
- Model Scheduler: `faceattr` in MITTEL + NAH
- PerceptionFrame `gender` Feld war schon vorhanden
- Live bestätigt: Active Models zeigt `faceattr` in NAH-Szenario

---

## Offene Bugs — NÄCHSTE SITZUNG

### 🔴 KRITISCH 1: ReID Valve-Crash
**Was:** `reid_needed = False` hardcoded (~Zeile 1714, tappas_pipeline.py)
**Ursache (vermutet):** `libre_id.so` crasht mit cv2::resize wenn Pose-Detections
(HAILO_LANDMARKS) im ROI sind — Pose läuft IMMER → Race/Crash mit ReID-Cropper
**Fix-Ansatz:**
- Python Pad-Probe VOR ReID-Valve: Pose-Detections (has HAILO_LANDMARKS) temporär
  aus ROI entfernen, NACH ReID wieder zufügen
- ODER: ReID in Pipeline VOR Pose-Branch platzieren
**Datei:** `core/perception/tappas_pipeline.py`

### 🔴 KRITISCH 2: Hand Valve-Crash
**Was:** `hand_needed = False` hardcoded (~Zeile 1715, tappas_pipeline.py)
**Ursache:** `libwhole_buffer.so` → `create_crops` crasht bei Valve-Öffnung
**Fix-Ansatz:**
- WHOLE_BUFFER_SO-Wrapper entfernen
- Direkte Pipeline ohne hailocropper:
  `valve → queue → videoscale → videoconvert → video/x-raw → hailonet(hand_landmark_lite.hef)`
- Erst Input-Größe prüfen: `python3 -c "import hailo; ..."`
- gesture_classifier.py ist fertig und wartet
**Datei:** `core/perception/tappas_pipeline.py`

### 🟡 MITTEL 3: Face Landmark Misalignment
**Was:** Landmarks (Augen, Mundwinkel) passen nicht zu Gesichtspunkten im Overlay
**Ursache:** 50-Zeilen BBox-aus-Landmarks-Recalculation (~Zeile 1547-1605 in _on_buffer)
macht es SCHLIMMER als SCRFD-native Ausgabe
**Fix-Ansatz:**
- Den gesamten Block `# --- Face-BBox aus Landmarks berechnen ---` ENTFERNEN
- Original SCRFD BBox + Landmarks unverändert lassen
- hailooverlay rendert dann SCRFD-nativ
- Wenn dann immer noch falsch: separate Letterbox-Korrektur analysieren
**Datei:** `core/perception/tappas_pipeline.py` (~Zeile 1547-1605)

---

## System-Stand beim Handoff

```
Branch:      main (Commits deployed + rebootet)
Letzter Commit: 7e30d07
Audit:       39/39 PASS
FPS:         ~20 stabil
NPU aktiv:   YOLOv11m + SCRFD + ArcFace + FaceAttr (NAH-Szenario)
ReID:        DISABLED (Valve-Crash bekannt)
Hand:        DISABLED (Valve-Crash bekannt)
Pose:        AKTIV permanent
Action:      AKTIV (Temporal Buffer läuft)
Voice:       UNIFIED via personality_engine
```

---

## Start-Checklist neue Sitzung

1. `CLAUDE.md` lesen
2. Diese Handoff-Datei lesen
3. MCP Status + Audit: `python3 ~/moloch/moloch_audit.py --auto` → 39/39
4. **Ziel 1:** Hand-Valve-Fix (WHOLE_BUFFER_SO ersetzen)
5. **Ziel 2:** ReID-Valve-Fix (Pose-Detection-Race lösen)
6. **Ziel 3:** Face Landmark Alignment (BBox-Recalc entfernen)

---

## Alle geänderten Dateien (diese Sitzung, commit 78b1039 + 7e30d07)

| Datei | Art |
|-------|-----|
| `core/voice_pipeline.py` | _speak() → personality_engine |
| `core/perception/tappas_pipeline.py` | YOLO11m + ActionInferrer |
| `core/perception/perception_frame.py` | person_action Feld |
| `core/perception/action_inference.py` | NEU |
| `core/perception/gesture_classifier.py` | NEU |
