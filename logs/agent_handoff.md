# M.O.L.O.C.H. Übergabeprotokoll
**Datum:** 2026-03-28, 15:50 CET
**Von:** Claude Opus 4.6 — Tracking + BBox + Landmark Session
**Service-Status:** LAEUFT (20 FPS, ArcFace 0.78+, Audit PASS)
**USE_TAPPAS:** 1 (aktiv)

---

## ERLEDIGT IN DIESER SESSION (12 Tasks)

### 1. Tracking Gain-Tuning + BBox-Smoothing ✅
- camera.py: TRACKING_GAIN_PAN 0.7→0.4, TILT 0.5→0.3, MAX_STEP 30→15/20→12
- 3-Frame Error-Smoothing eingebaut (Ringbuffer in process_detection)
- Deadzone war schon vorhanden (0.05)
- Commit: `803780c`

### 2. FaceAttr + Pose FPS im Panel ✅
- panel_models.py: ("FaceAttr", "faceattr") zu TAPPAS_MODELS
- status_key_map: "faceattr": "faceattr_active"
- FPS-Detail zeigt jetzt scrfd, arcface, yolov8m, faceattr, pose
- Commit: `5710c08`

### 3. Park-Position = Tür ✅
- camera_home.json: Pan 50.0, Tilt -20.0 (aus ONVIF-Preset moloch_tuer)
- camera.py + camera_manager.py Defaults angepasst
- Commits: `ba1d05e`, `7698e52`

### 4. Tension-Popup Kontrast ✅
- popup_npu_thresh.py: Tooltip-Farbe FG_DIM #666→#999 (besser lesbar)
- Commit: `4d48d60`

### 5. Smart-Tracking Freigabe via eWeLink ✅
- camera_manager.py: Gate-0-Lock entfernt
- toggle_smart_tracking() nutzt jetzt Cloud-API (smartTraceEnable=1/0)
- Ein/Aus per IPC oder Panel-Button
- Commit: `3e5eac2`

### 6. Hybrid-Tracking: ST + MOLOCH ✅
- **Kernlogik:** Kein Gesicht → Smart-Tracking bleibt an (Kamera-Sensoren schneller)
- Face erkannt → MOLOCH uebernimmt fuer Praezisions-Tracking
- Auto-ST: Wenn MOLOCH BBox 20 Cycles lang >35% off-center → ST einschalten
- ST wird erst abgeschaltet wenn Kamera settled + Face erkannt
- Commits: `80e4795`, `80117bb`

### 7. Aggressives Tracking (Vollgas) ✅
- autonomous_tracker.py: pan_gain 0.65, tilt_gain 0.50, max_step 25/18
- Speed 1.0, Cooldown 50ms, EMA alpha 0.70
- Dead-Zone: frozen<20px, coast<25px, resume>50px (kein Ping-Pong mehr)
- Commits: `18d3bc7`, `3093668`, `20de79d`

### 8. STMovementLearner erweitert ✅
- Rohpositionen aufzeichnen (Ringbuffer 500)
- Pan/Tilt Geschwindigkeiten berechnen (deg/s)
- get_learned_dynamics(): avg/max/median Velocity
- Commit: `df24943`

### 9. Gesicht-Zentrierung Y-Referenz ✅
- frame_center_y: 0.33→0.40 (Gesicht naeher an Bildmitte statt oberes Drittel)
- Commit: `20de79d`

### 10. Face-BBox aus Landmarks berechnet ✅ (DURCHBRUCH!)
- **Alter SHRINK_Y=0.50 Hack ersetzt** durch landmark-basierte BBox
- SCRFD liefert 5 Landmarks (Augen, Nase, Mundwinkel) — die sind KORREKT
- BBox wird jetzt aus Landmark-Extremen + Padding berechnet
- Padding: 25% X, 10% oben (Stirn), 50% unten (Kinn)
- Landmarks werden korrekt auf neue BBox umgerechnet (bbox-relativ → Frame → neue bbox-relativ)
- ArcFace Similarity: 0.25→0.78+ (vorher kaputt wegen falscher Landmarks)
- Commits: `bf9d06b`, `41c7d92`, `354ac27`, `559b54c`, `a88473f`, `bfbb1fa`

### 11. Pose-Skeleton Letterbox-Fix ✅
- POSE_POSTPROCESS_FUNC: "filter"→"filter_letterbox"
- Skeleton war auf Y gestaucht (Schultern auf Huefthoehe)
- Jetzt korrekte Koerperproportionen
- Commit: `d573f44`

### 12. BBox Doppelte Letterbox — Ursache verstanden ✅
- Versuch "filter" statt "filter_letterbox" → GESCHEITERT (alles kaputt)
- Grund: hailoaggregator macht create_flattened_bbox() nochmal wenn scaling_bbox nicht cleared
- **Richtige Loesung:** _letterbox Postprocess BEHALTEN, BBox aus Landmarks ableiten
- Revert: `a396aa5`

---

## KRITISCH: SEGV Regel (IMMER GUELTIG!)
- bbox.ymin()/xmin() auf Pose-Detections → SEGV nach ~50s
- Gilt fuer _on_buffer UND _on_pre_overlay
- NIEMALS bbox.*() auf Detections mit HAILO_LANDMARKS
- Sicher: get_label(), get_confidence(), get_objects_typed()

---

## BEKANNTE OFFENE PROBLEME

### 1. Landmark-Flackern bei Kamerabewegung
- ST bewegt Kamera dauerhaft → Landmarks springen leicht
- Fix: Hysterese in ST/MOLOCH-Umschaltung (Face muss N Frames fehlen bevor ST uebernimmt)
- Status: NICHT IMPLEMENTIERT

### 2. Kamera Hot-Plug (aus CLAUDE.md)
- Stecker raus → nur Reboot hilft
- Status: OFFEN

### 3. ArcFace Threshold
- Aktuell 0.55, Similarity jetzt 0.78+ → Threshold KANN hoch auf 0.65+
- Aber: Neu-Enrollment empfohlen weil BBox-Berechnung sich geaendert hat
- Status: OFFEN (Threshold erhoehen nach Enrollment)

---

## AUFGABEN NAECHSTE SESSION

### PRIO 1: Landmark-Flackern reduzieren
- Hysterese: Face muss 5+ Frames fehlen bevor ST uebernimmt
- EMA-Smoothing auf Landmark-Positionen (nicht nur BBox-Center)
- Datei: core/mpo/autonomous_tracker.py (_should_moloch_track)

### PRIO 2: ArcFace Neu-Enrollment
- Alte Embeddings passen nicht optimal zur neuen landmark-basierten BBox
- IPC: enrollment_start mit ~20 Frames
- Danach Threshold auf 0.65+ hoch

### PRIO 3: Face-BBox Padding feintunen
- Aktuell: pad_top=-0.10, pad_bot=0.50, pad_x=0.25
- Evtl. noch anpassen nach visuellem Feedback

### PRIO 4: Verbleibende Gate-1 Tasks
- G1-T03: Auto-Resume aus Manuell + Spruch
- RTSP-Reconnect (Hot-Plug Bug)
- Dev-Tools: gst_lint.py, config_guard.py, valve_test.py, baseline_capture.py

---

## SYSTEM-ZUSTAND
FPS:            20 (stabil)
NPU RAM:        ~55MB / 8192MB (<1%)
ArcFace:        0.78+ Similarity (DEUTLICH BESSER als vorher 0.63)
Tracking:       Hybrid ST+MOLOCH, Error 6-24px
Pose:           Skeleton korrekt skaliert (Letterbox-Fix)
Face-Landmarks: Korrekt positioniert (Augen, Nase, Mund)
Face-BBox:      Aus Landmarks berechnet (kein Shrink-Hack mehr)
Smart-Tracking: Via eWeLink freigeschaltet, Hybrid-Logik aktiv

## GEAENDERTE DATEIEN DIESE SESSION
- core/hardware/camera.py (Gains, Smoothing, Home-Position)
- core/camera_manager.py (ST Toggle, Home-Default)
- core/gui/panel_models.py (FaceAttr, Pose FPS)
- core/gui/popups/popup_npu_thresh.py (Tooltip-Kontrast)
- core/mpo/autonomous_tracker.py (Gains, Hybrid-ST, STMovementLearner, Zentrierung)
- core/perception/tappas_pipeline.py (Face-BBox aus Landmarks, Pose Letterbox-Fix)
- config/camera_home.json (Tuer-Position)
