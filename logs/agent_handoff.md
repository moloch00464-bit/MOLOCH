# M.O.L.O.C.H. Übergabeprotokoll
**Datum:** 2026-03-28, 16:40 CET
**Von:** Claude Opus 4.6 — Mega-Session: Tracking + BBox + Landmarks + NPU
**Service-Status:** LAEUFT (20 FPS, ArcFace 0.78+ @thresh=0.65, stabil)
**USE_TAPPAS:** 1 (aktiv)

---

## ERLEDIGT IN DIESER SESSION (16 Aenderungen)

### Tracking-System komplett ueberarbeitet
1. **Gain-Tuning** — camera.py: 0.7→0.4, MaxStep 30→15, 3-Frame BBox-Smoothing
2. **Aggressives Tracking** — autonomous_tracker.py: pan_gain 0.65, tilt_gain 0.50, max_step 25/18, Speed 1.0, Cooldown 50ms, EMA 0.70
3. **Dead-Zone Ping-Pong behoben** — frozen 30→20px, coast 40→25px, resume 35→50px (vorher nur 5px Differenz)
4. **Gesicht-Zentrierung** — Y-Referenz 0.33→0.40 (naeher an Bildmitte)
5. **Park-Position = Tuer** — camera_home.json + Defaults: Pan 50, Tilt -20

### Hybrid-Tracking (MOLOCH + Kamera Smart-Tracking)
6. **Smart-Tracking Freigabe** — camera_manager.py Gate-0-Lock entfernt, eWeLink Cloud-API
7. **Hybride Logik** — Kein Face → ST bleibt an (Sensoren schneller). Face → MOLOCH uebernimmt
8. **Auto-ST** — Wenn MOLOCH BBox 20 Cycles >35% off-center → ST einschalten
9. **STMovementLearner** — Rohpositionen + Velocity-Aufzeichnung (deg/s)

### BBox + Landmarks (DURCHBRUCH)
10. **Face-BBox aus Landmarks** — SCRFD 5-Punkt Landmarks sind korrekt, BBox wird daraus berechnet. Alter SHRINK_Y=0.50 Hack ersetzt. Padding: pad_x=0.25, pad_top=-0.10, pad_bot=0.50
11. **Landmark-Transformation** — Nach BBox-Resize werden Landmarks korrekt umgerechnet (bbox-relativ → Frame → neue bbox-relativ)
12. **Pose-Skeleton Fix** — POSE_POSTPROCESS_FUNC: "filter"→"filter_letterbox", Skeleton war Y-gestaucht

### NPU + Panel
13. **ArcFace Threshold** — settings.json: 0.55→0.65 (Similarity jetzt 0.78-0.93)
14. **ArcFace Enrollment** — Live-Enrollment 15 Frames frontal, 0.93 Similarity
15. **Panel FaceAttr** — TAPPAS_MODELS + status_key_map + FPS-Detail
16. **NPU Popup korrigiert** — Modelle zeigen echten Status (AKTIV/INAKTIV)

---

## KRITISCHE REGELN (IMMER GUELTIG!)

### SEGV-Regel
- `bbox.ymin()/xmin()` auf Pose-Detections → SEGV nach ~50s
- NIEMALS bbox.*() auf Detections mit HAILO_LANDMARKS
- Sicher: get_label(), get_confidence(), get_objects_typed()

### Valve-Crash (cv2::resize)
- ReID + Hand Valve kann NICHT zur Laufzeit geoeffnet werden
- Crash: `cv2::resize Assertion failed !ssize.empty()` in kompiliertem SO
- Betrifft: librepvgg_reid_postprocess.so, hand_landmark SO
- Ursache: SO bekommt leeren Frame beim Valve-Wechsel
- **reid_needed = False, hand_needed = False** in tappas_pipeline.py ~Zeile 1569
- Fix erfordert: Pipeline-String Aenderung (videoscale vor SO) oder TAPPAS-SO Update

### BBox Doppelte Letterbox
- filter_letterbox/scrfd_10g_letterbox MUESSEN bleiben (entfernen = alles kaputt)
- hailocropper internal-offset=true MUSS bleiben
- Face-BBox wird in Python aus Landmarks berechnet (kompensiert Doppelkorrektur)
- Pose nutzt filter_letterbox (korrigiert Skeleton-Skalierung)

---

## OFFENE PROBLEME

### 1. ReID + Hand Valve-Crash (BLOCKER fuer diese Modelle)
- HEFs geladen im NPU-RAM, Valves geschlossen
- Auch permanentes Oeffnen crasht (nicht nur dynamisches Schalten)
- Moegliche Loesungen:
  a) videoscale/videoconvert VOR dem SO im Pipeline-String
  b) Initial-Frame durch Valve schicken bevor SO aktiv wird
  c) TAPPAS SO Update abwarten
- Dateien: core/perception/tappas_pipeline.py (Pipeline-String ~Zeile 1150-1170)

### 2. Landmark-Flackern bei Kamerabewegung
- ST bewegt Kamera → Landmarks springen
- Fix: Hysterese in ST/MOLOCH-Umschaltung (Face muss N Frames fehlen)
- Datei: core/mpo/autonomous_tracker.py (_should_moloch_track)

### 3. Face-BBox Padding Feintuning
- Aktuell: pad_x=0.25, pad_top=-0.10, pad_bot=0.50
- Evtl. noch anpassen nach visuellem Feedback
- Datei: core/perception/tappas_pipeline.py ~Zeile 1435

### 4. ArcFace Instabilitaet bei Kamerabewegung
- Similarity schwankt 0.43-0.93 je nach Kamera-Winkel/Bewegung
- Frontal: 0.93, seitlich/bewegt: 0.43
- Fix: Mehr Enrollment-Winkel ODER Threshold auf 0.50 senken

---

## SYSTEM-ZUSTAND

```
FPS:              20 (stabil)
NPU RAM:          ~55MB / 8192MB (<1%)
Aktive Modelle:   SCRFD, ArcFace, YOLOv8m, FaceAttr, Pose (5 von 7)
Inaktive Modelle: Person-ReID, Hand Landmark (Valve-Crash)
ArcFace:          0.78+ Similarity @thresh=0.65
Tracking:         Hybrid ST+MOLOCH, Error 6-24px
Smart-Tracking:   Via eWeLink freigeschaltet
Pose-Skeleton:    Korrekt skaliert (Letterbox-Fix)
Face-Landmarks:   Korrekt positioniert (Augen, Nase, Mund)
Face-BBox:        Aus Landmarks berechnet
```

## GEAENDERTE DATEIEN

| Datei | Aenderungen |
|-------|-------------|
| core/hardware/camera.py | Gains, Smoothing, Home-Position |
| core/camera_manager.py | ST Toggle freigeschaltet, Home-Default |
| core/gui/panel_models.py | FaceAttr, Pose FPS |
| core/gui/popups/popup_npu_thresh.py | Modelle aktualisiert, ehrlicher Status |
| core/mpo/autonomous_tracker.py | Gains, Hybrid-ST, Learner, Zentrierung, Dead-Zone |
| core/perception/tappas_pipeline.py | Face-BBox aus Landmarks, Pose Letterbox, Valve-Status |
| core/model_orchestrator.py | person_reid in active_map |
| core/moloch_service.py | person_reid_active im Status-Dict |
| config/camera_home.json | Tuer-Position (50, -20) |
| config/settings.json | ArcFace Threshold 0.65 |
