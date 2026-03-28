# M.O.L.O.C.H. Übergabeprotokoll
**Datum:** 2026-03-28, 13:10 CET
**Von:** Claude Opus 4.6 Session (Face-BBox + Pose + PerceptionMemory + Gate 6 + Audit v3)
**Service-Status:** LAEUFT (20 FPS, stabil, 39/39 Audit PASS)
**USE_TAPPAS:** 1 (aktiv)

---

## Aktueller Code-Stand
- **Branch**: main, Commit `f24285a`
- **tappas_pipeline.py**: STABIL — 0 Crashes, Face+Pose+Scheduler aktiv
- **temporal_memory.py**: NEU — PerceptionMemory + RoutineTracker (Gate 6)
- **moloch_audit.py**: v3.0 — 39 Tests (5 neue Sektionen)
- **Backup SSD2**: `/mnt/moloch-data/backups/moloch_20260328_084744`

## KRITISCH — SEGV Regel (BEIDE Probes betroffen!)
- `bbox.ymin()` auf Pose-Detections → SEGV nach ~50s
- Gilt fuer `_on_buffer` UND `_on_pre_overlay`!
- NIEMALS `bbox.*()` auf Detections mit HAILO_LANDMARKS
- `get_label()`, `get_confidence()`, `get_objects_typed()` sind SICHER

## Erledigt — Diese Session

### 1. Face-BBox Letterbox-Fix
- `FACE_BBOX_SHRINK_X=1.0, SHRINK_Y=0.50, ANCHOR_BOTTOM=1.0`
- SCRFD-Landmarks werden NICHT umgerechnet

### 2. Pose-Modell aktiviert
- Root Cause: `output-format-type=HAILO_FORMAT_TYPE_FLOAT32` entfernt
- Doppelte Dequantisierung → 0 Detections → jetzt 17 Keypoints korrekt

### 3. Pose-Landmarks korrekt
- YOLO-Person entfernen, Pose-Person behalten (Landmarks BBox-relativ)

### 4. SEGV durch Pose-BBox Zugriff gefixt
- Auch _on_pre_overlay ist NICHT sicher fuer bbox-Methoden auf Pose

### 5. PerceptionMemory System (Gate 5.1 / ChatGPT Vision)
- EntityTracker: Familiarity/Stability/Motion
- AttentionMap: 8x8 Spatial Grid
- SmoothedState: Scheduler-Glaettung (kein Flattern)
- Alles in `core/perception/temporal_memory.py`

### 6. Gate 6: RoutineTracker
- Lernt Tageszeit-Muster (Anwesenheit, Bewegung, Person)
- Anomalie-Erkennung (ab 3 Tagen Daten)
- Persistent: `/mnt/moloch-data/memory/routines.json`

### 7. Audit v3.0
- 5 neue Sektionen: TAPPAS, PerceptionMemory, Modelle, Panel, Faehigkeiten
- 39/39 PASS

### 8. Gate 6-10 Roadmap
- Gate 6: Temporale Intelligenz (RoutineTracker IMPLEMENTIERT)
- Gate 7: Lokales LLM (hailo-ollama, Nacht-Reflexion)
- Gate 8: Raeumliche Intelligenz (Segmentierung, 3D-Raumkarte)
- Gate 9: Tentakel-Netzwerk (WLED, MQTT, HA)
- Gate 10: Emergente Autonomie (Selbst-Lernen)
- Gespeichert: `docs/GATE_6_10_ROADMAP.md`

## Offene Punkte
1. Person-BBox zu gross (YOLO) — Pose-BBox SEGV-unsicher, Y-Shrink als Alternative
2. Pose-Landmarks Skalierung bei Aufstehen/Entfernen
3. Hand/ReID Valves deaktiviert (cv2::resize Crash)
4. NPU Load meldet 0.0 an CoreIntegrator
5. Tracking-Suchgeschwindigkeit zu langsam
6. Gate 7-10 implementieren

## Gate-Status Komplett
| Gate | Status | Module |
|------|--------|--------|
| 0 | ✅ PASS | Systemschliessung |
| 0.5 | ✅ PASS | TAPPAS Pipeline |
| 1 | ✅ AKTIV | Action Bridge + Event Bus |
| 2 | ✅ AKTIV | Memory (5 Module) |
| 3 | ✅ AKTIV | Awareness (4 Module) |
| 4 | ✅ AKTIV | Personality (4 Module) |
| 5 | ✅ AKTIV | Autonomy (4 Module) |
| 5.1 | ✅ NEU | PerceptionMemory |
| 6 | ✅ NEU | RoutineTracker |
| 7-10 | 📋 GEPLANT | Roadmap geschrieben |
