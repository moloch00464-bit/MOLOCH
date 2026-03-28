# M.O.L.O.C.H. Übergabeprotokoll
**Datum:** 2026-03-28, 11:15 CET
**Von:** Claude Opus 4.6 Session (BBox-Analyse + SEGV-Fix + System-Audit)
**Service-Status:** LAEUFT (20 FPS, stabil)
**USE_TAPPAS:** 1 (aktiv)

---

## Aktueller Code-Stand
- **Branch**: main, Commit `ee47bf7`
- **tappas_pipeline.py**: STABIL auf Stand von `38648c2` (NICHT veraendern ohne Grund!)
- **Backup SSD2**: `/mnt/moloch-data/backups/moloch_20260328_084744`

## KRITISCH — SEGV Root Cause (heute gefunden)
- `bbox.ymin()` auf Pose-Detections aufrufen → SEGV nach ~50 Sekunden
- 16 Crashes heute, alle 43-70s nach Start, deterministisch
- **REGEL**: NIEMALS `bbox.ymin()/ymax()/xmin()/xmax()` auf Detections mit HAILO_LANDMARKS
- Pose-Detections NUR ueber `det.get_objects_typed(hailo.HAILO_LANDMARKS)` filtern + `continue`

## OFFEN — Face-BBox zu gross (Hauptaufgabe fuer naechste Session)

### Problem
SCRFD Face-BBox ist ~25-30% groesser als Gesicht. Ursache: Doppelte Letterbox-Korrektur.

### Getestet und GESCHEITERT
1. SO-Funktion `scrfd_10g` (ohne letterbox) → Landmarks kaputt
2. `internal-offset=false` → kein Unterschied
3. BBox in Probe schrumpfen → Landmarks verschieben sich
4. `bbox.ymin()` in _on_buffer → SEGV

### Naechster Ansatz (ungetestet, API verifiziert)
Face-BBox schrumpfen UND Landmark-Punkte mitrechnen:
```python
# In _on_pre_overlay (stabil, laeuft seit Wochen):
bbox = det.get_bbox()
shrink = 0.85
cx, cy = bbox.xmin() + bbox.width()*0.5, bbox.ymin() + bbox.height()*0.5
nw, nh = bbox.width()*shrink, bbox.height()*shrink
new_bbox = hailo.HailoBBox(cx-nw*0.5, cy-nh*0.5, nw, nh)
new_det = hailo.HailoDetection(new_bbox, "face", det.get_confidence())

# Landmarks korrigieren (relativ zur neuen BBox):
for sub in det.get_objects():
    if hasattr(sub, 'get_points'):  # HailoLandmarks
        pts = sub.get_points()
        offset = (1.0 - shrink) / (2.0 * shrink)
        new_pts = [hailo.HailoPoint((p.x() - offset*...) , ..., p.confidence()) for p in pts]
        sub.set_points(new_pts)
    new_det.add_object(sub)
```

## Weitere offene Punkte
- Pose Landmarks verstreut (POSE_POSTPROCESS_FUNC="filter" statt "filter_letterbox")
- Hand/ReID Valves deaktiviert
- NPU Load meldet 0.0 an CoreIntegrator
- Scheduler flattert NAH→IDLE→NAH
- Tracking-Suchgeschwindigkeit zu langsam
- hailo-ollama fuer Gate 5.1 (Qwen2.5-1.5B)

## Neue Dateien (heute erstellt)
- `.claude/skills/moloch-dev.md` — Skill mit NEVER-DO Regeln
- `docs/DANGER_MAP.md` — Datei-Risiko-Karte
- `docs/TOOLS_PLAN.md` — 8 geplante Scripts
- `scripts/preflight.py`, `postflight.py`, `smoke_test.py`, `danger_check.py`
