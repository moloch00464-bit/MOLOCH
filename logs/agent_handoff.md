# Agent Handoff — Session 2026-03-30 (BBox-Overlay + Tracker-Fix)
**Datum:** 2026-03-30 | **Reboot:** gerade ausgelöst, noch nicht abgeschlossen

---

## Was in dieser Sitzung erledigt wurde

### ✅ BBox-Overlay im Panel (kein hailooverlay mehr nötig)
- `moloch_service.py`: `panel_detections` in `moloch_status.json` eingefügt
  - Alle Detektionen aus `get_detections()`: normalisierte BBoxen [0-1], `face_id`, `face_similarity`
- `panel_preview.py`: PIL `ImageDraw` zeichnet BBoxen direkt auf den Frame
  - Face erkannt (face_id) = **Cyan**, Face unbekannt = **Gelb**, Person = **Grün**
  - Label: Name + Similarity (z.B. "markus 0.72") oder "face 0.77"
  - In `except`-Block gewickelt → Preview-Freeze unmöglich
- Commit: `ef4d1a1`

### ✅ Tracker Stuck-at-Limit Fallback
- `autonomous_tracker.py`: neue Erkennung in `_track_tracking_target()`
- Bedingung: Kamera ≥ 8s am mechanischen Pan-/Tilt-Anschlag UND Error treibt weiter rein
- Aktion: EMA-Filter (`_smooth_x`, `_smooth_y`) zurücksetzen + `SEARCHING` starten
- Verhindert: Kamera dreht sich in Ecke durch Artefakt-Detection (Wand/Decke als Gesicht)
- Commit: `8be3a67`

---

## Diagnose: Warum war Kamera in der Ecke?

```
Tracker war bei pos=(-166.4, +76.7)deg — mechanischer Anschlag beider Achsen
Error konstant (+291, -149)px — nie konvergiert
Ursache: False-Positive Detection (Wand/Decke/Fenster als Gesicht erkannt)
Face-Conf: 0.77 (konstant) → SCRFD hat statisches Artefakt als Face klassifiziert
Tracker konnte nicht erkennen dass er am Limit feststeckte → fuhr immer weiter rein
Fix: STUCK-AT-LIMIT nach 8s → SEARCH Mode
```

---

## System-Stand nach Reboot

```
Branch:      main (Commit 8be3a67)
Reboot:      gerade ausgelöst — noch nicht abgeschlossen
FPS:         war 19.9 vor Reboot
SEGV:        0 seit letztem Reboot
hailooverlay: ENTFERNT aus Pipeline (blockierte SHM nach ~75s)
BBox-Overlay: PIL-basiert aus Status-JSON (panel_detections)
Pose:        AUS (Valve zu, Modell im RAM)
ReID:        AUS
Smart-Track: Permanent AUS
```

---

## Sofort nach Reboot prüfen

1. `mcp moloch_status` → FPS=20, Frame Age <5s
2. `mcp moloch_logs` → keine SEGV, keine CrashLoops
3. **Visuell**: BBoxen im Panel? (Cyan/Gelb für Gesicht, Grün für Person)
4. **Kamera**: Startet normal (nicht in Ecke)?
5. Falls Stuck-Limit aktiv: `[STUCK-LIMIT]` in Logs sichtbar

---

## Offene Bugs — NÄCHSTE SITZUNG

### 🔴 ArcFace Similarity niedrig (0.14–0.29)
- Sollte ≥ 0.65 für Markus-Erkennung
- **Diagnose-Idee**: Snapshot wenn Face im Bild → prüfen ob Face-Crop (112x112) plausibel ist
- Mögliche Ursache: SCRFD-BBox korrekt aber ArcFace-Crop falsch zugeschnitten nach Pipeline-Refactor
- Face-DB: mit gleicher GStreamer-Pipeline trainiert (sollte kompatibel sein)
- Zu prüfen: `scrfd_10g_letterbox` Postprocess → liefert korrekte 112x112 Crops für ArcFace?

### 🟡 ArcFace Re-Enrollment nötig?
- Wenn Similarity dauerhaft <0.3 nach Diagnose → Face-DB neu aufbauen
- Enrollment NUR via IPC `enrollment_start` (CLAUDE.md Regel 11 — NIEMALS offline)

### 🟡 Voice-Zentralisierung
- `moloch_console.py` ruft TTS direkt statt über `personality_engine.speak()`
- Memory: `feedback_one_voice.md` — Agent 5 (Voice) zuständig

### 🟡 PaddleOCR Integration
- HEFs: `/mnt/moloch-data/hailo/models/zoo/ocr/`
- Pipeline-Code vorhanden (mit Valves) → muss auf "permanent AN" umgebaut werden
- Niedrige Priorität

### 🟡 Pose + ReID reaktivieren
- Pose: hailooverlay SEGV (Race mit NPU-Shared-Memory in C-Code)
- ReID: libre_id.so crasht bei leeren BBoxen
- Beide Modelle im RAM (Valves zu) — warten auf Root-Cause

---

## Start-Checklist neue Sitzung

1. `CLAUDE.md` lesen
2. Diese Handoff-Datei lesen
3. MCP Status → FPS=20, BBoxen sichtbar?
4. **Ziel 1:** ArcFace Similarity debuggen (Face-Crop prüfen)
5. **Ziel 2:** Ggf. Re-Enrollment via IPC
6. **Ziel 3:** Voice-Zentralisierung (Agent 5)

---

## Architektur-Notizen (stabil)

```
GStreamer Pipeline (hailooverlay ENTFERNT):
  rtspsrc → hailosrc → [YOLO|SCRFD|ArcFace|FaceAttr] → hailotracker → appsink

BBox-Flow NEU (kein hailooverlay):
  _on_buffer (Hailo API) → _detections[] (normalisiert [0-1])
  → panel_detections in Status-JSON (via _write_status_json)
  → panel_preview.py PIL.ImageDraw.rectangle()

Tracker-Feed (5 Hz):
  get_detections() → pixel_bbox (×1280/720) → tracker.update_detection()
  → _track_tracking_target() → AbsoluteMove
  NEU: STUCK-AT-LIMIT Erkennung nach 8s

Valves: EINMALIG beim Start gesetzt, danach nie mehr geschaltet.
```
