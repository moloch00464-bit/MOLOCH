# M.O.L.O.C.H. — LIVE-STATE BRIEFING für Claude Chat
# Stand: 2026-04-06 | Zum Copy-Pasten in Claude Chat

---

## SYSTEM-STATUS JETZT

```
FPS:        18.8 (Pipeline läuft stabil)
RAM:        42% von 4 GB
CPU-Temp:   44.6°C
Frame Age:  0.0s (Kamera live)
Face-ID:    markus (Confidence 0.84)
Szenario:   NAH
Tracker:    frozen (PTZ-Modus unklar)
```

**Alle 7 Worker aktiv, 0 Errors:**
```
ActivityWorker  : 88 Inferences
FaceWorker      : 1283 Inferences
HandWorker      : 660 Inferences
PersonAttrWorker: 442 Inferences
PoseWorker      : 872 Inferences
ReIDWorker      : 531 Inferences
YOLOWorldWorker : 44 Inferences

SuperRes + LowLight: nicht geladen (normal — on-demand)
ROI Dispatcher  : 2656 Frames, 54 Dropped
```

---

## LETZTE 5 COMMITS

| Hash | Was |
|------|-----|
| e50b8fa | fix: Pose-BBox + Hand-BBox Rechteck in panel_preview.py |
| c47b768 | feat: zweiten MCP-Server moloch-unconscious in .mcp.json |
| b1cb3a1 | feat: moloch_unconscious_mcp.py — 2. MCP-Server für Unterbewusstsein |
| 47a3144 | fix: model_scheduler face_detected Fallback → NAH→IDLE Oszillation behoben |
| 508d57c | fix: MCP-Server SSH → stdio Transport (lokal auf Pi) |

---

## OFFENE BUGS (priorisiert)

### PRIO 1 — FPS-Reporting-Bug (ROT: tappas_pipeline.py)
- **Symptom:** status.json zeigt manchmal FPS=0.2, echter Durchsatz ~20 FPS
- **Ursache:** FPS-Zähler in `_on_buffer` wird nicht korrekt aktualisiert
- **Folge:** UnconsciousEngine reagiert auf falsche FPS ("reduce" Mood ständig)
- **Zuständig:** vision-Agent (`tappas_pipeline.py`)

### PRIO 2 — Hand Landmarks im Panel fehlen
- Hängt möglicherweise am FPS-Bug (ROI Dispatcher bekommt scheinbar wenig Frames)
- Nach FPS-Fix prüfen ob Hand-LM auftauchen
- **Zuständig:** vision (`roi_dispatcher.py`) oder gui (`panel_preview.py`)

### PRIO 3 — ArcFace Similarity zu niedrig (0.37, Threshold 0.65)
- Face-ID zeigt "unbekannt" obwohl Markus im Bild
- Neu-Enrollment nötig: `scripts/enroll_face_worker.py`
- **Zuständig:** vision-Agent

### PRIO 4 — Tracker STUCK-AT-LIMIT
- pos=(-88.0,+15.9) = mechanischer Anschlag, state=frozen (kein SEARCH)
- `core/mpo/autonomous_tracker.py` → `_track_tracking_target()`
- **Zuständig:** tracking-Agent

### PRIO 5 — moloch_unconscious IPC Handler fehlt
- `uc_inject_impulse` schreibt via IPC, aber moloch_service.py kennt den
  "unconscious_impulse" Action-Handler noch nicht → Impulse werden still ignoriert
- **Zuständig:** service-Agent (`moloch_service.py`)

---

## WAS ZULETZT GEBAUT WURDE (diese Session)

1. **MCP stdio-Transport** — .mcp.json nutzte SSH (Pi SSH'd zu sich selbst → Fehler). Fix: stdio direkt.
2. **NAH→IDLE Oszillation** — YOLO-aus → person_count=0 → IDLE → Loop alle 3s. Fix: 2 Zeilen in model_scheduler.py
3. **Unterbewusstsein-MCP-Server** — neuer zweiter Server `moloch_unconscious_mcp.py` mit 5 Tools
4. **Pose-BBox + Hand-BBox** — Rechteck-Zeichnen in panel_preview.py gefixt

---

## NEUE KOMPONENTE: Unterbewusstsein-MCP-Server

**Datei:** `mcp/moloch_unconscious_mcp.py`
**Tools:** `uc_get_state`, `uc_get_mood`, `uc_get_history`, `uc_inject_impulse`, `uc_reflect`
**Liest:** `/dev/shm/moloch_status.json` + `moloch_impulse.json`
**Schreibt:** via IPC (atomic `moloch_cmd_NNNN.json`)
**Status:** Server registriert, IPC-Handler in moloch_service.py fehlt noch (PRIO 5)

---

## NÄCHSTE SESSION STARTEN MIT

1. `moloch_status()` — FPS, RAM, Face-ID prüfen
2. `moloch_npu_workers()` — Worker-Health prüfen
3. `git status` — muss clean sein
4. `logs/agent_handoff.md` lesen — offene Bugs kennen
5. Dann mit PRIO 1 anfangen (FPS-Reporting-Bug)

---

## LOKOMOTIVE-ERINNERUNG

- Antworte auf jeden Coding-Auftrag mit: **LOKOMOTIVE aktiv.**
- ROT-Dateien: einmal ankündigen, Git Backup, dann durcharbeiten
- NEVER 2: `pan_delta = -error_x` — NICHT anfassen
- NEVER 1: GStreamer-String NICHT blind ändern (→ SEGV)
- Nach jeder Änderung: `python3 ~/moloch/moloch_audit.py --auto` — bei FAIL sofort STOPP
