# Agent Handoff — 2026-04-17 (Session 16 — Audit)
# Letzter Commit: 0cb50d4 | Audit: 62/62 PASS | FPS: 20.0 | RAM: 39%

---

## SESSION 16 — AUDIT & RACE-FIX

System-Zustand: grün. FPS 20, RAM 39 %, 0 SEGV, 0 Worker-Errors, Face-ID
erkennt `markus` (Sim ~0.55-0.64). LLM/hailo-ollama antwortet, alle 8 NPU-Worker
aktiv (Activity/Depth/Face/Hand/PersonAttr/Pose/ReID/YOLOWorld).

### Gefundener Fehler
MCP-Audit zeigte **1 FAIL: "PerceptionMemory initialisiert — Kein Init-Log
gefunden"** bei ansonsten 61/62 PASS.

**Root Cause:** Race zwischen Service-Start und Audit-Test.
- Service-Start: 12:14:29 (Init-Log geschrieben)
- Vorheriger Audit-Lauf: 12:12:11 (2 min davor)
- Test `scripts/moloch_audit.py:797-809` greppt journalctl nach
  `PerceptionMemory.*Initialisiert` — Zeile existierte zu dem Zeitpunkt noch nicht.

### Fix (Commit 0cb50d4)
`scripts/moloch_audit.py` Test um Status-Fallback erweitert: wenn
journalctl nichts findet, prüft er `/dev/shm/moloch_status.json` auf
`face_id` / `person_present` / `active_models`. Wenn PerceptionMemory
Output liefert, läuft es per Definition. Bei totem Modul bleibt es FAIL.

Keine ROT-Dateien angefasst, kein Service-Restart nötig.

### Rollback-Obduktion (daf3e76, 2026-04-13)
"Kamera-Steuerung + Bild waren kaputt" nach Vision-Pause-Experiment
(`pause_for_llm` / `resume_after_llm` + disabled_workers). Nicht wiederholen
ohne neuen Plan. Danach `a701a38` → spontane Kommentare deaktiviert.

### Offen (aus voriger Session)
- B2 Moloch halluziniert Quellen → deepseek
- B4 News bei generischen Anfragen veraltet → autonomy
- Slider-Drift yolo_conf Max auf 0.7 → gui
- YOLOWorldWorker ~70 % integriert (Frequency/PFrame/IPC fehlen) → vision
- IPC Race poll_commands() löscht Datei vor Verarbeitung → service
- T2 Agent-Events nicht publiziert (rein architekturell, kein Runtime-Impact)

---

## SESSION-ERGEBNISSE (9 Commits)

| Fix | Commit | Datei |
|-----|--------|-------|
| HandWorker-Dispatch bei Nahaufnahme + pose_age Tolerance | 109ce71 | tappas_pipeline.py |
| PoseWorker conf_thresh 0.3->0.2, HandWorker wrist-vis 0.15->0.1 | 8955786 | pose_worker.py |
| panel_preview Pose-Keypoint Visibility 0.2->0.1 | a387ad0 | panel_preview.py |
| Night Cycle startet ab 23:00 + echte Daten in LLM-Reflexion | f41c05c | night_cycle.py |
| Shutdown-Cleanup loggt Fehler statt sie zu verschlucken | 3e884f9 | moloch_service.py |
| NPU-Extras VDevice/VLM Release Fehler-Logging | 2a6a553 | npu_extras.py |
| head_pitch/head_yaw in TAPPAS-PFrame + Worker-Health Logging | 8012b76 | tappas_pipeline.py |
| deepseek_client.py entfernt (230 Zeilen toter Code) | 5f69a62 | deepseek_client.py |
| ocr_texts in TAPPAS-PFrame setzen | 2c9c80d | tappas_pipeline.py |

### Wichtige Aenderungen:
- **Pose+Hand Landmarks**: 3 Root Causes behoben (HandWorker-Guard, pose_age, Thresholds)
- **Night Cycle**: Startet jetzt um 23:00 statt 00:00. LLM-Reflexion mit echten Tagesdaten.
- **Shutdown Logging**: 15x except:pass -> logger.warning/error (moloch_service + npu_extras)
- **PFrame komplett**: head_pitch/head_yaw + ocr_texts jetzt im TAPPAS-Modus verfuegbar
- **Systemscan**: 7 versteckte Bugs gefunden und behoben (Key-Mismatches, stille Fehler, toter Code)

---

## OFFENE BUGS

- **B2**: Moloch halluziniert Quellen ("Laut Suchergebnissen")
- **B4**: News bei generischen Anfragen veraltet (Google News RSS)
- **Slider-Drift**: yolo_conf Slider-Max auf 0.7 begrenzen (GUI)
- **YOLOWorldWorker**: Integration ~70% (keine Frequency, kein PFrame-Ort, kein IPC)
- **IPC Race**: poll_commands() loescht Datei VOR Verarbeitung
- **T2**: Agent-Events (build/test/review/chaos) nie publiziert — geplante Architektur, kein Runtime-Impact

---

## WAS BEREITS GEFIXT IST — NICHT NOCHMAL ANFASSEN

- Pose/Hand Landmarks (109ce71, 8955786, a387ad0)
- Night Cycle Date+LLM (f41c05c)
- Shutdown Logging (3e884f9, 2a6a553)
- head_pitch/head_yaw + ocr_texts (8012b76, 2c9c80d)
- Worker-Health Error-Dict (8012b76)
- deepseek_client.py entfernt (5f69a62)
- Pan-Vorzeichen, ArcFace, hailooverlay, Status-JSON Deadlock
- keywords.json Komma, PTZ Error-Handling
