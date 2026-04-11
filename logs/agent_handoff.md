# Agent Handoff — 2026-04-11 (Session 15)
# Letzter Commit: 2c9c80d | Audit: 62/62 PASS | FPS: 20.0 | RAM: 54%

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
