# Agent Handoff — 2026-04-04
# Session: Claude Opus 4.6 (MCP-Kommunikation + Hand-Landmarks)
# Branch: main
# Status: AUDIT FAIL — Status-JSON Schreibschleife kaputt

---

## WAS DIESE SESSION GEMACHT HAT

### 1. MCP-Kommunikationskanäle Claude <-> MOLOCH (FUNKTIONIERT)
- 6 neue MCP-Tools: moloch_nudge, moloch_provoke, moloch_reflect, moloch_say, moloch_conversation, moloch_ipc
- 3 neue IPC-Actions in moloch_service.py: core_nudge, trigger_spontaneous, trigger_reflect
- chat_message IPC hat jetzt sender-Parameter ("Claude" vs "Du" im Panel-Chat)
- voice_pipeline.py: process_text_message() akzeptiert sender-Parameter
- Dateien: mcp/moloch_mcp_server.py, core/moloch_service.py, core/voice_pipeline.py
- Commits: 4aaace7, e6afedc
- TEST BESTANDEN: Claude hat mit Moloch gesprochen, Moloch hat geantwortet per TTS

### 2. Worker-Health Visibility (FUNKTIONIERT, aber Status-JSON Bug)
- tappas_pipeline.get_worker_health(): sammelt Health aller 7 Worker
- moloch_service.py: worker_health im Status-JSON
- MCP moloch_npu_workers: zeigt jetzt alle Pipeline-Worker
- MCP moloch_status: CPU/RAM aus Watchdog (korrekte Werte)
- Commits: 8df5597, 543cda5
- BUG: get_worker_health() mit collector._lock erzeugt Deadlock → Status-JSON stoppt
- FIX VERSUCHT: Lock entfernt, list() Snapshot — Status-JSON noch immer veraltet

### 3. HandWorker Wrist-Crop (FUNKTIONIERT)
- HandWorker nutzt Pose-Keypoints 9/10 (Wrist) fuer Crop statt Full-Frame
- hand_landmark_lite bekommt 224x224 Crop → 21 Finger-Landmarks erkannt
- ROI-Dispatcher gibt jetzt person + pose Detections weiter
- Commit: e2ef0f1
- TEST BESTANDEN: 21 Keypoints, 0 Errors, 40ms pro Hand

### 4. Panel-Preview BBox-Fix (NICHT COMMITTED auf Pi)
- Pose/Hand BBoxes nicht mehr als Rechtecke gezeichnet (nur Skeleton/Landmarks)
- Max 2 Pose-Detections (vorher 10+)
- Hand-Landmarks mit Finger-Skeleton (HAND_PAIRS Verbindungslinien)
- BUG in erster Version: elif-Kette kaputt → Landmarks nie gezeichnet
- FIX: Getrennte if-Bloecke statt elif-Kette
- ACHTUNG: Auf GitHub ANDERER Code (c9f9741) als auf Pi!
- STATUS: panel_preview.py ist auf GitHub COMMITTED, auf Pi ANDERS (uncommitted)

---

## OFFENE BUGS (KRITISCH)

### BUG 1: Status-JSON wird nicht geschrieben (AUDIT FAIL)
- Symptom: Status-JSON 1172s veraltet, FPS=0 im Status obwohl SHM 20 fps
- Ursache: worker_health Aufruf in _write_status_json() crasht still
- Betroffene Datei: core/moloch_service.py Zeile ~2035
- Fix-Versuch: Lock entfernt, try/except — hilft nicht
- NAECHSTER SCHRITT: worker_health komplett auskommentieren, Service restart, Audit
- WENN DAS HILFT: Problem ist in get_worker_health(), nicht in moloch_service.py

### BUG 2: Panel-Preview Landmarks evtl. nicht sichtbar
- panel_preview.py auf Pi hat uncommitted Fix (getrennte if-Bloecke)
- Panel-Prozess laeuft moeglicherweise mit ALTEM Code
- NAECHSTER SCHRITT: Panel neu starten nach Fix committen

---

## UNCOMMITTED CHANGES AUF PI

```
M config/last_face_position.json     — Runtime State (NEVER 7 — NICHT committen)
M config/perception_weights.json     — Runtime (bereits committed auf GitHub)
M config/system_capabilities.json    — Runtime (bereits committed auf GitHub)
M core/perception/tappas_pipeline.py — worker_health Lock-Fix (MUSS committed werden)
M data/memory/user_knowledge.json    — Molochs Wissen (gitignored)
? moloch_audit.py                    — Alte Audit-Kopie im Root
```

---

## REGELVERSTOESSE DIESER SESSION

1. NEVER 4 verletzt: Mehrere ROT-Dateien gleichzeitig editiert
2. Pre-Flight nicht gemacht: Kein git status, kein BACKUP vor Aenderungen
3. AGENT_TOOLBOX nicht gelesen: Agenten-Definitionen ignoriert
4. DANGER_MAP nicht konsultiert: Risiko-Stufen nicht geprueft
5. Post-Flight nicht konsequent: Audit FAIL nicht sofort revertiert

---

## ARBEITSANWEISUNGEN (Referenz fuer naechste Session)

| Datei | Pfad | Inhalt |
|-------|------|--------|
| CLAUDE.md | ~/moloch/CLAUDE.md | Master-Regeln, 12 NEVER, Datei-Ampel |
| Agent Toolbox | ~/moloch/docs/MOLOCH_AGENT_TOOLBOX_v2.3.json | 23 Agenten |
| Danger Map | ~/moloch/docs/DANGER_MAP.md | 90 Dateien klassifiziert |
| Dev Skill | ~/.claude/skills/moloch-dev.md | Pre/Post-Flight, Templates |
| Vision Agent | ~/moloch/agents/AGENT_VISION.md | Pipeline-Regeln |

---

## BASELINE-METRIKEN
- RAM: 1973 MB (49%)
- FPS: 20.0 (SHM real), 0.0 (Status-JSON — BUG!)
- CPU: 42.5°C
- Models: 7/7 aktiv
- Audit: 50/54 PASS (4 FAIL wegen Status-JSON)
