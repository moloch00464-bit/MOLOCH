---
name: moloch-agent
description: Welcher MOLOCH-Agent fuer welche Aufgabe? Laedt den richtigen AGENT_*.md automatisch. Nutze IMMER bevor Du Code schreibst.
allowed-tools: Read Grep Glob Agent
---

# M.O.L.O.C.H. Agent-Auswahl

Bevor Du Code schreibst: Bestimme den Domain-Agenten und lies seine Datei.
**1 Aufgabe = 1 Agent. Kein Mix.**

---

## AGENT-MAPPING

| Aufgabe / Stichwort | Agent laden |
|---------------------|-------------|
| GStreamer, Pipeline, TAPPAS, Hailo NPU, HEF, Modell, Perception, FPS, BBox | `.claude/agents/vision.md` |
| PTZ, Tracking, Such-FSM, Arbiter, pan, tilt, FOLLOW, SEARCH, COAST | `.claude/agents/tracking.md` |
| ONVIF, RTSP, Kamera, eWeLink, Sonoff, LED, IR, Alarm, Fan, PWM | `.claude/agents/hardware.md` |
| Panel, Tkinter, GUI, Popup, panel_*.py, popup_*.py, Button, Label | `.claude/agents/gui.md` |
| Whisper, TTS, Piper, Spotify, Persoenlichkeit, Shadow, Guardian, Stimme | `.claude/agents/voice.md` |
| moloch_service, IPC, ServiceProxy, Memory, Qdrant, Integration | `.claude/agents/service.md` |
| Chaos, Stresstest, Absturz, Lasttest, Stabilitaet | `.claude/agents/stresstest.md` |
| ESP32, WiFi-Geraet, Peripherie, Firmware, Tentakel | `agents/AGENT_TENTACLE.md` |
| Strategie, Meta-Entscheidung, Philosophie, Priorisierung | `agents/AGENT_DEEPSEEK.md` |

---

## MEHRERE DOMAINS?

Wenn eine Aufgabe 2 Domains beruehrt:
- Immer den **primaeren Domain** waehlen (wo die Aenderung stattfindet)
- Sekundaeren Agent NUR lesen, nicht als Arbeits-Agent starten
- Beispiel: "Tracking-FPS im Panel anzeigen" → gui Agent (Panel ist das Ziel)

---

## TERRITORIUM (Dateizuordnung)

| Agent | Darf editieren |
|-------|---------------|
| vision | core/perception/*.py, core/inference_engine.py, core/model_orchestrator.py |
| tracking | core/mpo/*.py, core/ptz_tracker.py, core/ptz_arbiter.py, core/action_bridge.py |
| hardware | core/hardware/*.py, core/camera_manager.py |
| gui | core/gui/*.py, core/gui/popups/*.py |
| voice | core/speech/*.py, core/tts/*.py, core/audio/*.py, core/personality/*.py |
| service | core/moloch_service.py, core/ipc_router.py, core/memory/*.py, core/core_integrator.py |

**Cross-Domain-Edits = Abbruch + User fragen.**

---

## KOMMUNIKATION

```
~/moloch/logs/agent_handover.txt  — Uebergabe zwischen Sessions
~/moloch/logs/bug_report.txt      — Gefundene Bugs
~/moloch/logs/test_results.txt    — Testergebnisse
```
