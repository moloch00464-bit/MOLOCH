---
name: moloch-agent
description: Welcher MOLOCH-Agent fuer welche Aufgabe? Laedt den richtigen AGENT_*.md automatisch. Nutze IMMER bevor Du Code schreibst.
---

# M.O.L.O.C.H. Agent-Auswahl

Bevor Du Code schreibst: Bestimme den Domain-Agenten und lies seine Datei.
**1 Aufgabe = 1 Agent. Kein Mix.**

---

## AGENT-MAPPING

| Aufgabe / Stichwort | Agent laden |
|---------------------|-------------|
| GStreamer, Pipeline, TAPPAS, Hailo NPU, HEF, Modell, tappas_pipeline, Perception, FPS, BBox, Valve | `agents/AGENT_VISION.md` |
| PTZ, Tracking, Such-FSM, Arbiter, tracker, pan, tilt, FOLLOW, SEARCH, COAST | `agents/AGENT_TRACKING.md` |
| ONVIF, RTSP, Kamera, eWeLink, Sonoff, LED, IR, Alarm, Thermik, Fan, PWM | `agents/AGENT_HARDWARE.md` |
| Panel, Tkinter, GUI, Popup, panel_*.py, popup_*.py, Button, Label, tk.after | `agents/AGENT_GUI.md` |
| Whisper, TTS, Piper, Spotify, Persoenlichkeit, Shadow, Guardian, Tension, Stimme, Sprache | `agents/AGENT_VOICE.md` |
| moloch_service, IPC, ServiceProxy, Memory, Qdrant, Langzeit, Integration, Heartbeat | `agents/AGENT_SERVICE.md` |
| Chaos, Stresstest, Absturz, Lasttest, 8 Szenarien, Stabilitaet | `agents/AGENT_STRESSTEST.md` |
| ESP32, WiFi-Geraet, Peripherie, Firmware, Tentakel | `agents/AGENT_TENTACLE.md` |
| Strategie, Meta-Entscheidung, Philosophie, Was soll ich tun?, Priorisierung | `agents/AGENT_DEEPSEEK.md` |

---

## MEHRERE DOMAINS?

Wenn eine Aufgabe 2 Domains beruehrt:
- Immer den **primaeren Domain** waehlen (wo die Aenderung stattfindet)
- Sekundaeren Agent NUR lesen, nicht als Arbeits-Agent starten
- Beispiel: "Tracking-FPS im Panel anzeigen" → AGENT_GUI (Panel ist das Ziel) + AGENT_TRACKING lesen

---

## INSTANZ STARTEN

```
Lies ~/moloch/CLAUDE.md und ~/moloch/agents/AGENT_[DOMAIN].md.
Aufgabe: [Beschreibung]
```

---

## TERRITORIUM (Dateizuordnung)

| Agent | Darf editieren |
|-------|---------------|
| VISION | core/perception/*.py, core/inference_engine.py, core/model_orchestrator.py |
| TRACKING | core/mpo/*.py, core/ptz_tracker.py, core/ptz_arbiter.py, core/action_bridge.py |
| HARDWARE | core/hardware/*.py, core/camera_manager.py |
| GUI | core/gui/*.py, core/gui/popups/*.py |
| VOICE | core/speech/*.py, core/tts/*.py, core/audio/*.py, core/personality/*.py, core/console/moloch_console.py |
| SERVICE | core/moloch_service.py, core/ipc_router.py, core/memory/*.py, core/core_integrator.py |

**Agenten editieren NUR ihr Territorium. Cross-Domain-Edits = Abbruch + User fragen.**

---

## KOMMUNIKATION

```
~/moloch/logs/agent_handover.txt  — Uebergabe zwischen Sessions
~/moloch/logs/bug_report.txt      — Gefundene Bugs
~/moloch/logs/test_results.txt    — Testergebnisse
```
