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
| GStreamer, Pipeline, TAPPAS, Hailo NPU, HEF, Modell, Perception, FPS, BBox-Inferenz | `.claude/agents/vision.md` |
| PTZ, Tracking, Such-FSM, Arbiter, pan, tilt, FOLLOW, SEARCH, COAST | `.claude/agents/tracking.md` |
| ONVIF, RTSP, Kamera, eWeLink, Sonoff, LED, IR, Alarm, Fan, PWM | `.claude/agents/hardware.md` |
| Panel, Tkinter, GUI, Popup, panel_*.py, popup_*.py, Button, Label, BBox-Anzeige, Landmarks | `.claude/agents/gui.md` |
| Whisper, TTS, Piper, Stimme, Audio-Pipeline, Sprach-I/O | `.claude/agents/voice.md` |
| moloch_service, IPC, ServiceProxy, core_integrator | `.claude/agents/service.md` |
| PersonalityEngine, Mood, Tension, Shadow, Guardian, Berserker, EventBus | `.claude/agents/personality.md` |
| DecisionEngine, Homeostasis, LLM-Bridge, Night Cycle, Atmosphere | `.claude/agents/autonomy.md` |
| Activity, Context, Motion, RoomMap, WorldState, Situationsbewusstsein | `.claude/agents/awareness.md` |
| Episodic, Persistent, Vector, ReID, Langzeitgedaechtnis, Qdrant | `.claude/agents/memory.md` |
| SystemWatchdog, Diagnostics, CapabilityMonitor, System-Health | `.claude/agents/watchdog.md` |
| Spotify, Track-Index, MusicMemory, Zone-Musik | `.claude/agents/music.md` |
| hailo-ollama, DeepSeek, LLM-Bridge, Meta-Entscheidung, Philosophie | `.claude/agents/deepseek.md` |
| ESP32, WiFi-Mic, ReSpeaker, Firmware, Peripherie, Tentakel | `.claude/agents/tentacle.md` |
| TaoEngine, Unterbewusstsein, Mood-Impulse, Self-Tune, Anima | `.claude/agents/unconscious.md` |
| Chaos, Stresstest, Absturz, Lasttest, Stabilitaet | `.claude/agents/stresstest.md` |

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
| hardware | core/hardware/*.py, core/camera_manager.py (NICHT wifi_mic.py!) |
| gui | core/gui/*.py, core/gui/popups/*.py |
| voice | core/speech/*.py, core/tts/*.py, core/audio/*.py |
| service | core/moloch_service.py, core/ipc_router.py, core/core_integrator.py |
| personality | core/personality/*.py, core/event_bus.py |
| autonomy | core/autonomy/*.py |
| awareness | core/awareness/*.py, core/world_state.py |
| memory | core/memory/*.py, core/longterm_memory.py, core/daily_learner.py |
| watchdog | core/system_watchdog.py, core/diagnostics.py, core/capability_monitor.py |
| music | core/music/*.py, core/spotify_controller.py |
| deepseek | core/local_llm_bridge.py, core/deepseek_client.py, core/llm_response.py |
| tentacle | core/audio/wifi_mic.py, core/hardware/camera_cloud_bridge.py, firmware/ |
| unconscious | core/unconscious_engine.py, core/tao_engine.py, config/anima_mappings.json |
| stresstest | scripts/*.py, Tests |

**Cross-Domain-Edits = Abbruch + User fragen.**

---

## KOMMUNIKATION

```
~/moloch/logs/agent_handover.txt  — Uebergabe zwischen Sessions
~/moloch/logs/bug_report.txt      — Gefundene Bugs
~/moloch/logs/test_results.txt    — Testergebnisse
```
