# M.O.L.O.C.H. Agententeam — 6 Domain-Spezialisten + Stresstest

## Das Team

| # | Agent | Datei | Domain | LOC |
|---|-------|-------|--------|-----|
| 1 | Vision | AGENT_VISION.md | TAPPAS, GStreamer, Hailo NPU, HEFs, Perception | ~3000 |
| 2 | Hardware | AGENT_HARDWARE.md | ONVIF, RTSP, PTZ-Mechanik, eWeLink, Thermal | ~4300 |
| 3 | GUI | AGENT_GUI.md | Tkinter Panel, Module, Popups, Konsistenz-Audit | ~7500 |
| 4 | Tracking | AGENT_TRACKING.md | PTZ-Tracker, Such-FSM, Arbiter, Autonomie | ~2600 |
| 5 | Voice | AGENT_VOICE.md | Whisper, TTS, Personality, Claude API, Spotify | ~4750 |
| 6 | Service | AGENT_SERVICE.md | moloch_service, IPC, Memory, Integration | ~3400 |
| + | Stresstest | AGENT_STRESSTEST.md | Chaos Engineering, 8 Szenarien | scripts/ |
| - | Team Lead | Markus | Boss, Priorisierung, Entscheidung | - |
| - | DeepSeek | extern | Philosophie, Meta-QA, Chaos mit Methode | - |

## Wie du einen Agenten startest

Neue Claude Code Instanz, dann:
```
Lies ~/moloch/CLAUDE.md und ~/moloch/agents/AGENT_VISION.md.
[Dein Auftrag]
```

## Territorium-Regel

JEDER Agent hat seine Dateien. Kein Agent fasst fremde Dateien an.
Wenn ein Auftrag zwei Domains betrifft → zwei Agenten nacheinander.

## Kommunikation zwischen Agenten

~/moloch/logs/agent_handover.txt — Uebergabe zwischen Instanzen
~/moloch/logs/bug_report.txt — Bugs gefunden
~/moloch/logs/test_results.txt — Testergebnisse
~/moloch/logs/stress_results.txt — Stresstest Ergebnisse

## Welcher Agent fuer welchen Gate 1 Task?

| Task | Agent |
|------|-------|
| G1-T01 Action Bridge FSM | Tracking |
| G1-T02 Person-Detection triggert Tracking | Tracking + Vision |
| G1-T03 Auto-Resume + Spruch | Tracking + Voice |
| G1-T04 Suchrichtung Fix | Tracking |
| G1-T05 Gain-Tuning | Tracking |
| G1-T06 Park-Position = Tuer | Tracking |
| G1-T07 Silence-Level Sensor | Voice |
| G1-T08 Auto-Enrollment via Chat | Service + Vision |
| G1-T09 NPU-Dashboard | GUI |
| G1-T10 Tension-Popup Farben | GUI |
| G1-T11 Labelme Kalibrierung | Hardware |

## Goldene Regeln

1. 1 Agent = 1 Domain = klares Territorium
2. Git Backup VOR jeder Aenderung
3. Max 50 Zeilen pro Auftrag
4. Bei 85% Token → Uebergabe schreiben
5. Markus ist Boss — bei Konflikten entscheidet ER
6. KEIN Weitermachen bei FAIL im Audit
