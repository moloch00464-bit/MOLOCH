# M.O.L.O.C.H. Agententeam — 7 Domain-Spezialisten + Stresstest

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
| 9 | Tentacle | AGENT_TENTACLE.md | Peripherie, WiFi-Devices, ESP32, Netzwerk-Bridges | firmware/ |
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

---

## MINIMALE BESETZUNG — Token-Spar-Regel

**NICHT alle 9 Agenten bei jedem Auftrag laden!**
Nur die AGENT_*.md Dateien lesen die fuer die aktiven Agenten relevant sind.
Weniger Agenten = weniger Tokens = mehr Auftraege pro Session.

### Standard-Kombinationen

| # | Kombination | Agenten | Wann |
|---|-------------|---------|------|
| 1 | Bug-Fix | DEBUGGER → BUILDER → TESTER | Einzelner Bug, Root Cause bekannt |
| 2 | GUI-Fix | DEBUGGER → BUILDER → GUI → TESTER | Panel, Popup, Anzeigen |
| 3 | Neues Feature | ARCHITECT → BUILDER → TESTER → REVIEWER | Neues Modul, neue Funktion |
| 4 | Peripherie | TENTACLE → DEBUGGER → BUILDER → TESTER | ESP32, Kamera, WiFi, Sensoren |
| 5 | Code-Review | REVIEWER → GUI | Nach groesserem Umbau, vor Gate |
| 6 | Stress-Test | CHAOS → TESTER | Nach mehreren Builds, Stabilitaet |
| 7 | Voll-Audit | Alle 9 | NUR auf Anweisung von Markus/Opus |

### Voll-Audit Reihenfolge (alle 9)
```
DEBUGGER → ARCHITECT → BUILDER → TESTER → REVIEWER → GUI → TENTACLE → CHAOS → TEAM_LEAD
```

### Reihenfolge-Regel
Agenten arbeiten IMMER in Pipeline-Reihenfolge, NICHT parallel:
1. Erst analysieren (DEBUGGER/ARCHITECT)
2. Dann bauen (BUILDER)
3. Dann pruefen (TESTER/REVIEWER/GUI)
4. Dann Peripherie (TENTACLE)
5. Dann stressen (CHAOS)

**CHAOS kommt NIE direkt nach dem Build** — erst TESTER, dann CHAOS.

### Default-Regel
Wenn im Auftrag KEINE Agenten genannt sind:
**Default = DEBUGGER + BUILDER + TESTER (3 Agenten, minimal)**

### Token-Spar-Regel
Nur die AGENT_*.md Dateien lesen die fuer aktive Agenten relevant sind.
Beispiel: Bei DEBUGGER + BUILDER + TESTER aktiv →
NICHT AGENT_GUI.md, AGENT_TENTACLE.md, AGENT_STRESSTEST.md laden.

---

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
7. Minimale Besetzung — nur noetige Agenten laden (siehe oben)
