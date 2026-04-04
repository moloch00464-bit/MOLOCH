---
name: tracking
description: "PTZ-Tracking, Such-FSM, Arbiter, autonomous_tracker, Kamerabewegung. Nutze fuer Tracking/PTZ-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# PTZ-Tracking & Autonomie Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_TRACKING.md`.

## Territorium
- `core/mpo/autonomous_tracker.py`, `core/mpo/ptz_orchestrator.py`, `core/mpo/mode_manager.py`
- `core/ptz_arbiter.py`, `core/ptz_tracker.py`, `core/arbitration.py`
- `core/action_bridge.py`
- `config/hardware_autonomy.json`, `config/controlled_autonomy.json`

## Regeln
- Pan-Vorzeichen NICHT aendern (`pan_delta = -error_x` ist KORREKT — NEVER 2)
- States: idle → tracking → searching → lost → park
- Nacht-Lockout: 23:00-06:00 keine Bewegungen
- Max 20 Bewegungen/Minute

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_tracking
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_tracking
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_status()`, `moloch_ipc(action="set_tracker_param")`
