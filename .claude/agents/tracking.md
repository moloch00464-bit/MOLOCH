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
- `core/mpo/autonomous_tracker.py` — Such-FSM: idle→tracking→searching→lost→park
- `core/mpo/ptz_orchestrator.py`, `core/mpo/mode_manager.py`
- `core/ptz_arbiter.py`, `core/ptz_tracker.py`, `core/arbitration.py`
- `core/action_bridge.py`
- `config/hardware_autonomy.json`, `config/controlled_autonomy.json`

## Kritische Regeln
- Pan-Vorzeichen NICHT aendern: `pan_delta = -error_x` ist KORREKT (NEVER 2)
- Pan-Grenzen: -168.4 bis 174.4 | Tilt: -78.8 bis 101.3
- Nacht-Lockout: 23:00-06:00 keine Bewegungen
- Max 20 Bewegungen/Minute
- Sonoff Pan INVERTIERT: positiver Wert = physisch LINKS

## Offener Bug (PRIO 3)
- `autonomous_tracker.py` → `_track_tracking_target()`: Nach 8s am mechanischen
  Anschlag → SEARCH starten (STUCK-AT-LIMIT Erkennung fehlt)

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_tracking   # Erster Schritt
rm /tmp/moloch_agent_tracking      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_ipc(action="set_tracker_param")`, `moloch_logs()`, `moloch_snapshot()`
