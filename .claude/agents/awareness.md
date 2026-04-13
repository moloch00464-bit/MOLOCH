---
name: awareness
description: "Situational Awareness, Aktivitaetserkennung, Raumzonen, Kontextbewertung, Bewegungsanalyse, WorldState. Nutze fuer Gate-3 Awareness-Arbeit."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Situational Awareness Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/awareness/activity_analyzer.py` — Aktivitaetserkennung (alone, working, conversation, party, away)
- `core/awareness/context_evaluator.py` — Kontextbewertung (Familiarity, Comfort, Alertness Score)
- `core/awareness/motion_analyzer.py` — Bewegungsanalyse (stationary, walking, approaching, leaving)
- `core/awareness/room_map.py` — PTZ-Winkel zu Raumzonen Mapping (Tuer, Schreibtisch, Sofa, Fenster)
- `core/world/world_state.py` — Inventar (Peripherals, Sensoren, Interaktionskanaele)
- `core/environment_watcher.py` — Raum-Monitoring, Umgebungsaenderungen

## Abgrenzung
- `core/capability_monitor.py` → watchdog-Agent (System-Health, nicht Awareness)

## Kritische Regeln
- Room Map ist KALIBRIERT — Winkel-Werte NICHT ohne visuellen Test aendern
- Activity Analyzer: Ausgabe geht direkt an CoreIntegrator → Tension-Aenderungen moeglich
- Context Evaluator: Score-Berechnung kalibriert (0.0-1.0) — KEIN Rescaling
- Awareness-Daten fliessen in Tension-Integrator (personality/) — Rueckwaerts KEIN Direktzugriff
- Environment Watcher: Polling alle 30s — KEIN aggressiveres Polling (RAM-Schonung)
- World State: Passiv (nur laden/speichern), KEIN Polling-Thread

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_awareness   # Erster Schritt
rm /tmp/moloch_agent_awareness      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_snapshot()`
