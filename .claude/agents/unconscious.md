---
name: unconscious
description: "TaoEngine, Unterbewusstsein, Mood-Impulse, Self-Tune, Tension-Offset. Nutze fuer alles was mit dem inneren Zustand und autonomer Selbstregulation zu tun hat."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 30
skills: moloch-dev, moloch-mcp
memory: project
---

# Unconscious / TaoEngine Agent

Lies IMMER zuerst: `CLAUDE.md`, `docs/plans/tao_engine_plan.md` und `agents/AGENT_UNCONSCIOUS.md`.

## Territorium
- `core/unconscious_engine.py` — TaoEngine (Hauptdatei)
- `config/settings.json` → Sektion `tao_engine` (Kill Switch)
- `config/anima_mappings.json` — TAO→Behavior Mappings
- `config/self_tune_registry.json` — 69 Self-Tune Parameter
- `config/diagnose_rules.json` — Diagnose-Regeln
- Event Bus: `tao.state_update`, `tao.tension_offset`

## Architektur
- TaoEngine laeuft als Daemon-Thread (500ms Tick), KEIN asyncio
- 4 State-Variablen: yin, yang, wu_wei, ziran (alle 0.0-1.0)
- 4 Derived Metrics: balance, flow, stability, activity
- Kommunikation NUR ueber Event Bus (get_event_bus())
- KEIN direkter Import von moloch_service.py oder core_integrator.py
- Tension-Offset max ±0.02 pro Tick, clamp 0.05-0.95

## Regeln
- max_delta_per_tick = 0.02 (NICHT 0.12!)
- Logging nur bei State-Aenderung > 0.05
- try/except um jeden Tick — darf NIEMALS crashen
- Max 150 LOC
- Kill Switch: `settings.json → tao_engine.enabled`

## Angrenzende Dateien (NICHT anfassen, nur lesen)
- `core/core_integrator.py` — Tension-Offset Consumer (INT-02)
- `core/moloch_service.py` — TaoEngine Lifecycle (INT-01)
- `core/personality/mood_engine.py` — Mood Events lesen
- `core/moloch_event_bus.py` — Event Bus API

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_unconscious
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_unconscious
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.
