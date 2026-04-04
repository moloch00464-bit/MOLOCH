---
name: gui
description: "Tkinter Panel, Module, Popups, panel_*.py, popup_*.py. Nutze fuer GUI-Aenderungen."
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
maxTurns: 20
skills: moloch-dev
memory: project
---

# GUI Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_GUI.md`.

## Territorium
- `core/gui/panel_main.py`, `core/gui/panel_*.py`
- `core/gui/popups/popup_*.py`
- `core/gui/panel_styles.py` — NUR LESEN, NIE AENDERN!

## Regeln
- Panel ist MODULAR: 1 Datei = 1 Aufgabe
- Kommunikation NUR via ServiceProxy/IPC
- Keine Inter-Modul Imports (ausser panel_styles.py)
- 4 GB RAM — sparsam mit Widgets/Timern
- `panel_styles.py` ist TABU (ausser Markus sagt explizit)

## MCP-Tools
`moloch_status()`, `moloch_snapshot()`
