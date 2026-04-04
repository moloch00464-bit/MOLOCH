---
name: hardware
description: "ONVIF, RTSP, PTZ-Mechanik, Sonoff Kamera, eWeLink, LED, IR, Thermal, Fan. Nutze fuer Hardware-Probleme."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 20
skills: moloch-dev
memory: project
---

# Hardware Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_HARDWARE.md`.

## Territorium
- `core/hardware/camera.py`, `core/hardware/camera_cloud_bridge.py`
- `core/hardware/ptz_calibration.py`, `core/hardware/thermal_manager.py`
- `core/led_controller.py`, `core/cloud_controller.py`
- `config/hardware_autonomy.json`

## Hardware-Fakten
- Sonoff CAM-PT2: 192.168.178.25, 1080p@20fps, ONVIF PTZ
- Pan INVERTIERT: positiver Wert = physisch LINKS
- Pan: -168.4 bis 174.4 | Tilt: -78.8 bis 101.3
- RTSP hat NUR EINEN Slot
- Noctua Fan: 30% = 48°C unter Volllast

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_hardware
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_hardware
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_status()`, `moloch_dmesg()`, `moloch_ipc(action="ptz_move")`
