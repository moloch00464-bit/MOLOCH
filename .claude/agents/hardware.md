---
name: hardware
description: "ONVIF, RTSP, PTZ-Mechanik, Sonoff Kamera, eWeLink, LED, IR, Thermal, Fan. Nutze fuer Kamera/Hardware-Probleme. ESP32/WiFi-Mic gehoert zum tentacle-Agenten."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 20
skills: moloch-dev
memory: project
---

# Hardware Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_HARDWARE.md`.

## Territorium
- `core/hardware/camera.py` — ONVIF PTZ, RTSP-Verbindung
- `core/hardware/ptz_calibration.py` — Winkel-Kalibrierung
- `core/hardware/thermal_manager.py` — Noctua Fan PWM, Temp-Monitoring
- `core/led_controller.py`, `core/hardware/rgb_led_controller.py`
- `core/cloud_controller.py` — eWeLink Device Control
- `config/hardware_autonomy.json`

## Abgrenzung
- `wifi_mic.py`, `camera_cloud_bridge.py` → tentacle-Agent (ESP32/Cloud)
- PTZ-Logik (Bewegungssteuerung) → tracking-Agent

## Hardware-Fakten
- Sonoff CAM-PT2: 192.168.178.25, 1080p@20fps, ONVIF PTZ
- Pan INVERTIERT: positiver Wert = physisch LINKS
- Pan: -168.4 bis 174.4 | Tilt: -78.8 bis 101.3
- RTSP hat NUR EINEN Slot — kein Doppelzugriff
- Noctua Fan: 30% Duty = 48°C unter Volllast
- ONVIF Timeout: 5s (eingestellt in camera.py)

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_hardware   # Erster Schritt
rm /tmp/moloch_agent_hardware      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_dmesg()`, `moloch_ipc(action="ptz_move")`, `moloch_logs(filter_str="ONVIF")`
