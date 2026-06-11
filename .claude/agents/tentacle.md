---
name: tentacle
description: "ESP32 WiFi-Mikrofon, ReSpeaker Lite, Arduino-Firmware, UDP-Audio, RGB-LED, eWeLink Cloud, externe WiFi-Devices. Nutze fuer Peripherie/Firmware-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 20
skills: moloch-dev
memory: project
---

# Tentacle & Peripherie Agent

Lies IMMER zuerst: `CLAUDE.md`, `agents/AGENT_TENTACLE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/audio/wifi_mic.py` — ESP32 WiFi-Mikrofon UDP-Stream Client (Pi-Seite)
- `core/hardware/camera_cloud_bridge.py` — eWeLink Cloud API Bridge
- `core/cloud_controller.py` — eWeLink Device Control
- `firmware/respeaker_wifi_mic/` — ESP32 Arduino Firmware (komplett)
- `scripts/test_respeaker_udp.py` — UDP-Test Script

## Hardware-Fakten (ESP32 ReSpeaker Lite)
- Board: XIAO ESP32-S3 + XMOS XU316, MAC b8:f8:62:fa:16:74
- WiFi: Direkt-AP `MOLOCH_DIRECT`, statisch 10.42.0.2, Ping ~2ms
- UDP Audio: Port 12345 (16kHz Mono, 320B/Paket), Port 12346 (48kHz Stereo)
- UDP LED: Port 8888, Format: `LED:farbe [modus] [geschwindigkeit]`
- HTTP Port 80: /audio/status (GET), /audio/start, /audio/stop
- `/audio/mode?rate=16000|48000` braucht **POST** — GET liefert 404!
  Schaltet live, kein Stop/Start noetig: `curl -X POST "http://10.42.0.2/audio/mode?rate=16000"`
- OTA: ArduinoOTA, Hostname "moloch-mic"
- RGB-LED: WS2812 auf GPIO1 via neopixelWrite()
- Flash: `arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32S3 -p /dev/ttyACM0`
- I2S Pins (verifiziert): BCLK=GPIO8, LRCK=GPIO7, DIN=GPIO44, DOUT=GPIO43, MCLK=GPIO9

## Kritische Regeln
- Firmware-Aenderungen: IMMER Backup der .ino Datei vor Compile
- UDP-Audio: NIEMALS TCP verwenden (zu hohe Latenz)
- eWeLink: API v2 Login (api_keys.json), Token auto-refresh
- Arduino Flash: NUR via arduino-cli — NICHT via Arduino IDE
- WiFi-Hotspot (nmcli "Hotspot"): autoconnect=yes, Channel 6 — NICHT veraendern
- RTSP hat NUR EINEN Slot — kein Doppelzugriff

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_tentacle   # Erster Schritt
rm /tmp/moloch_agent_tentacle      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_dmesg()`, `moloch_ipc()`
