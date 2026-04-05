---
name: watchdog
description: "System-Health Monitoring, Frame-Freeze, Temperaturen, RAM, Netzwerk, LLM-Check, ONVIF-Errors, CapabilityMonitor. Nutze fuer Watchdog/Health-Arbeit."
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
maxTurns: 20
skills: moloch-dev, moloch-mcp
memory: project
---

# System Watchdog Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/system_watchdog.py` — Health-Monitoring (Frame-Freeze, ONVIF, Temp, RAM, Disk, Netz, LLM)
- `core/diagnostics.py` — System-Diagnostik, Fehler-Analyse, Status-Report
- `core/capability_monitor.py` — Faehigkeiten-Matrix (aktive Features, 13 Checks)
- `core/status.py` — Status-JSON Aufbau und IPC-Schnittstelle

## Watchdog-Schwellwerte (Referenz)
| Check | Schwelle | Aktion |
|-------|----------|--------|
| Frame-Freeze | SHM > 2s alt | Restart-Trigger |
| ONVIF-Errors | > 5 AbsoluteMove-Fehler | ONVIF-Reset |
| CPU-Temp | > 75°C | Fan-Boost |
| CPU-Temp | > 85°C | Service-Stopp |
| RAM | > 90% | Homeostasis-Trigger |
| Disk Root | > 90% | Log-Rotation |
| Netzwerk | Ping 192.168.178.1 | alle 30s |
| LLM | hailo-ollama Port 8000 | alle 60s |

## Kritische Regeln
- Watchdog laeuft als eigener Thread — KEIN blocking in Check-Funktionen (max 2s pro Check)
- Restart-Trigger: NUR via moloch_service IPC — NIEMALS direktes os.kill
- Tension-Spike bei Health-Problemen: max +0.3 pro Event (kein Overflow)
- ONVIF-Reset: IMMER mit Timeout absichern (timeout=30, NEVER 5)
- Log-Rotation: ERST nach Backup loeschen

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_watchdog   # Erster Schritt
rm /tmp/moloch_agent_watchdog      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_dmesg()`, `moloch_audit()`, `moloch_service()`
