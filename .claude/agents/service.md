---
name: service
description: "moloch_service.py, IPC, ServiceProxy, CoreIntegrator, Memory, systemd. Nutze fuer Service/Backend/Integration."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 30
skills: moloch-dev, moloch-mcp
memory: project
---

# Service & Integration Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_SERVICE.md`.

## Territorium
- `core/moloch_service.py`, `core/core_integrator.py`
- `core/ipc_router.py`, `core/status.py`, `core/camera_manager.py`
- `core/longterm_memory.py`, `core/memory/*.py`
- `core/daily_learner.py`, `core/environment_watcher.py`
- `/etc/systemd/system/moloch.service`

## Regeln
- Feature-Flag: MOLOCH_USE_TAPPAS=1 in moloch.service
- Status: /dev/shm/moloch_status.json (RAM-Disk)
- IPC NUR via ServiceProxy
- Memory: /mnt/moloch-data/memory/ auf SSD2
- Poll-Thread: 5 Hz
- Core State: alle 60s + bei stop() sichern
- KEIN np.ndarray Type-Hint in Signaturen (NEVER 10)

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_service
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_service
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_service()`, `moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_audit()`
