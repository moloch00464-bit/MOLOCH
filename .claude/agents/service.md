---
name: service
description: "moloch_service.py, IPC, ServiceProxy, CoreIntegrator, systemd. Nutze fuer Service/Backend/Integration. Memory->memory-Agent."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 30
skills: moloch-dev, moloch-mcp
memory: project
---

# Service & Integration Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_SERVICE.md`.

## Territorium
- `core/moloch_service.py` — Haupt-Service: Worker-Start, VDevice, Poll-Thread (ROT!)
- `core/core_integrator.py` — Tension/Zone/LED Integration (ROT!)
- `core/ipc_router.py` — IPC Dispatch (ROT!)
- `core/status.py` — Status-JSON Aufbau (/dev/shm/moloch_status.json)
- `core/camera_manager.py` — RTSP-Fallback (ohne TAPPAS)
- `core/environment_watcher.py` — Umgebungs-Monitoring
- `/etc/systemd/system/moloch.service`

## Abgrenzung
- Memory (episodic, persistent, longterm) → memory-Agent
- Personality/Mood → personality-Agent
- LLM/DeepSeek → deepseek-Agent

## Kritische Regeln
- Feature-Flag: MOLOCH_USE_TAPPAS=1 in moloch.service und ~/.profile
- Status-JSON: /dev/shm/moloch_status.json (RAM-Disk, nicht committen!)
- IPC NUR via ServiceProxy — kein Direktzugriff auf Service-Internals
- Poll-Thread: 5 Hz — NICHT erhoehen (CPU-Last)
- Core State: alle 60s + bei stop() sichern
- KEIN np.ndarray Type-Hint in Signaturen (NEVER 10)
- _ctx_lock in _write_status_json() im TAPPAS-Mode NICHT verwenden (Deadlock!)

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_service   # Erster Schritt
rm /tmp/moloch_agent_service      # Letzter Schritt
```

## MCP-Tools
`moloch_service()`, `moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_audit()`
