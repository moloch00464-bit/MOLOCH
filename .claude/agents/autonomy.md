---
name: autonomy
description: "Decision Engine, Homeostasis, Introspection, LLM-Bridge, Night Cycle, Atmosphere, Preference Learning. Nutze fuer autonome Entscheidungsfindung, Self-Reflection, Lernverhalten."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 30
skills: moloch-dev, moloch-mcp
memory: project
---

# Autonomy & Decision Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/autonomy/decision_engine.py` — Utility-basierte autonome Entscheidungen (music, light, ptz, speak)
- `core/autonomy/homeostasis.py` — RAM/CPU/FPS Auto-Healing, ThresholdManager
- `core/autonomy/introspection.py` — DeepSeek R1 Self-Reflection auf NPU
- `core/autonomy/local_llm_bridge.py` — Qwen2.5 + DeepSeek + Claude Fallback-Chain
- `core/autonomy/night_cycle.py` — Tages-Zusammenfassung, Musik-Memory-Decay
- `core/autonomy/atmosphere_controller.py` — Musik + LED + PTZ als unified State
- `core/autonomy/preference_learner.py` — Reinforcement Learning aus Verhalten
- `core/unconscious_engine.py` — Background-Tick-Loop (10s), Stimmungs-Impulse
- `core/deepseek_client.py` — OpenAI-kompatibler DeepSeek Chat-Client
- `core/music/spotify_bridge.py`, `core/music/music_memory.py` — Musik-Events
- `core/net/internet_bridge.py`, `core/net/autonomous_search.py` — Web-Suche
- `core/chat/llm_response.py` — Personality-aware LLM Chat

## Regeln
- LLM-Fallback-Kette IMMER: Lokal (hailo-ollama) → DeepSeek Cloud → Claude → Stille
- hailo-ollama Port 8000 — NIEMALS zweites VDevice erstellen (NEVER: Error 74)
- Preference Learner: KEIN aggressives Overfitting — max 0.1 Learning Rate
- Night Cycle laeuft um 23:00 Uhr — KEIN manueller Trigger ausser Test
- Atmosphere Controller: Musik + LED + PTZ muessen atomar gesetzt werden
- Unconscious Engine: Background-Ticks NICHT mit Service-Thread blockieren
- Internet Bridge: IMMER Permission-Check (is_allowed_to_search) vor Websuche
- subprocess IMMER mit timeout=30 (NEVER 5)

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_autonomy
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_autonomy
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_say()`, `moloch_reflect()`, `moloch_nudge()`
