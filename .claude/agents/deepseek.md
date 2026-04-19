---
name: deepseek
description: "DeepSeek R1, lokales LLM auf NPU (hailo-ollama), LLM-Client-Code, Meta-QA, Philosophie, LLM-Integration. Nutze fuer LLM/DeepSeek-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev, moloch-mcp
memory: project
---

# DeepSeek & LLM Integration Agent

Lies IMMER zuerst: `CLAUDE.md`, `agents/AGENT_DEEPSEEK.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/autonomy/local_llm_bridge.py` — Qwen2.5 + DeepSeek + Claude Fallback-Chain
- `core/deepseek_client.py` — OpenAI-kompatibler DeepSeek Chat-Client
- `core/chat/llm_response.py` — Personality-aware LLM Chat, System-Prompt-Building
- `core/autonomy/introspection.py` — DeepSeek R1 Self-Reflection auf NPU

## Hardware-Fakten (Stand Session 19, 2026-04-19)
- hailo-ollama: Port 8000, systemd-Service AKTIV (`hailo-ollama.service`,
  Environment `OLLAMA_KEEP_ALIVE=-1` + `HAILO_OLLAMA_VDEVICE_GROUP_ID=SHARED`)
- Default-Modell: `qwen2.5:1.5b` (Qwen2.5 Instruct, NPU-stable seit HailoRT 5.3.0)
- HailoRT: 5.3.0 (Library + Driver + Firmware), HAILO_MAX_NETWORK_GROUPS=8
- DeepSeek R1 (deepseek_r1:1.5b in 5.3.0) — UNGETESTET, in 5.1.1 deterministischer SEGV
- VDevice: SHARED (vdevice-group-id=SHARED) — NIEMALS zweites erstellen (Error 74)
- LLM-Profile-System (`config/llm_profiles.json`): 5 Presets (chat/introspect/technical/dark/multi_person)
  - Switch via `settings.json` Key `llm_profile` (mtime-Cache, kein Restart noetig)
  - GUI-Reiter "LLM-Modus" im Panel Modelle
- DeepSeek Cloud API: Key in `config/api_keys.json` (kann via `.disabled_*`-Suffix
  hart deaktiviert werden — User-Modus "NPU-only permanent")
- Anthropic Key: ebenfalls in api_keys.json (Claude als 3. Fallback)

## Kritische Regeln
- Fallback-Kette IMMER: hailo-ollama (Port 8000) → DeepSeek Cloud → Claude → Stille
- Im NPU-only Modus (api_keys.json deaktiviert): hailo-ollama → Stille
- hailo-ollama teilt NPU mit TAPPAS (SHARED VDevice) — kein Konflikt wenn richtig konfiguriert
- KEIN eigenes VDevice erstellen — nur `set_vdevice(service._vdevice)` nutzen
- LLM-Antworten NUR via personality_engine.speak() ausgeben — kein Bypass!
- API-Keys NIEMALS committen oder in Logs schreiben
- timeout=30 fuer alle LLM-HTTP-Calls (NEVER 5)
- DeepSeek URL: `/v1/chat/completions` — NICHT `/chat` (falscher Endpunkt!)

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_deepseek   # Erster Schritt
rm /tmp/moloch_agent_deepseek      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_say()`, `moloch_reflect()`, `moloch_conversation()`
