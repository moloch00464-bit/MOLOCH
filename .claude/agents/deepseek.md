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

## Hardware-Fakten
- hailo-ollama: Port 8000, Binary installiert, KEIN systemd-Service (offener Bug PRIO 5)
- Modell: deepseek_r1_distill_qwen:1.5b (~1.5B Parameter)
- VDevice: SHARED (vdevice-group-id=SHARED) — NIEMALS zweites erstellen (Error 74)
- DeepSeek Cloud API: Key in `config/api_keys.json` (OpenAI-kompatibel)
- Anthropic Key: ebenfalls in api_keys.json

## Kritische Regeln
- Fallback-Kette IMMER: hailo-ollama (Port 8000) → DeepSeek Cloud → Claude → Stille
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
