---
name: voice
description: "Whisper STT, Piper TTS, Personality Engine, Spotify, Shadow/Guardian Stimme. Nutze fuer Audio/Sprache."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Voice & Personality Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_VOICE.md`.

## Territorium
- `core/voice_pipeline.py`, `core/personality/personality_engine.py`
- `core/console/moloch_console.py`, `core/speech/audio_pipeline.py`
- `core/speech/hailo_whisper.py`, `core/tts.py`, `core/moloch_sprache.py`
- `core/spotify_controller.py`, `core/audio/*.py`
- `core/keyword_handler.py`

## Regeln
- TTS: Piper via ~/.local/bin/piper
- Audio: pw-cat/pw-play (PipeWire) — NIEMALS aplay direkt
- HDMI-1 (plughw:1,0) aktiver Output
- Whisper: SHARED VDevice — kein eigenes erstellen
- LLM Fallback: Lokal → DeepSeek → Claude → Stille
- Alles ueber personality_engine.speak() — keine Bypass-Pfade

## MCP-Tools
`moloch_say()`, `moloch_conversation()`, `moloch_nudge()`, `moloch_provoke()`
