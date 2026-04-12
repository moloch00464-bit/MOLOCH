---
name: voice
description: "Whisper STT, Piper TTS, Audio-Pipeline, Sprach-I/O, Keyword-Handler. Nutze fuer Audio/Sprache. Personality->personality-Agent. Spotify->music-Agent."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Voice & Audio Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_VOICE.md`.

## Territorium
- `core/voice_pipeline.py` — STT→Intent→TTS Haupt-Pipeline
- `core/speech/hailo_whisper.py` — Whisper auf NPU (SHARED VDevice)
- `core/speech/audio_pipeline.py` — Mic-Source-Router (WiFi-Mic vs USB)
- `core/tts.py` — Piper TTS Wrapper
- `core/moloch_sprache.py` — Deutsch NLP, Sprach-Stil
- `core/keyword_handler.py` — Wake-Word / Keyword-Detection
- `core/audio/*.py` — Audio-Utilities

## Abgrenzung
- Personality/Mood/Zones → personality-Agent
- Spotify/Musik → music-Agent
- ESP32 WiFi-Mic Firmware → tentacle-Agent

## Kritische Regeln
- TTS: Piper via `~/.local/bin/piper`
- Audio-Output: pw-play (WAV) / pw-cat -p --raw (PCM) — NIEMALS aplay direkt
- HDMI-1 (plughw:1,0) ist der aktive Audio-Output
- Whisper: SHARED VDevice — NIEMALS eigenes erstellen (Error 74)
- LLM-Fallback-Kette: hailo-ollama (DeepSeek R1 lokal) → DeepSeek API (Cloud) → Stille
- ALLE TTS-Ausgaben gehen durch personality_engine.speak() — kein Bypass!

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_voice   # Erster Schritt
rm /tmp/moloch_agent_voice      # Letzter Schritt
```

## MCP-Tools
`moloch_say()`, `moloch_conversation()`, `moloch_nudge()`, `moloch_provoke()`, `moloch_logs(filter_str="TTS")`
