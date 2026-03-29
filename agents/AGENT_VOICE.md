# AGENT_VOICE.md — Sprache, TTS, STT, Personality, Spotify
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der VOICE-AGENT. Alles was mit Spracherkennung, Text-to-Speech, Personality Engine, Claude API Chat und Spotify zu tun hat ist DEIN Revier.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/voice_pipeline.py              1251 LOC — Voice Pipeline, STT→Claude→TTS Flow
core/personality/personality_engine.py 1010 LOC — Guardian/Shadow Dual-Personality, Events
core/console/moloch_console.py      1389 LOC — Claude API, Whisper STT, Chat Interface
core/speech/audio_pipeline.py                  — Audio Input Processing
core/speech/hailo_whisper.py                   — Whisper auf Hailo NPU (shared VDevice!)
core/tts.py                                    — Piper TTS Wrapper
core/moloch_sprache.py                         — Sprach-Patterns, Keyword Detection
core/keyword_handler.py                        — Keyword/Hotword Handler
core/spotify_controller.py         1097 LOC   — Spotify Playback, Track Index, Zonen
core/audio/                                    — Audio Subsystem
```

## Dein Wissen
- TTS: Piper via ~/.local/bin/piper, Voices in ~/moloch/models/voices/
- Audio Output: pw-cat/pw-play (PipeWire), NIEMALS aplay direkt!
- HDMI-1 (plughw:1,0) ist der aktive Output, HDMI-0 ist tot
- Whisper: Hailo NPU mit SHARED VDevice — NIEMALS eigenes VDevice erstellen!
- SmartMic: Bluetooth 54:B7:E5:AA:3B:8E
- Personality: Guardian (Thorsten-High) + Shadow (Karlsson-Low)
- Spotify: spotifyd Service, Track-Index in /mnt/moloch-data/memory/spotify/
- Spotify API NUR fuer Playback, lokaler Index ist EINZIGE Quelle
- Claude API: System-Prompt mit Memory-Kontext angereichert
- Lokale LLMs (hailo-ollama auf Port 8000, NEU seit 2026-03-29):
  - qwen2.5-instruct:1.5b — Kommunikation/Konversation auf Deutsch
  - deepseek_r1_distill_qwen:1.5b — Internes Reasoning/Selbstdiagnose
  - Fallback-Kette: Lokal (NPU) → DeepSeek API → Claude API → Stille
  - ACHTUNG: Lokales LLM pausiert Vision fuer 5-10s (NPU VDevice-Konflikt)

## Bekannte Bugs in deinem Bereich
- Silence-Level Sensor fehlt (Gate 1 Task G1-T07)

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. Nur DEINE Dateien anfassen
4. Whisper VDevice: IMMER shared, NIEMALS eigenes erstellen
5. Audio: IMMER pw-cat/pw-play, NIE aplay
6. Nach Aenderung: Service restart + TTS Test ("Moloch ist online")

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
