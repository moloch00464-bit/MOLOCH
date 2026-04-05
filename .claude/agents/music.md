---
name: music
description: "Spotify, Musik-Reaktion, Track-Polling, Music Memory, Zone-basierte Musik, spotifyd. Nutze fuer alle Musik/Spotify-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
maxTurns: 20
skills: moloch-dev
memory: project
---

# Music & Spotify Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/spotify_controller.py` — Authentifizierung, Playback-Kontrolle, Token-Refresh
- `core/music/spotify_bridge.py` — Track-Polling, Events (music_track_started/finished)
- `core/music/music_memory.py` — Track-Person-Mood Assoziationen (max 50 pro Track)
- `/mnt/moloch-data/memory/spotify/track_index.json` — 4941 Artists, 24454 Tracks (READ-ONLY!)

## Hardware-Fakten
- spotifyd 0.4.2: systemd Service, `use_mpris = false` (kein DBus)
- Audio-Ausgabe: HDMI-1 (plughw:1,0) via PipeWire
- TTS und Spotify mixen ueber PipeWire — NIEMALS aplay direkt (Device busy!)
- Volume: softvol (HDMI hat keine ALSA Mixer Controls)

## Kritische Regeln
- Lokaler Track-Index ist EINZIGE Quelle — KEINE Spotify-Suche/Recommendations API
- Musik-Auswahl IMMER via Track-Index, NICHT via Spotify Search API
- Zone-Mapping: Guardian=Futurepop/Synthwave | Shadow=Dark Electro/EBM | Berserker=Aggrotech
- Audio: pw-play fuer WAV, pw-cat -p --raw fuer PCM — NIEMALS aplay
- Playback NUR via `sp.start_playback(uris=[...])` (Spotify API)
- Track-Polling: 30s Intervall — kein aggressiveres Polling

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_music   # Erster Schritt
rm /tmp/moloch_agent_music      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_say()`
