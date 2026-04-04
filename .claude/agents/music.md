---
name: music
description: "Spotify, Musik-Reaktion, Track-Polling, Music Memory, Beat-Erkennung, Zone-basierte Musik. Nutze fuer alle Musik/Spotify-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
maxTurns: 20
skills: moloch-dev
memory: project
---

# Music & Spotify Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/music/spotify_bridge.py` — Track-Polling, Events (music_track_started/finished)
- `core/music/music_memory.py` — Track-Person-Mood Assoziationen speichern
- `core/spotify_controller.py` — Authentifizierung, Playback-Kontrolle, Liking
- `/mnt/moloch-data/memory/spotify/track_index.json` — 4941 Artists, 24454 Tracks

## Hardware-Fakten
- spotifyd 0.4.2: systemd Service, `use_mpris = false` (kein DBus noetig)
- Audio-Ausgabe: HDMI-1 (plughw:1,0) via PipeWire
- TTS und Spotify mixen ueber PipeWire — NIEMALS aplay direkt (Device busy)
- Volume: softvol (HDMI hat keine ALSA Mixer Controls)
- Spotify API: NUR fuer Playback (sp.start_playback(uris=[...])), Token auto-refresh

## Regeln
- Lokaler Index ist EINZIGE Quelle — KEINE Spotify-Suche/Recommendations
- Zone Artists: Guardian=Futurepop/Synthwave, Shadow=Dark Electro/EBM, Berserker=Aggrotech
- Musik-Auswahl IMMER via Track-Index, NICHT via Spotify Search API
- pw-play fuer WAV, pw-cat fuer raw PCM — NIEMALS aplay
- Music Memory: Assoziationen max 50 pro Track (RAM-Schonung)
- Track-Polling: 30s Intervall — KEIN aggressiveres Polling

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_music
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_music
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_say()`
