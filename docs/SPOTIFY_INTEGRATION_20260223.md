# Spotify Integration - Status 2026-02-23

## 1. Installierte Komponenten

| Komponente | Version/Details | Status |
|-----------|----------------|--------|
| **spotifyd** | Spotify Connect Daemon, ALSA Backend, 320kbps | AKTIV (systemd user service) |
| **spotipy** | Python Spotify Web API Wrapper, OAuth 2.0 | Installiert |
| **spotify_controller.py** | 992 Zeilen, Singleton, Thread-safe | Produktiv |

## 2. Erstellte/Geaenderte Dateien

| Datei | Typ | Beschreibung |
|-------|-----|-------------|
| `core/spotify_controller.py` | NEU | Hauptcontroller (Playback, Mood, Auto-DJ, Genre-Lock) |
| `core/gui/panel_spotify.py` | NEU | Panel-Widget mit Transport, Volume, Smart Buttons |
| `core/gui/panel_main.py` | GEAENDERT | SpotifyModule eingebunden |
| `core/moloch_service.py` | GEAENDERT | IPC Commands + Status-Writing fuer Spotify |
| `core/console/moloch_console.py` | GEAENDERT | Voice Commands + Claude API SPOTIFY-Tags |
| `.env.spotify` | NEU | Web API Credentials |
| `scripts/spotify_auth.py` | NEU | OAuth Token-Generator |
| `scripts/analyze_spotify.py` | NEU | Extended History Analyzer |
| `~/.config/spotifyd/spotifyd.conf` | NEU | Daemon-Config (ALSA, HDMI, Device "MOLOCH") |
| `~/.config/systemd/user/spotifyd.service` | NEU | Autostart Service |

## 3. Voice Commands

### Direkte Keywords (ohne Claude API)
```
"stopp/pause/halt/ruhe"     -> Pause
"weiter/play/musik an"      -> Play
"naechster/skip/next"       -> Next Track
"zurueck/vorheriger"        -> Previous Track
"lauter/leiser"             -> Volume +/-
"was laeuft/welcher song"   -> Track-Info ansagen
"spiel <Query>"             -> Suche + Play
```

### Claude API Tags (in Antwort eingebettet)
```
[SPOTIFY:play/pause/toggle/skip/previous]
[SPOTIFY:volume=70]
[SPOTIFY:search=Suicide Commando Hellraiser]
[SPOTIFY:artist=VNV Nation]
[SPOTIFY:mood=shadow]       (guardian/shadow/berserker)
```

## 4. Spotify-Profil

**Datei:** `/mnt/moloch-data/memory/spotify/spotify_profile.json`

| Metrik | Wert |
|--------|------|
| Zeitraum | 2015-2025 |
| Gesamte Streams | 100.836 |
| Eindeutige Kuenstler | 3.218 |
| Eindeutige Tracks | 16.467 |
| Gesamthoerstunden | 6.833h (~285 Tage) |
| Skip-Rate | 10,7% |
| Aktivstes Jahr | 2016 (15.187 Streams) |

**Top 5 Kuenstler:** Suicide Commando (185h), Vomito Negro (123h), ESA (122h), Chainreactor (105h), SIERRA (103h)

**Genre-Verteilung:** Dark Electro/EBM 23,9% | Industrial Techno 4,7% | Harsh EBM/Aggrotech 3,7%

Profil wird fuer Zone-Kuenstler-Mapping und Genre-Lock genutzt.

## 5. Was funktioniert

- Play, Pause, Skip, Previous, Volume, Shuffle
- Track-Info Abfrage (Artist, Song, Fortschritt)
- Artist-Suche mit Genre-Lock Filter
- Mood-basierte Musik (Guardian/Shadow/Berserker Zones)
- Auto-DJ mit automatischem Zone-Tracking (5s Polling)
- Zeitmodifikatoren (Morgen=Guardian, Abend=Shadow, Nacht=Shadow dominant)
- Smart Features: Aehnliche Tracks, Top Tracks, Neue Musik, Tracks nach Jahr
- Genre-Lock (Dark Scene Whitelist, Mainstream Blacklist)
- Panel UI mit Live-Status, Transport Buttons, Volume Slider
- spotifyd Daemon laeuft (Device "MOLOCH" sichtbar in Spotify App)
- Console Voice Commands + Claude API Integration

## 6. Was fehlt noch

| Feature | Prioritaet | Beschreibung |
|---------|-----------|-------------|
| Genre-Lock Tuning | MITTEL | Bei API-Fehlern wird Artist erlaubt statt geblockt |
| Auto-DJ Verfeinerung | NIEDRIG | Zeitmodifikatoren noch nicht field-tested |
| Panel Widget Live | MITTEL | `panel_spotify.py` existiert, muss in Service integriert werden |
| Playlist-Management | NIEDRIG | Create/Edit/Delete nicht implementiert |
| Repeat-Modi | NIEDRIG | Repeat 1/All/Off fehlt |
| Queue-Management | NIEDRIG | Reihenfolge aendern nicht moeglich |

## 7. Credentials

**Speicherort:** `/home/molochzuhause/moloch/.env.spotify`
```
SPOTIPY_CLIENT_ID=<Client ID>
SPOTIPY_CLIENT_SECRET=<Client Secret>
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

**spotifyd Password:** `~/.config/spotifyd/.spotify_pass` (via `password_cmd` gelesen)

Keine hardcodierten Secrets im Code.

## 8. Auth Token Status

| Element | Status |
|---------|--------|
| OAuth Scopes | `user-read-playback-state`, `user-modify-playback-state`, `user-read-currently-playing`, `playlist-read-private`, `user-library-read`, `user-top-read` |
| Token Cache | `~/.cache/spotipy/.cache` — LEER (muss neu generiert werden) |
| Lazy Auth | Controller authentifiziert erst bei erstem Befehl |
| Re-Auth Script | `scripts/spotify_auth.py` |

**Bei Token-Problemen:**
```bash
python3 /home/molochzuhause/moloch/scripts/spotify_auth.py
```
