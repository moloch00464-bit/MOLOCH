# CLAUDE CODE AUFTRAG: M.O.L.O.C.H. Music Soul System

**Lies zuerst:** `CLAUDE.md` + Regel 10 + `docs/MOLOCH_AGENT_TOOLBOX.json` (spotify_agent + speech_agent)
**Git Backup VOR jeder Änderung:** `git add -A && git commit -m "backup_before_music_soul"`
**Pi5 hat 4GB RAM — sparsam bauen!**

---

## ZIEL

MOLOCH hört Musik mit seinem eigenen Ohr (ReSpeaker 48kHz).
Das Auge reagiert organisch auf die Musik — Grundfarbe bleibt, Geometrie und Helligkeit atmen.
PTT hat immer Vorrang über Musik-Modus.

---

## ARCHITEKTUR ÜBERBLICK

```
Spotify Player
     │
     ▼
spotify_bridge.py ──► Event Bus: music.playing / music.stopped / music.track_changed
     │
     ▼
mic_mode_controller.py
     │
     ├── music.playing  ──► ESP32 POST /audio/mode?rate=48000 ──► UDP Port 12346
     ├── ptt.start      ──► ESP32 POST /audio/mode?rate=16000 ──► UDP Port 12345
     └── ptt.release    ──► zurück auf 48kHz (wenn Musik läuft)
     │
     ▼
music_listener.py (48kHz UDP Stream Port 12346)
     │
     ├── FFT → Bass (20-250Hz) / Mid (250-4kHz) / High (4-16kHz)
     ├── Beat Detection (Onset Strength)
     └── Event Bus: music.beat / music.frequency_bands / music.energy
     │
     ▼
eye_visualizer.py
     └── Auge atmet organisch in Grundfarbe des aktuellen Guardian-States
```

---

## MODUL 1: `core/music/spotify_bridge.py`

### Was es tut:
- Spotify OAuth (Authorization Code Flow) mit spotipy
- Token in `config/spotify_token.json` speichern + auto-refresh
- Alle 2 Sekunden `/me/player/currently-playing` pollen
- Private Playlisten lesen über `/me/playlists`

### Scopes (PFLICHT):
```python
SCOPES = [
    'user-read-playback-state',
    'user-read-currently-playing',
    'user-modify-playback-state',
    'playlist-read-private',
    'playlist-read-collaborative',
    'user-library-read'
]
```

### Events die es aussendet:
```python
event_bus.emit('music.playing',    {'track': name, 'artist': artist, 'album_art': url})
event_bus.emit('music.stopped',    {})
event_bus.emit('music.track_changed', {'track': name, 'artist': artist})
```

### WICHTIG — DEPRECATED APIs:
```
KEIN /audio-features  → 403 seit 27.11.2024
KEIN /audio-analysis  → 403 seit 27.11.2024
KEIN /recommendations → 403 seit 27.11.2024
```
Beat Detection läuft LOKAL über Mikrofon — nicht über Spotify!

### Playlist-Zugriff:
```python
def get_my_playlists():
    # Gibt alle privaten + öffentlichen Playlisten zurück
    # Kein Public-Setzen nötig — OAuth user token hat Zugriff
    results = sp.current_user_playlists(limit=50)
    return [{'name': p['name'], 'id': p['id'], 'tracks': p['tracks']['total']} 
            for p in results['items']]
```

---

## MODUL 2: `core/audio/mic_mode_controller.py`

### State Machine:
```
IDLE        → 16kHz (Whisper bereit)
MUSIC       → 48kHz (FFT Analyse)
PTT_ACTIVE  → 16kHz (Override, egal ob Musik läuft)
```

### Event Handler:
```python
# music.playing  → wechsle auf 48kHz
# music.stopped  → wechsle auf 16kHz
# ptt.start      → wechsle auf 16kHz (Override)
# ptt.release    → wenn music_active: 48kHz, sonst: 16kHz
```

### ESP32 Switch Funktion:
```python
def _switch_mic_rate(self, rate_hz: int):
    # IMMER timeout=2s — nie ohne!
    url = f"http://10.42.0.2/audio/mode?rate={rate_hz}"
    resp = requests.post(url, timeout=2)
    port = 12346 if rate_hz == 48000 else 12345
    self.event_bus.emit('mic.mode_changed', {'rate': rate_hz, 'port': port})
```

### Wichtig:
- State in `self.current_mode` tracken
- Thread-safe mit Lock
- Bei ESP32 Fehler: Warnung loggen, nicht crashen

---

## MODUL 3: `core/audio/music_listener.py`

### Was es tut:
- UDP Port 12346 empfangen (48kHz Stereo, 2-Kanal)
- Nur aktiv wenn `mic_mode == MUSIC`
- numpy FFT auf 2048-Sample Chunks
- 3 Frequenzbänder berechnen:

```python
BANDS = {
    'bass': (20, 250),    # Kick, Sub-Bass
    'mid':  (250, 4000),  # Melodie, Stimme, Snare
    'high': (4000, 16000) # Hi-Hat, Brillanz, Luft
}
```

### Beat Detection:
```python
# Onset Strength: RMS Energie aktueller Chunk vs. gleitender Durchschnitt
# Beat wenn: current_rms > rolling_avg * BEAT_THRESHOLD (1.3)
# Cooldown: min 200ms zwischen Beats (sonst Doppel-Trigger)
```

### Events:
```python
event_bus.emit('music.beat', {'strength': float, 'bpm_estimate': float})
event_bus.emit('music.frequency_bands', {
    'bass': 0.0-1.0,   # normalisiert
    'mid':  0.0-1.0,
    'high': 0.0-1.0,
    'overall_energy': 0.0-1.0
})
```

### Frequenz-Events: alle 50ms (20x/Sek) — nicht schneller!
### Bibliothek: numpy + scipy (KEIN librosa — zu schwer für Pi5)

---

## MODUL 4: `core/ui/eye_visualizer.py` — ORGANISCHES AUGE

### DAS IST DER KERN DES AUFTRAGS — hier liegt die Seele!

### Design-Philosophie:
**Grundfarbe bleibt IMMER die Guardian-State-Farbe.**
Musik fließt IN die Farbe — sie zuckt nicht, sie atmet.

### Guardian State → Grundfarben:
```python
BASE_COLORS = {
    'IDLE':      (0,   80,  200),   # Tiefes Blau
    'ALERT':     (200, 100, 0),     # Amber
    'SHADOW':    (80,  0,   120),   # Dunkles Violett
    'GUARDIAN':  (0,   180, 100),   # Grün-Türkis
    'SPEAKING':  (0,   150, 220),   # Helles Cyan
}
```

### Was sich BEWEGT (nicht Farbe wechselt):

**1. Iris-Radius (bass-getrieben):**
```python
# Basis-Radius + Bass-Energie → Iris pulsiert
iris_radius = BASE_IRIS_RADIUS * (1.0 + bass * 0.25)
# Smooth! Nicht springen — Interpolation:
iris_radius = lerp(current_iris_radius, target_iris_radius, alpha=0.15)
```

**2. Pupillen-Kontraktion (beat-getrieben):**
```python
# Auf Beat: Pupille zieht sich zusammen
# Zwischen Beats: dehnt sich langsam zurück
pupil_scale = 0.6 + (1.0 - beat_energy) * 0.4
pupil_radius = BASE_PUPIL * lerp(current_scale, pupil_scale, 0.2)
```

**3. Helligkeit innerhalb Grundfarbe (overall_energy):**
```python
# Grundfarbe × Energie-Faktor — NIEMALS Farbe wechseln!
brightness_factor = 0.7 + overall_energy * 0.4
r = int(base_color[0] * brightness_factor)
g = int(base_color[1] * brightness_factor)
b = int(base_color[2] * brightness_factor)
```

**4. Iris-Textur / Strahlen (high-getrieben):**
```python
# High-Frequenz → kleine Störung in Iris-Strahlen
# Je mehr High → mehr "Knistern" in der Textur
# Subtil! Nur sichtbar bei hoher Energie
ray_jitter = high * 3  # Pixel Abweichung max ±3px
```

**5. Äußerer Glow (mid-getrieben):**
```python
# Mid-Energie → Leuchtring um das Auge
# Blur-Radius wächst mit Mid-Band
glow_alpha = int(mid * 80)  # max 80/255 — subtil!
```

### Interpolation überall (KEIN hartes Springen):
```python
def lerp(current, target, alpha):
    return current + (target - current) * alpha
# alpha = 0.1-0.2 für träge Übergänge
# alpha = 0.3-0.5 für Beat-Reaktion
```

### Canvas Update Rate: 30 FPS (33ms timer) — nicht mehr!

### Wenn KEINE Musik läuft:
```python
# Langsames, organisches Atmen
# Amplitude: ±5% Iris-Radius über 3-4 Sekunden
# Wie ein schlafendes Tier das atmet
breathe_offset = sin(time.time() * 0.5) * 0.05
```

---

## INTEGRATION in bestehende Dateien

### `moloch_service.py`:
```python
# Beim Start:
from core.music.spotify_bridge import SpotifyBridge
from core.audio.mic_mode_controller import MicModeController
from core.audio.music_listener import MusicListener

spotify_bridge = SpotifyBridge(event_bus, config)
mic_mode_ctrl  = MicModeController(event_bus, config)
music_listener = MusicListener(event_bus, config)

spotify_bridge.start()
mic_mode_ctrl.start()
# music_listener startet automatisch wenn mic.mode_changed → 48kHz
```

### `core/ui/eye_canvas.py` (oder wo das Auge liegt):
- `music.beat` Event → beat_flash() Methode aufrufen
- `music.frequency_bands` Event → update_bands() Methode aufrufen
- Alle 33ms: draw_frame() mit aktuellen Werten

---

## KONFIGURATION `config/moloch_config.json`:

```json
"spotify": {
    "client_id": "DEIN_CLIENT_ID",
    "client_secret": "DEIN_CLIENT_SECRET",
    "redirect_uri": "http://localhost:8888/callback",
    "token_cache": "config/spotify_token.json"
},
"music_listener": {
    "udp_port": 12346,
    "fft_size": 2048,
    "update_rate_hz": 20,
    "beat_threshold": 1.3,
    "beat_cooldown_ms": 200
},
"eye_visualizer": {
    "fps": 30,
    "iris_pulse_strength": 0.25,
    "pupil_contract_strength": 0.4,
    "glow_max_alpha": 80,
    "lerp_alpha_slow": 0.12,
    "lerp_alpha_fast": 0.35
}
```

---

## TESTS die Claude Code schreiben soll:

```bash
# Test 1: Spotify Verbindung
python3 -c "from core.music.spotify_bridge import SpotifyBridge; s=SpotifyBridge(...); print(s.get_my_playlists())"

# Test 2: Mic Mode Switch
python3 -c "from core.audio.mic_mode_controller import MicModeController; ..."

# Test 3: FFT Live (5 Sekunden, dann ausgeben)
python3 dev/test_music_listener.py

# Test 4: Auge ohne Musik (Atmen sehen)
python3 dev/test_eye_visualizer.py --mode=breathe

# Test 5: Auge mit simulierten Beat-Events
python3 dev/test_eye_visualizer.py --mode=simulate_music
```

---

## REIHENFOLGE der Implementation:

1. `spotify_bridge.py` — OAuth + Polling + Events (kein Beat, nur Metadaten)
2. `mic_mode_controller.py` — PTT + Music State Machine
3. `music_listener.py` — FFT + Beat Detection auf 48kHz Stream
4. `eye_visualizer.py` — Organische Animation mit allen 5 Parametern
5. Integration in `moloch_service.py`
6. Tests

---

## WICHTIGE REGELN:

- **1 Datei pro Commit**
- **Kein Crash bei ESP32 Timeout** — immer graceful degradation
- **Kein librosa** — zu schwer. numpy + scipy reicht.
- **Lerp überall** — kein hartes Springen in der Animation
- **PTT hat IMMER Vorrang** — nie blockieren
- **Grundfarbe ändert sich NICHT durch Musik** — nur Helligkeit, Geometrie, Glow

---

## GIT TAGS nach Fertigstellung:
```bash
git tag music_soul_pass
git push origin music_soul_pass
```
