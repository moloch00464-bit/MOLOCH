# M.O.L.O.C.H. Known Issues
**Stand: 2026-01-24**

## Kritisch

*Keine kritischen Issues*

## Mittel

### 1. Doppelte TTS-Module
- `core/tts.py` (Datei) und `core/tts/` (Ordner) existieren parallel
- Python importiert den Ordner, nicht die Datei
- Aktuell wird keins davon benutzt (TTS ist in moloch_console.py)
- **Empfehlung**: Aufräumen, ungenutzte Module entfernen

### 2. Bare Except Clauses
- 10+ Stellen mit `except:` statt `except Exception:`
- Fängt auch SystemExit und KeyboardInterrupt
- **Dateien**: status.py, eye_viewer.py, tts.py, environment_watcher.py

### 3. Hardcoded Paths
- `core/tts/config/voices.json` hat absolute Pfade
- Bricht wenn User-Home sich ändert
- **Empfehlung**: Path.home() verwenden

## Niedrig

### 4. Fehlender Console Desktop Shortcut
- Nur "Control" und "Talk" auf Desktop
- "Console" Shortcut fehlt

### 5. Whisper auf CPU
- Hailo-10H unterstützt nur tiny/base Modelle
- Small läuft auf CPU (~6s Latenz)
- Akzeptabel, aber nicht optimal

### 6. Temperatur 60°C
- Leicht erhöht aber im grünen Bereich
- Bei Dauerlast überwachen

## Behoben (2026-01-24)

- [x] Log Rotation: hailort.log bereinigt (1.5MB → 0)
- [x] Timeline Crash-Detection implementiert
- [x] TTS Nuscheln behoben (pitch=0, length_scale=1.15)
- [x] Interaction Bias integriert
