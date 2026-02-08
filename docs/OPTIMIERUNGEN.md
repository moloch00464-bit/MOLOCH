# M.O.L.O.C.H. Optimierungen
**Stand: 2026-01-24**

## Durchgeführt

### 1. TTS Klarheit verbessert
```python
# Vorher
default_voice = "de_DE-karlsson-low"
pitch_semitones = 2

# Nachher
default_voice = "de_DE-thorsten-high"
length_scale = 1.15
pitch_semitones = 0
```
**Effekt**: Deutlichere, langsamere Sprache

### 2. Timeline Crash-Detection
- `system_startup()` loggt Offline-Zeit
- `system_shutdown()` loggt sauberen Exit
- Crash erkannt wenn letzter Event kein shutdown war

### 3. Interaction Bias
- `brain/bias/interaction_bias.json` erstellt
- In System Prompt integriert
- Sprachstil: trocken, knapp, keine Corporate-Phrasen

### 4. Log Rotation
- hailort.log geleert (1.5MB gespart)
- Alte Logs nach logs/*.bak verschoben

## Empfohlene Optimierungen

### Prio 1: TTS Module aufräumen
```bash
# Problem: 3 TTS-Implementierungen
core/tts.py          # Ungenutzt
core/tts/            # Ungenutzt
core/console/moloch_console.py:MolochTTS  # Aktiv
```
**Aktion**: Ungenutzte löschen oder konsolidieren

### Prio 2: Bare Except fixen
```python
# Vorher
except:
    pass

# Nachher
except Exception as e:
    logger.error(f"Error: {e}")
```

### Prio 3: Paths dynamisch machen
```python
# Vorher (voices.json)
"model_path": "/home/molochzuhause/moloch/models/..."

# Nachher
"model_path": "~/moloch/models/..."  # Oder Path.home()
```

### Prio 4: Whisper Caching
- Model einmal laden, im RAM halten
- Singleton-Pattern existiert bereits
- Prüfen ob RAM-Verbrauch stabil bleibt

### Prio 5: Vision FPS reduzieren
- Aktuell: 5 FPS im Preview
- Könnte auf 2 FPS reduziert werden
- Spart CPU wenn niemand hinschaut

## Nicht empfohlen

### Whisper auf Hailo
- Hailo unterstützt nur tiny/base
- Erkennung zu schlecht
- CPU-small ist besserer Kompromiss

### Mehr RAM kaufen
- Pi5 4GB reicht
- Swap wird kaum genutzt (0B)
- 8GB wäre nur für Whisper-medium nötig
