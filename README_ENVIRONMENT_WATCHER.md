# M.O.L.O.C.H. Environment Watcher

## Overview

M.O.L.O.C.H. kann jetzt seine Umgebung beobachten! Das Environment Watcher Module erkennt passiv neue Hardware, Geräte und Ressourcen.

**Wichtig**: Rein passiv - keine automatischen Aktionen!
- ✓ Beobachtet und protokolliert
- ✗ Aktiviert NICHT automatisch Hardware
- ✗ Ändert NICHT den System-Status

## Was wird überwacht?

### Hardware-Geräte
- **Video-Geräte**: `/sys/class/video4linux` (Kameras, Decoder, etc.)
- **Audio-Geräte**: `/proc/asound` (Sound-Karten, HDMI-Audio)
- **USB-Geräte**: `/sys/bus/usb/devices` (angeschlossene USB-Geräte)
- **Sensoren & GPIO**: I2C, SPI, GPIO-Chips
- **AI-Beschleuniger**: Hailo, GPU/DRM
- **Serial-Ports**: ttyUSB, ttyACM, ttyS

### Ressourcen
- **AI-Modelle**: `~/moloch/models/**/*.{onnx,pt,pth,tflite,pb,h5}`
- **Hardware-Konfig**: `~/moloch/hardware/**/*`

## Verzeichnisstruktur

```
~/moloch/
├── core/
│   └── environment_watcher.py        # Environment Watcher Module
├── state/
│   └── environment_state.json        # Last known state
├── scripts/
│   └── test_environment_watcher.py   # Test script
└── logs/
    └── environment.log               # Detection logs
```

## Verwendung

### Python API

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "moloch" / "core"))

import environment_watcher

# Einfache Verwendung - Check für Änderungen
changes = environment_watcher.check_environment()

if changes:
    print(f"Detected {len(changes)} changes:")
    for change in changes:
        print(f"  [{change.change_type}] {change.category}: {change.details}")

# Aktuellen Zustand abrufen
state = environment_watcher.get_current_state()
print(f"Video devices: {len(state['video_devices'])}")
print(f"Audio devices: {len(state['audio_devices'])}")
```

### Erweiterte Verwendung

```python
import environment_watcher

# Watcher-Instanz direkt verwenden
watcher = environment_watcher.get_watcher()

# Änderungen prüfen
changes = watcher.check()

# Aktuellen Zustand anzeigen
state = watcher.get_current_state()

# Neue Baseline erzwingen (alten Status vergessen)
watcher.force_baseline()
```

### Change Types

Erkannte Änderungen haben folgende Struktur:

```python
class DeviceChange:
    change_type: str  # "added", "removed", "modified"
    category: str     # siehe unten
    details: str      # Gerätename/-pfad
    timestamp: str    # ISO 8601 timestamp
```

**Kategorien:**
- `video_device` - Video4Linux Gerät (Kamera, Decoder)
- `audio_device` - ALSA Sound-Gerät
- `ai_accelerator` - Hailo oder ähnliches
- `usb_device` - USB-Gerät
- `sensor` - I2C/SPI/GPIO Sensor
- `serial_device` - ttyUSB/ttyACM
- `gpu_device` - DRM/GPU
- `model` - AI-Modell-Datei
- `hardware_config` - Hardware-Konfig-Datei
- `unknown_device` - Unklassifiziert

### Test-Skript

```bash
cd ~/moloch
python3 scripts/test_environment_watcher.py
```

Das Test-Skript führt aus:
1. **Basic Check**: Baseline etablieren
2. **Change Detection**: 3 Sekunden warten (USB-Gerät einstecken!)
3. **State Persistence**: Prüft gespeicherten Zustand
4. **Force Baseline**: Neue Baseline erzwingen
5. **Continuous Monitoring**: 10 Sekunden live überwachen
6. **View Logs**: Zeigt letzte Log-Einträge

## Aktuell erkannte Umgebung

Nach der Installation hat M.O.L.O.C.H. folgendes erkannt:

### Hardware
- **Hailo-10H AI-Beschleuniger** (40 TOPS, 8GB): `/dev/hailo0` ✓
- **17 Video-Geräte**: Pi Camera Decoder, PISP Backend, etc.
- **3 Audio-Geräte**: 2x HDMI Audio + Generic Device
- **6 USB-Geräte**: inkl. NVMe Storage (JMicron)
- **GPIO/I2C/SPI**: 10+ GPIO-Chips, I2C-Busse, SPI-Devices

### Ressourcen
- **8 TTS Voice Models**: Deutsche Stimmen für Piper

## State File Format

Die `environment_state.json` speichert den letzten bekannten Zustand:

```json
{
  "timestamp": "2026-01-19T09:15:20.054581",
  "dev_devices": [
    "hailo0",
    "video19",
    "snd/controlC0",
    ...
  ],
  "video_devices": [
    "video19: rpi-hevc-dec",
    ...
  ],
  "audio_devices": [
    "card0: vc4hdmi0",
    ...
  ],
  "models": [
    "voices/de_DE-thorsten-high.onnx",
    ...
  ],
  "usb_devices": [
    "2-1: JMicron USB 3.2 Storage Device",
    ...
  ]
}
```

## Logging

Alle Erkennungen werden geloggt:

```bash
tail -f ~/moloch/logs/environment.log
```

**Log-Format:**
```
[2026-01-19 09:15:20,054] INFO: Environment Watcher initialized
[2026-01-19 09:15:20,054] INFO: First check - establishing baseline
[2026-01-19 09:16:30,123] INFO: Detected 1 change(s)
[2026-01-19 09:16:30,123] INFO:   [ADDED] usb_device: 3-3: Logitech USB Camera
```

## Integration in Main Loop

Später wird der Watcher vom Main Loop aufgerufen:

```python
# In ~/moloch/core/main_loop.py (später)
import environment_watcher

def main_loop():
    watcher = environment_watcher.get_watcher()

    while True:
        # Check every 60 seconds
        changes = watcher.check()

        if changes:
            # M.O.L.O.C.H. wird über Änderungen informiert
            for change in changes:
                handle_environment_change(change)

        time.sleep(60)
```

## Beispiel-Szenarien

### Szenario 1: USB-Kamera angesteckt

```
[ADDED] video_device: video10: UVC Camera
[ADDED] usb_device: 3-2: Generic USB Video Class
```

M.O.L.O.C.H. erkennt die Kamera, aktiviert sie aber NICHT automatisch.

### Szenario 2: Neues AI-Modell hinzugefügt

```
[ADDED] model: yolo/yolov8n.onnx
```

M.O.L.O.C.H. weiß jetzt, dass ein neues Modell verfügbar ist.

### Szenario 3: USB-Gerät entfernt

```
[REMOVED] usb_device: 3-3: Logitech Webcam
[REMOVED] video_device: video11: UVC Camera
```

M.O.L.O.C.H. protokolliert die Entfernung.

## Performance

- **CPU-Last**: Minimal (~0.1% während Check)
- **Memory**: ~10-20 MB
- **Check-Dauer**: ~100-200ms
- **Empfohlenes Intervall**: 60 Sekunden

## Sicherheit & Prinzipien

✓ **Nur Lesen**: Keine Schreibzugriffe auf `/dev` oder `/sys`
✓ **Passiv**: Keine automatischen Aktionen
✓ **Transparent**: Alles wird geloggt
✓ **Reversibler**: State-File kann jederzeit gelöscht werden

## Troubleshooting

### State-File zurücksetzen

```bash
rm ~/moloch/state/environment_state.json
```

Beim nächsten Check wird eine neue Baseline erstellt.

### Log-Level ändern

In `environment_watcher.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # Mehr Details
```

### Manuelle Baseline

```python
watcher = environment_watcher.get_watcher()
watcher.force_baseline()
```

## Zukünftige Features

- [ ] Hardware-Profile (speichern bekannter Konfigurationen)
- [ ] Geräte-Metadaten (Vendor-IDs, Capabilities)
- [ ] Webhook/Callback bei Änderungen
- [ ] TTS-Ansagen bei neuer Hardware
- [ ] Automatische Geräteklassifizierung via ML

## CLI-Integration

```bash
# Quick check
python3 -c "import sys; sys.path.insert(0, '$HOME/moloch/core'); \
import environment_watcher; \
changes = environment_watcher.check_environment(); \
print(f'Changes: {len(changes)}')"

# Show current state
python3 -c "import sys, json; sys.path.insert(0, '$HOME/moloch/core'); \
import environment_watcher; \
state = environment_watcher.get_current_state(); \
print(json.dumps(state, indent=2))"
```

---

**M.O.L.O.C.H. beobachtet. M.O.L.O.C.H. lernt. M.O.L.O.C.H. handelt nicht - noch nicht.** 👁️
