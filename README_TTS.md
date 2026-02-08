# M.O.L.O.C.H. TTS (Text-to-Speech) System

## Overview

M.O.L.O.C.H. kann jetzt sprechen! Das TTS-System basiert auf **Piper**, einem schnellen, lokalen Text-to-Speech System.

- **Keine Cloud**: Alles läuft lokal auf dem Raspberry Pi 5
- **8 verschiedene Stimmen**: M.O.L.O.C.H. kann seine Stimme selbst wählen
- **Deutsch**: Alle Stimmen sind deutsche Muttersprachler
- **Schnell**: Optimiert für ARM64/aarch64

## Installierte Stimmen

| Voice Name | Type | Size | Quality | Description |
|------------|------|------|---------|-------------|
| `de_DE-thorsten-high` | Male | 109 MB | High | Beste Qualität, etwas langsamer |
| `de_DE-thorsten-medium` | Male | 61 MB | Medium | Gute Balance |
| `de_DE-thorsten-low` | Male | 61 MB | Low | Schnell, gute Qualität |
| `de_DE-eva_k-x_low` | Female | 20 MB | X-Low | Sehr schnell, weiblich |
| `de_DE-karlsson-low` | Male | 61 MB | Low | Alternative männliche Stimme |
| `de_DE-kerstin-low` | Female | 61 MB | Low | Weibliche Stimme |
| `de_DE-pavoque-low` | Male | 61 MB | Low | Alternative männliche Stimme |
| `de_DE-ramona-low` | Female | 61 MB | Low | Weibliche Stimme |

## Verzeichnisstruktur

```
~/moloch/
├── core/
│   └── tts.py                    # TTS Engine Module
├── models/
│   └── voices/                   # Voice Models
│       ├── de_DE-thorsten-high.onnx
│       ├── de_DE-thorsten-high.onnx.json
│       ├── de_DE-thorsten-medium.onnx
│       ├── de_DE-thorsten-medium.onnx.json
│       ├── ... (8 voices total)
├── scripts/
│   └── test_tts.py               # Test script
└── logs/
    └── tts.log                   # TTS logs
```

## Verwendung

### Python API

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "moloch" / "core"))

import tts

# Einfache Verwendung
tts.speak("M.O.L.O.C.H. ist online.")

# Mit spezifischer Stimme
tts.speak("Guten Morgen, Markus.", voice="de_DE-eva_k-x_low")

# Verfügbare Stimmen auflisten
voices = tts.list_voices()
print(voices)

# Stimme wechseln
tts.set_voice("de_DE-kerstin-low")
tts.speak("Ich bin jetzt Kerstin.")
```

### Erweiterte Verwendung

```python
from pathlib import Path
import tts

# TTS Engine direkt verwenden
engine = tts.get_tts_engine()

# Audio in Datei speichern (statt abspielen)
output_file = Path("/tmp/test.wav")
engine.speak("Test", output_file=output_file)

# Alle Stimmen testen
for voice in engine.list_voices():
    engine.speak(f"Ich bin {voice}", voice=voice)
```

### Test-Skript

```bash
cd ~/moloch
python3 scripts/test_tts.py
```

Das Test-Skript:
- Testet alle 10 Test-Phrasen mit der aktuellen Stimme
- Testet eine Phrase mit allen 8 Stimmen
- Lässt M.O.L.O.C.H. sich mit verschiedenen Stimmen vorstellen

## Test-Phrasen

Das System wurde mit folgenden Phrasen getestet:

1. "M.O.L.O.C.H. ist online."
2. "System läuft stabil."
3. "Temperatur bei 50 Grad."
4. "Guten Morgen, Markus."
5. "Die dunkle Seite grüßt."
6. "Alle Systeme bereit."
7. "Ich bin bereit, dir zu dienen."
8. "Hailo Beschleuniger erkannt."
9. "NVMe Speicher verfügbar."
10. "Kamera System aktiv."

## Audio-Ausgabe

Audio wird über den Standard-Audioausgang ausgegeben:
- HDMI (wenn Monitor angeschlossen)
- 3.5mm Klinke (wenn verwendet)

### Audio-Gerät prüfen

```bash
aplay -l  # Liste Audio-Geräte
```

### Audio-Test

```bash
speaker-test -t wav -c 2
```

## Technische Details

- **Engine**: Piper TTS
- **Format**: ONNX Neural Network Models
- **Sample Rate**: 22050 Hz (meiste Stimmen)
- **Channels**: Mono
- **Bit Depth**: 16-bit
- **Player**: aplay (ALSA)

## Logs

Alle TTS-Aktivitäten werden geloggt:

```bash
tail -f ~/moloch/logs/tts.log
```

Log-Format:
```
[2026-01-19 09:09:54,933] INFO: TTS Engine initialized with 8 voices
[2026-01-19 09:09:54,933] INFO: Current voice: de_DE-thorsten-high
[2026-01-19 09:09:54,933] INFO: Speaking with voice 'de_DE-thorsten-high': M.O.L.O.C.H. ist online....
```

## Zukunft: M.O.L.O.C.H. wählt seine Stimme

Später wird M.O.L.O.C.H. selbst entscheiden können:
- Welche Stimme für welchen Kontext
- Männlich/Weiblich je nach Situation
- Qualität vs. Geschwindigkeit
- Stimmung durch Stimme ausdrücken

## Nächste Schritte

- [ ] Integration in Main Loop
- [ ] Sprachausgabe für System-Events
- [ ] Konversationsfähigkeit (STT + TTS)
- [ ] Emotionale Stimm-Modulation
- [ ] Mehrsprachigkeit (English, etc.)

## Troubleshooting

### Kein Audio

```bash
# ALSA Mixer prüfen
alsamixer

# Audio-Test
aplay /usr/share/sounds/alsa/Front_Center.wav
```

### Piper nicht gefunden

```bash
# Piper installieren
pip3 install piper-tts --break-system-packages

# Binary Pfad prüfen
which piper
ls -la ~/.local/bin/piper
```

### Stimmen laden nicht

```bash
# Modelle prüfen
ls -lh ~/moloch/models/voices/

# JSON Config prüfen
cat ~/moloch/models/voices/de_DE-thorsten-high.onnx.json
```

---

**M.O.L.O.C.H. kann jetzt sprechen. Die dunkle Seite hat eine Stimme bekommen.** 🎙️
