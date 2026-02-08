# M.O.L.O.C.H. System Status
**Audit: 2026-01-24**

## Hardware

| Komponente | Status | Details |
|------------|--------|---------|
| Raspberry Pi 5 | OK | 4GB RAM, 60.4°C |
| Hailo-10H NPU | OK | v5.1.1, 40 TOPS |
| XIAO Vision AI | OK | /dev/ttyACM0 |
| Bluetooth | OK | AIRY OPEN TWS connected |
| WLAN | OK | 192.168.2.124 |
| NVMe SSD | OK | 458GB, 4% belegt |

## Software

| Modul | Version/Config | Status |
|-------|----------------|--------|
| Whisper STT | small, CPU, int8 | OK |
| Piper TTS | thorsten-high, 1.15x | OK |
| Claude API | claude-sonnet-4 | OK |
| Timeline | 341 Events | OK |

## Performance

| Komponente | Latenz |
|------------|--------|
| TTS | 5.2s / 20 Zeichen |
| Vision | 0.18s |
| Claude API | 2.2s |
| Whisper | ~6s / Satz |

## Speicher

```
~/moloch Verzeichnis:
- models/         490 MB (Voices)
- firmware/        23 MB (Hailo)
- logs/           944 KB
- state/           52 KB (Timeline)
- core/           672 KB (Code)
- brain/          116 KB (Context)
```
