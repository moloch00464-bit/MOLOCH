# M.O.L.O.C.H. - Multi-Objective Localized Observation & Communication Hub

> **"Die dunkle Seite macht mehr Spass!"**

Lies `moloch_context.json` fuer vollstaendige Hardware-Specs und Projektdetails.

## Wer ist Markus?

Markus ("First Moloch"), 47, DGM-Anlagenfahrer mit 25 Jahren Industrieautomation bei Honsel in Nuernberg. Arbeitet mit KUKA/ABB Robotern, SPS-Steuerungen, 400 bar Hydraulik. Kann loeten, crimpen, 230V, 3D-Druck, Hardware-Debugging. Wenn er was ueber Schaltungen sagt: **ZUHOEREN - er hat recht!**

Kommunikationsstil: Direkt, Kumpel-Level, Dark Humor. Kein Corporate-Sprech, kein Cheerleader, keine Bullet-Point-Orgien. Fraenkisch. Kompliziert != besser.

## System-Ueberblick

```
+---------------------------------------------------------+
|                    M.O.L.O.C.H. v3.x                    |
|                                                         |
|  BRAIN:  Raspberry Pi 5 (192.168.178.24)                 |
|          NVMe SSD, SSH User: markus                     |
|  NPU:    Hailo-10H (40 TOPS Edge AI Inferenz)           |
|                                                         |
|  AUGEN:                                                 |
|   - Sonoff CAM-PT2    192.168.178.25  RTSP+ONVIF, PTZ    |
|   - Seed AI Cam       WiFi           Eigener NPU        |
|   - DIFANG 3-Kopf     (WGT geplant)  3x PTZ Skulptur    |
|                                                         |
|  OHREN:                                                 |
|   - SmartMic          Bluetooth      Gepairt, aktiv!    |
|   - ReSpeaker Lite    USB            Noch nicht inst.   |
|                                                         |
|  STIMME:                                                |
|   - HDMI Audio        aktiv          TTS funktioniert!  |
+---------------------------------------------------------+
```

## Kamera-Zugriff (GETESTET)

### Sonoff CAM-PT2 (aktiv seit 2026-02-01)
```bash
# RTSP Stream - FUNKTIONIERT
ffplay rtsp://USER:PASS@CAMERA_IP:554/av_stream/ch0

# Stream-Info: 1920x1080 @ 20fps, H.264, Audio: PCM A-Law 8kHz

# Python OpenCV
import cv2
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture("rtsp://USER:PASS@CAMERA_IP:554/av_stream/ch0", cv2.CAP_FFMPEG)
```

## Netzwerk

| Geraet | IP | Protokoll |
|--------|-----|-----------|
| Pi5 (Brain) | 192.168.178.24 | SSH |
| Sonoff CAM-PT2 | 192.168.178.25 | RTSP/ONVIF |
| Subnetz | 192.168.178.0/24 | |

## Audio-Setup

### SmartMic (Bluetooth - aktiv)
```bash
# Bluetooth Address: 54:B7:E5:AA:3B:8E
bluetoothctl info 54:B7:E5:AA:3B:8E
```

### TTS Ausgabe (HDMI - aktiv)
```bash
espeak -v de "MOLOCH ist online. Subjekt erfasst." --stdout | aplay
```

## Coding-Regeln

- **Sprache**: Python bevorzugt
- **Kommentare**: Deutsch
- **Stil**: Pragmatisch, DGM-Grade Zuverlaessigkeit
- **Testing**: Immer erst testen bevor deployed wird
- **Prinzip**: Kompliziert != besser
- **Diagnose**: Wichtige Infos sofort sagen, nicht auf Nachfrage warten

## Snapshots

Kamera-Snapshots werden gespeichert unter: `/home/molochzuhause/moloch/snapshots/`

## Vision Pipeline

Detaillierte Pipeline-Config: `config/moloch_vision_pipeline.json`

### Verteilte AI-Architektur
```
Seeed AI Cam (Ethos-U55)     Sonoff CAM-PT2 (1080p)
       |                            |
       | "Person erkannt!"          | RTSP Stream
       +------------+---------------+
                    |
              Pi5 + Hailo-10H (40 TOPS)
              Face Detection -> Recognition
                    |
         +----------+----------+
         |          |          |
      Signal     TTS       Dashboard
      Alerts    HDMI      (geplant)
```

### Face Recognition Crew-DB
```
/home/molochzuhause/moloch/faces/
  |- ray/
  |- lilly/
  |- meise/
  |- sven/
  |- franzi/
  |- markus/
  |- unknown/
```
Pro Person 3-5 Fotos aus verschiedenen Winkeln hinzufuegen.

### ONVIF PTZ Steuerung
```python
from onvif import ONVIFCamera
cam = ONVIFCamera('CAMERA_IP', 80, 'CHANGE_ME', 'CHANGE_ME')
# PTZ: 340 Grad horizontal, 180 Grad vertikal
```

### Autonome Kamerasteuerung
Config: `config/hardware_autonomy.json`
Controller: `core/hardware/sonoff_ptz_controller.py`

Modi:
- **FOLLOW_MARKUS**: Kamera folgt Markus wenn erkannt
- **SEARCH_MODE**: Langsamer Scan wenn Markus verschwindet (max 120s)
- **IDLE_TRACKING**: Mikrobewegungen bei Voice-Interaktion

Sicherheitslimits:
- Nachtsperre 23:00-06:00
- Max 20 Bewegungen/Minute
- Manual Override: "Moloch, stopp Kamera"

### Kontrollierte Autonomie
Config: `config/controlled_autonomy.json`

Autonomie-Level:
- **Level 1 (Reaktiv)**: Sensorwerte einbauen, Kontext erkennen - AKTIV
- **Level 2 (Suggestiv)**: Vorschläge machen, Bestätigung erforderlich - AKTIV
- **Level 3 (Autonom)**: Selbständige Ausführung - DEAKTIVIERT

Subtilitäts-Engine:
- Erkennt indirekte Sprache ("Boah ist das warm hier")
- Prüft Sensorwerte und formuliert Vorschlag
- Führt NICHTS ohne Bestätigung aus

Speech-Style: Trocken, knapp, leicht ironisch. Kein Dauerkommentar.

```python
from core.hardware import SonoffPTZController, AutonomyMode
ptz = SonoffPTZController()
ptz.connect()
ptz.follow_face(face_x, face_y, frame_w, frame_h)
```

### Unified CameraController (beide Kameras)
```python
from core.hardware import get_camera_controller, CameraID

ctrl = get_camera_controller()
ctrl.connect_all()  # Verbindet Sonoff PTZ
ctrl.start_capture()  # Startet Background-Threads

# Frames holen
frame = ctrl.get_frame(CameraID.SONOFF)
both = ctrl.get_both_frames()

# PTZ steuern
ctrl.ptz_follow_face(face_x, face_y, w, h)
ctrl.ptz_center()
ctrl.ptz_stop()  # Manual Override

# Snapshot
ctrl.take_snapshot()  # -> /home/molochzuhause/moloch/snapshots/
```

## WGT 2026 Crew

Signal-Gruppe "SNG": Ray, Lilly, Meise, Sven, Franzi, Markus
