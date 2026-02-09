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
|  BRAIN:  Raspberry Pi 5 (192.168.178.24, SSH: moloch)                 |
|          2x NVMe SSD, SSH User: molochzuhause                     |
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

## Storage-Architektur (2x NVMe SSD)

KRITISCH: Pfade NIEMALS aendern ohne diese Struktur zu verstehen!

### SSD 1: System-SSD (sdb, 465 GB, ext4)
```
/               458 GB  - Raspberry Pi OS, Home, Code
/boot/firmware  510 MB  - Boot Partition (vfat)

Wichtige Pfade:
  /home/molochzuhause/moloch/           <- M.O.L.O.C.H. Repo (22 GB genutzt)
  /home/molochzuhause/moloch/models/    <- Piper TTS Voices (490 MB)
    voices/de_DE-thorsten-high.onnx      109 MB (Guardian/Shadow Stimme)
    voices/de_DE-karlsson-low.onnx        61 MB (Kobold Stimme)
    voices/de_DE-*.onnx                   ~61 MB pro Stimme (8 Stimmen)
  /home/molochzuhause/moloch/config/    <- JSON Configs
  /home/molochzuhause/moloch/core/      <- Python Source Code
  /home/molochzuhause/.local/bin/piper  <- Piper TTS Binary
```

### SSD 2: Daten-SSD (sda, 477 GB, NTFS)
```
/mnt/moloch-data/                       <- AI Daten + Modelle
  hailo/models/                          <- HEF Modelle fuer Hailo-10H (AKTIV)
    scrfd_10g.hef                         5.8 MB (Face Detection, 47 FPS)
    arcface_mobilefacenet.hef             2.6 MB (Face Recognition, 498 FPS)
    yolov8m_h10.hef                      21 MB (Person Detection, 39 FPS)
    yolov8s_pose_h10.hef                 14 MB (Pose/Keypoints, 36 FPS)
    (+ weitere HEF Modelle)
  hailo/drivers/                         <- Hailo Treiber
  hailo/config/                          <- Hailo Konfiguration
  hailo/repos/                           <- Hailo Repos & Referenz-Code
  hailo/cache/                           <- Build Cache
  qdrant/                                <- Qdrant Vector DB Storage
```

### Mount in /etc/fstab
```
PARTUUID=9d4d38d4-02  /               ext4    defaults,noatime  0  1
UUID=F4BE3BC4BE3B7E64  /mnt/moloch-data ntfs3  uid=1000,gid=1000,nofail 0 0
```

### Swap: 2 GB zram + 2 GB loop (4 GB total)

### HAILO MODELLE & AI DATEN:
SSD2 = /mnt/moloch-data/ (477GB NTFS) ist die EINZIGE Ablage fuer AI-Modelle!
Hier liegen ALLE Hailo HEF-Modelle, Whisper Modelle, AI Repos, Qdrant DB.
NIEMALS Modelle auf SSD1 suchen oder dorthin kopieren!
Vor jedem Modell-Task: `ls /mnt/moloch-data/hailo/models/` checken!

Aktive Hailo-10H Modelle (HailoRT 5.1.1, alle H10-nativ):
```
/mnt/moloch-data/hailo/models/
  scrfd_10g.hef              5.8 MB  Face Detection (640x640, ~47 FPS)
  arcface_mobilefacenet.hef  2.6 MB  Face Recognition (112x112, ~498 FPS)
  yolov8m_h10.hef           21 MB    Person Detection (640x640, ~39 FPS)
  yolov8s_pose_h10.hef      14 MB    Pose/Keypoints (640x640, ~36 FPS)
```

### REGELN:
- HEF Modelle IMMER auf /mnt/moloch-data/hailo/models/
- Piper Voices IMMER auf ~/moloch/models/voices/
- Code IMMER auf ~/moloch/core/
- NTFS-SSD ist uid=1000 gemountet (molochzuhause), kein chmod moeglich
- Bei "disk full" IMMER zuerst /mnt/moloch-data pruefen (477 GB fast leer)
- H8L HEF Modelle sind NICHT kompatibel mit Hailo-10H (Error 93)
- HailoRT 5.1.1 API: `configured.run([bindings], timeout=10000)` - Bindings als LISTE!

## SSH-Zugang

```bash
ssh molochzuhause@moloch        # oder ssh molochzuhause@192.168.178.24
# User: molochzuhause (NICHT markus!)
# Home: /home/molochzuhause/
# Non-interactive Shell: .bashrc wird NICHT geladen, Env Vars stehen in ~/.profile
```

## Aktive Controller-Architektur (Stand 2026-02-08)

### Kamera-Controller (Hybrid: ONVIF + eWeLink Cloud)
```python
# ONVIF PTZ (AbsoluteMove, volle 342.8 Grad)
from core.hardware.sonoff_camera_controller import SonoffCameraController, get_camera_controller
cam = get_camera_controller()  # Singleton
cam.connect()
cam.move_absolute(pan_deg, tilt_deg, speed)
cam.get_position()  # -> PTZPosition(pan, tilt, zoom)

# eWeLink Cloud (LED, IR, Alarm, Audio, Sleep)
from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig

# ACHTUNG: Pan-Achse ist INVERTIERT! x=+0.5 = LINKS, x=-0.5 = RECHTS
# Pan range: -168.4 (LINKS) bis 174.4 (RECHTS)
# Tilt range: -78.8 (runter) bis 101.3 (hoch)
```

### Personality Engine (Dual: Guardian/Shadow)
```python
from core.personality import get_personality_engine, MolochEvent
engine = get_personality_engine()
engine.speak_event(MolochEvent.GOOD_MORNING)            # Eine Stimme
engine.speak_event(MolochEvent.PERSON_UNKNOWN, conflict=True)  # Beide Stimmen
engine.manual_override("Moloch, Schatten")               # Switch
```
