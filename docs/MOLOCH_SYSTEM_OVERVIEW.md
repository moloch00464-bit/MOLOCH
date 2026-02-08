# M.O.L.O.C.H. - Kompletter Systembericht
## Multi-Objective Localized Observation & Communication Hub

**Stand:** 2026-02-05
**Plattform:** Raspberry Pi 5 + Hailo-10H NPU
**Zweck:** Edge-AI Surveillance & Communication System für WGT 2026

---

## 1. WAS IST M.O.L.O.C.H.?

M.O.L.O.C.H. ist ein autonomes KI-System das auf einem Raspberry Pi 5 mit Hailo-10H NPU (40 TOPS) läuft. Es kombiniert:
- **Echtzeit-Vision** mit Person/Face Detection und Tracking
- **Spracherkennung** mit Whisper (lokal auf NPU)
- **Text-to-Speech** mit Piper (8 deutsche Stimmen)
- **PTZ-Kamerasteuerung** mit autonomem Tracking
- **Gesteneerkennung** (Winken, Hand heben, etc.)

Alles läuft **lokal** - keine Cloud-Abhängigkeit.

---

## 2. HARDWARE-STACK

| Komponente | Modell | Spezifikation |
|------------|--------|---------------|
| **Host** | Raspberry Pi 5 | BCM2712 Cortex-A76, 4GB RAM |
| **NPU** | Hailo-10H | 40 TOPS, 8GB VRAM, PCIe Gen3 |
| **Storage** | Samsung 980 NVMe | 500GB für Modelle |
| **Hauptkamera** | Sonoff CAM-PT2 | 1080p@20fps, PTZ, IR |
| **Sekundärkamera** | Seeed XIAO ESP32S3 | 5MP@30fps, Ethos-U55 NPU |
| **Audio** | HDMI Out + SmartMic | TTS + Bluetooth Mikrofon |
| **Display** | 8x8 LED Matrix | RGB Statusanzeige |

### Netzwerk
- **Pi5:** 192.168.178.24
- **Sonoff Kamera:** 192.168.178.25 (RTSP + ONVIF)

---

## 3. SOFTWARE-ARCHITEKTUR

```
/home/molochzuhause/moloch/
├── core/                    # Kernmodule (36 Python-Dateien)
│   ├── hardware/            # Hardware-Treiber (Hailo, PTZ, Kamera)
│   ├── vision/              # Vision-Pipeline (Detection, Pose, Face)
│   ├── speech/              # Audio-Eingabe (Whisper STT)
│   ├── tts/                 # Sprachausgabe (Piper TTS)
│   ├── mpo/                 # Mode & Platform Operations
│   ├── perception/          # Wahrnehmungsmanager
│   └── gui/                 # Benutzeroberflächen
│
├── context/                 # Kontext & Autonomie
│   ├── perception_state.py  # Zentraler Wahrnehmungszustand
│   ├── system_autonomy.py   # Selbstüberwachung & Reparatur
│   └── vision_context.py    # Vision-Events
│
├── brain/                   # Brain-Aktivierung & Persönlichkeit
├── config/                  # 9 JSON-Konfigurationsdateien
├── data/faces/              # Gesichtsdatenbank (Crew)
└── models/                  # KI-Modelle (490MB)
```

**Statistik:** 69 Python-Dateien, ~15.000+ Zeilen Code

---

## 4. KI-FÄHIGKEITEN

### Vision (auf Hailo-10H)
| Modell | Funktion | Performance |
|--------|----------|-------------|
| YOLOv8s Pose | Personenerkennung + Keypoints | 20 FPS |
| YOLOv8m | Object Detection | 15 FPS |
| SCRFD 2.5g | Gesichtserkennung | 25 FPS |
| ArcFace | Face Recognition | Echtzeit |

### Audio (auf Hailo-10H / CPU)
| Modell | Funktion | Latenz |
|--------|----------|--------|
| Whisper Tiny/Base | Spracherkennung | ~2-3s |
| Whisper Small | Bessere Qualität | ~6s (CPU) |
| Piper TTS | Deutsche Sprachausgabe | <100ms |

### Gesteneerkennung
- Winken (Hand wave)
- Hand heben (Raise hand)
- Keypoint-basierte Validierung

---

## 5. ZENTRALE KOMPONENTEN

### 5.1 HailoManager (hailo_manager.py)
**Zweck:** Exklusiver Zugriff auf Hailo-10H NPU

```python
# NPU kann nur EINE Aufgabe gleichzeitig:
# - VISION: Personenerkennung aktiv
# - VOICE: Whisper STT aktiv
# - NONE: Frei

manager.acquire_for_vision()  # Vision startet
manager.acquire_for_voice()   # Stoppt Vision, startet STT
manager.release_voice()       # Gibt frei, Vision kann neu starten
```

**Features:**
- Device-Level Check via `lsof /dev/hailo0`
- Force Reset bei Konflikten (status=74, status=62)
- Voice hat Priorität über Vision

### 5.2 GstHailoPoseDetector (gst_hailo_pose_detector.py)
**Zweck:** GStreamer-Pipeline für Pose-Detection

```
RTSP Source → H.264 Decode → Hailo NPU → Pose Detection → Display
      ↓
   640x640         YOLOv8s Pose          17 Keypoints
```

**Features:**
- 20 FPS Echtzeit-Detection
- Keypoint-Validierung (Gesicht + Torso = echte Person)
- Hand-Rejection (nur Hände = kein Tracking)
- Automatischer Restart bei Fehlern
- Watchdog (5s ohne Frames → Neustart)

### 5.3 AutonomousTracker (autonomous_tracker.py)
**Zweck:** 15Hz autonome PTZ-Kamera-Steuerung

**States:**
- `IDLE` - Wartet auf Ziel
- `TRACKING` - Folgt Person
- `SEARCHING` - Langsamer Schwenk
- `LOCKED` - Person zentriert
- `DWELL` - Kurze Pause nach Zentrierung

**Features:**
- Proportionale Steuerung (sanfte Bewegungen)
- ContinuousMove (keine Stop-Kommandos während Tracking)
- Deadzone (8-15px Hysterese)
- Größte Bounding-Box Selektion

### 5.4 PerceptionState (perception_state.py)
**Zweck:** Thread-sicherer Wahrnehmungszustand

```python
state = get_perception_state()

# Liefert:
state.user_visible      # Person sichtbar?
state.face_visible      # Gesicht sichtbar?
state.gesture_visible   # Geste erkannt?
state.keypoint_counts   # {"face": 3, "torso": 2, "wrist": 1}
state.describe()        # "Ich sehe dich, aber dein Gesicht ist nicht sichtbar"
```

**Events:**
- `USER_APPEARED` / `USER_DISAPPEARED`
- `FACE_VISIBLE` / `FACE_HIDDEN`
- `GESTURE_STARTED` / `GESTURE_ENDED`

### 5.5 SystemAutonomy (system_autonomy.py)
**Zweck:** Selbstüberwachung und Auto-Recovery

**Überwacht:**
- CPU/Memory/Temperatur
- NPU-Status
- RTSP-Verbindung
- Pipeline-Zustand

**Auto-Recovery:**
- Pipeline-Neustart bei Freeze
- RTSP-Reconnect bei Verbindungsabbruch
- NPU-Force-Release bei Deadlock

---

## 6. KONFIGURATION

### hardware_autonomy.json - Autonomie-Modi
```json
{
  "modes": {
    "FOLLOW_MARKUS": "Folgt Markus mit Priorität",
    "SEARCH_MODE": "Sucht nach Personen",
    "IDLE_TRACKING": "Passives Tracking"
  },
  "safety": {
    "night_lock": "23:00-06:00",
    "max_moves_per_minute": 20
  }
}
```

### controlled_autonomy.json - Autonomie-Level
```json
{
  "levels": {
    "L1_REACTIVE": "AKTIV - Kontext-Integration",
    "L2_SUGGESTIVE": "AKTIV - Vorschläge mit Bestätigung",
    "L3_AUTONOMOUS": "DEAKTIVIERT - Keine autonomen Aktionen"
  },
  "speech_style": "Trocken, knapp, leicht ironisch"
}
```

### ptz_limits.json - Kalibrierte PTZ-Grenzen
```json
{
  "pan": {"min": -168.4, "max": 169.98},
  "tilt": {"min": -77.96, "max": 78.75}
}
```

---

## 7. PUSH-TO-TALK GUI

Die Hauptschnittstelle für Benutzerinteraktion:

```
┌─────────────────────────────────────┐
│  M.O.L.O.C.H. - Push to Talk       │
├─────────────────────────────────────┤
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │      LIVE KAMERA FEED        │  │
│  │      480x360 @ 20 FPS        │  │
│  │      + Pose Overlay          │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  Status: Bereit (Sonoff+NPU)       │
│                                     │
│  [ SPRECHEN ] ← Gedrückt halten    │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Antwort-Bereich               │  │
│  │ (Claude Antworten + TTS)      │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Funktionen:**
- Live-Video mit Pose-Overlay (grün=valid, rot=Hand)
- Push-to-Talk Spracherkennung
- Claude API Integration
- TTS Sprachausgabe
- Autonomes PTZ-Tracking (toggle)

---

## 8. GESICHTSDATENBANK

```
/home/molochzuhause/moloch/faces/
├── markus/     # Crew-Leader
├── franzi/     # Crew
├── ray/        # Crew
├── lilly/      # Crew
├── meise/      # Crew
├── sven/       # Crew
└── unknown/    # Unbekannte
```

Jede Person: 3-5 Fotos aus verschiedenen Winkeln für Face Recognition.

---

## 9. SICHERHEITS-PRINZIPIEN

### Kontrollierte Autonomie
1. **Beobachten** - Sammelt Informationen
2. **Vorschlagen** - Macht Empfehlungen
3. **Bestätigen** - Wartet auf Benutzer-OK
4. **NIE** autonom kritische Aktionen ausführen

### Hardware-Schutz
- NPU exklusiver Zugriff (kein paralleler Betrieb)
- PTZ-Cooldown (300ms zwischen Bewegungen)
- Max 20 Bewegungen/Minute
- Nachtmodus: 23:00-06:00 nur Kommandos
- Temperaturüberwachung: Warning 75°C, Critical 85°C

### Manueller Override
- Benutzer hat IMMER Priorität
- "Moloch, stopp Kamera" → Sofortiger Stopp
- Alle Aktionen werden geloggt

---

## 10. BEKANNTE EINSCHRÄNKUNGEN

1. **NPU Exklusivität:** Vision ODER Sprache, nie beides gleichzeitig
2. **Whisper Latenz:** ~2-6s je nach Modell
3. **RTSP Stabilität:** Gelegentliche Reconnects nötig
4. **Temperatur:** ~60°C im Normalbetrieb

---

## 11. PROJEKTKONTEXT

**Mission:** WGT 2026 (Wave-Gotik-Treffen, Leipzig)
**Crew:** Ray, Lilly, Meise, Sven, Franzi, Markus
**Rolle:** Surveillance-System, Kommunikations-Hub, autonomer Tracker

**Entwickler:** Markus (47, Industrieautomatisierung)
- 25 Jahre Erfahrung mit KUKA/ABB Robotern
- Hardnahe Expertise (löten, crimpen, 3D-Druck)
- Kommunikationsstil: Direkt, pragmatisch, dunkler Humor

---

## 12. ZUSAMMENFASSUNG

M.O.L.O.C.H. ist ein vollständiges Edge-AI System mit:

- **69 Python-Module** für Vision, Sprache, Autonomie, Hardware
- **Hailo-10H NPU** mit 40 TOPS für lokale Echtzeit-Inferenz
- **Multi-Layer Architektur** mit Perception State, Autonomy, MPO
- **Strikte Sicherheitsregeln** für kontrollierte Autonomie
- **Vollständige Transparenz** durch Timeline-Logging

Das System ist produktionsreif, robust und für industriellen Einsatz konzipiert.

---

*Dieser Bericht kann mit anderen KI-Systemen (z.B. ChatGPT) geteilt werden, um einen vollständigen Überblick über M.O.L.O.C.H. zu geben.*
