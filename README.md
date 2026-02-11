# M.O.L.O.C.H. PI GHOST 4.5

**M**aster **O**f **L**ocal **O**perations, **C**omputation & **H**ome-automation

Ein autonomer Hauskobold auf Raspberry Pi 5 mit Hailo-10H NPU - sieht, hoert, spricht, denkt.

## Hardware

| Komponente | Modell | Verbindung |
|-----------|--------|------------|
| Brain | Raspberry Pi 5 (8GB) | - |
| NPU | Hailo-10H (26 TOPS) | M.2 |
| SSD 1 | 465GB ext4 | NVMe (System + Code) |
| SSD 2 | 477GB NTFS | NVMe (/mnt/moloch-data/) |
| Kamera | Sonoff CAM-PT2 | ONVIF/RTSP (192.168.178.25) |
| Mikrofon | ReSpeaker Lite | USB (16kHz, Mono) |
| Speakers | USB Audio | ALSA |

## Architektur

```
moloch/
  core/
    gui/
      push_to_talk.py      # Haupt-GUI (Talk, Vision, PTZ)
      eye_control_panel.py  # Kamera-Steuerung (ONVIF + Cloud)
      camera_control_panel.py # Manuelles PTZ-Panel
    hardware/
      camera.py             # Kamera-Controller (ONVIF PTZ, Tracking)
      camera_cloud_bridge.py # eWeLink Cloud API (LED, IR, Sleep)
      ptz_calibration.py    # PTZ Kalibrierung
      hailo_manager.py      # NPU Resource Manager (Voice/Vision)
    memory/
      persistent_memory.py  # JSON-Langzeitgedaechtnis
      vector_memory.py      # Qdrant semantische Suche
    speech/
      hailo_whisper.py      # STT (NPU Whisper + CPU Fallback)
      audio_pipeline.py     # Audio-Analyse + Preprocessing
    tts/
      tts_manager.py        # TTS Interface (Preparation Mode)
      config/voices.json    # 8 deutsche Stimmen (Piper)
      selection/            # Kontextuelle Stimmenwahl
    tts.py                  # Aktiver Piper TTS Engine
    vision/
      gst_hailo_detector.py # GStreamer + Hailo Person Detection
      gst_hailo_pose_detector.py # Pose/Keypoint Detection
      gesture_detector.py   # Gesten-Erkennung (COCO Keypoints)
      identity_manager.py   # ArcFace Gesichtserkennung
      vision_mode_manager.py # Vision Modi (Tracking/Identity/Full)
    personality/
      personality_engine.py # Guardian/Shadow Dual-Persoenlichkeit
    mpo/
      ptz_orchestrator.py   # Vision -> PTZ Entscheidungen
      autonomous_tracker.py # Autonomes Kamera-Tracking
  context/
    vision_context.py       # Vision State Sharing (Thread-safe)
    perception_state.py     # Wahrnehmungs-State Machine
    system_autonomy.py      # System-Health + Recovery
  config/
    eye_capabilities.json   # Kamera-Referenz
    moloch_identity.json    # Persoenlichkeits-Config
    sonoff_camera.json      # Kamera-Credentials
  scripts/
    self_diagnosis.py       # 12 echte Funktionstests
    enroll_face_arcface.py  # ArcFace Enrollment
  tests/
    test_memory.py          # 37 Tests (JSON + Qdrant)
    test_voice.py           # 36 Tests (Config, Selection, Audio)
    test_vision.py          # 41 Tests (Daten, Gesten, Identity)
    test_integration.py     # 24 Tests (Cross-Module, Imports)
```

## Was funktioniert

- **Sehen**: Hailo NPU Person/Pose Detection, GStreamer Pipeline, 20+ FPS
- **Hoeren**: ReSpeaker Lite USB Mikrofon, Whisper STT (NPU + CPU Fallback)
- **Sprechen**: Piper TTS mit 8 deutschen Stimmen, Emergentis Drift-Layer
- **Erinnern**: JSON-Langzeitgedaechtnis + Qdrant semantische Vektorsuche
- **Denken**: Claude API fuer Konversation, Memory-Injection, Selbstdiagnose
- **Bewegen**: ONVIF PTZ (342 Grad Pan, 157 Grad Tilt), autonomes Tracking
- **Erkennen**: ArcFace Gesichtserkennung, Gesten (Wave, Hands Up, Pointing)
- **Steuern**: eWeLink Cloud Bridge (LED, IR, Alarm, Sleep)

## Starten

```bash
# Haupt-Talk-Interface (alles inklusive)
cd ~/moloch && python3 -m core.gui.push_to_talk

# Kamera-Panel (ONVIF + Cloud)
cd ~/moloch && python3 -m core.gui.eye_control_panel

# Selbstdiagnose
cd ~/moloch && python3 scripts/self_diagnosis.py quick

# Tests
cd ~/moloch && python3 -m pytest tests/ -v
```

## Netzwerk

- Pi: `192.168.178.24` (SSH: `molochzuhause@moloch`)
- Kamera: `192.168.178.25` (KEIN Internet - Fritz!Box gesperrt)
- Qdrant: `localhost:6333` (Docker, Storage auf SSD 2)

## Storage

- **SSD 1** (ext4): System, Code (`~/moloch/`), Piper Voices (`~/moloch/models/voices/`)
- **SSD 2** (NTFS): Hailo HEF Modelle (`/mnt/moloch-data/hailo/models/`), Qdrant DB, Treiber

## Tests

```
142 passed, 1 skipped (full suite, ~35s)

test_memory.py      37 tests  - JSON + Qdrant Memory
test_voice.py       36 tests  - Voice Config, Selection, Audio Pipeline
test_vision.py      41 tests  - Enums, Gesten, Identity, State Machines
test_integration.py 29 tests  - Cross-Module, Graceful Degradation, Imports
```

## Kamera-Hinweise

- **Pan-Achse ist INVERTIERT**: x=+0.5 = LINKS, x=-0.5 = RECHTS
- Pan Range: -168.4 bis 174.4 Grad (342.8 Grad total)
- Tilt Range: -78.8 bis 101.3 Grad
- AbsoluteMove mit Grad-Werten (nicht normalisiert)
- WSDL: `~/.local/lib/python3.13/site-packages/wsdl/`
