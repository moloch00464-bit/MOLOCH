# GATE 0.5 KONTEXT — Lies das NACH CLAUDE.md

## Was ist Gate 0.5
Umstieg von manuellem Python-Model-Orchestrator auf Hailo-native TAPPAS/GStreamer Pipeline mit Model Scheduler.

## Warum
- inference_engine.py macht naives cv2.resize(1920x1080 → 640x640) — BBox verschoben
- Modelle werden seriell geladen/entladen — langsam, NPU-RAM (8GB) unterausgelastet
- TAPPAS hat korrektes Letterbox-Preprocessing und Model Scheduler out-of-the-box

## Hardware
- Pi5 4GB RAM (NICHT 8GB!)
- Hailo-10H: 40 TOPS, 8GB LPDDR4, PCIe Gen3 x4, Firmware 5.1.1
- Kamera: Sonoff PT2, RTSP 1920x1080 (ch0), ONVIF PTZ
- WICHTIG: Sonoff Pan ist INVERTIERT — positiver Pan = physisch links

## Bereits gelöste Bugs (NICHT nochmal anfassen!)
- Pan-Vorzeichen: camera.py Zeile 721, pan_delta = -error_x (Minus ist korrekt!)
- Filter-Thresholds: Confidence 0.30, Height 0.10, Face-Area 0.05% (waren zu hoch)
- Claude Code hat core/mpo/autonomous_tracker.py gefixt — FALSCHE DATEI! Service importiert core/ptz_tracker.py, Tracking-Logik sitzt in core/hardware/camera.py

## Phasen

### Phase 1: Bestandsaufnahme (KEIN Code ändern)
- TAPPAS installiert? dpkg -l | grep hailo
- HEF-Inventar: find / -name "*.hef" 2>/dev/null
- TAPPAS Beispiele: find /opt/hailo -name "*face*" 2>/dev/null

### Phase 2: TAPPAS Testlauf (eigene Scripts in scripts/)
- test_gstreamer_basic.py — RTSP via GStreamer
- test_gstreamer_yolo.py — GStreamer + YOLO
- test_gstreamer_scrfd.py — GStreamer + SCRFD + Letterbox
- test_gstreamer_multi.py — Alle Modelle parallel
- IMMER: sudo systemctl stop moloch VORHER (NPU Konflikt!)

### Phase 3: Integration
- Neue Klasse: core/perception/tappas_pipeline.py
- Feature-Flag: MOLOCH_USE_TAPPAS=1 in Environment
- Fallback: ohne Flag → alter Code

### Phase 4: NPU-Stufenlogik
- IDLE → PERSON → FACE → INTERACTION
- Person-Detection triggert Tracking (fehlt aktuell!)
- Dynamisches Model-Switching basierend auf Zustand

### Phase 5: Stabilität
- 6h Stabilitätstest, dann 24h Dauerbetrieb
- FPS ≥25, RAM <3.5GB, Temp <70°C, 0 Crashes

## Dateien die ERSETZT werden (Gate 0.5)
- core/inference_engine.py → wird durch tappas_pipeline.py ersetzt
- core/model_orchestrator.py → wird durch Hailo Model Scheduler ersetzt
- core/perception/hailo_postprocess.py → TAPPAS macht Postprocessing

## Dateien die BLEIBEN
- core/hardware/camera.py (Pan-Fix drin, Tracking-Logik)
- core/core_integrator.py (Tension/Personality)
- core/moloch_service.py (empfängt Detections)
- core/gui/ (Panel)
- CLAUDE.md (Regeln)

## Vorgemerkt für Gate 1 (NICHT in Gate 0.5 implementieren!)
- Action Bridge FSM (Thought→Intent→Action→Result)
- silence_level Sensor (OpenCV Bewegungsanalyse → Aufwachgeschwindigkeit)
- Auto-Resume aus Manuell (Timeout → Moloch übernimmt + Spruch)
- Gain-Tuning (TRACKING_GAIN_PAN runter, MAX_STEP_PAN runter)
- Kamera-Park-Position = Tür (Home-Button)
- Kalibrierung mit labelme (manuelle Landmark-Korrektur)

## Regeln für Claude Code (zusätzlich zu CLAUDE.md)
1. Lies CLAUDE.md UND GATE_05_KONTEXT.md
2. Jede Phase hat eigene Scripts — NICHT den bestehenden Code ändern bis Phase 3
3. Moloch-Service STOPPEN vor NPU-Tests (sudo systemctl stop moloch)
4. Feature-Flag für neuen Code — Fallback immer möglich
5. Pan-Vorzeichen ist KORREKT (Minus in Zeile 721) — FINGER WEG
6. Filter-Thresholds sind KORREKT — FINGER WEG
7. HEFs müssen für H-10H kompatibel sein — prüfen vor Nutzung
