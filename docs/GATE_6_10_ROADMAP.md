# M.O.L.O.C.H. Gate 6-10 Roadmap
# Erstellt: 2026-03-28 | Nur Software, keine neue Hardware
# Hardware: Pi5 4GB + Hailo-10H 8GB + Sonoff PTZ + ReSpeaker ESP32
#           + PiSugar UPS + Noctua Fan + eWeLink LED + HDMI Audio

## Gate 6 — Temporale Intelligenz
Routine-Erkennung, Verhaltens-Vorhersage, Anomalie-Erkennung
Basis: PerceptionMemory (temporal_memory.py), EntityTracker, AttentionMap

## Gate 7 — Lokales LLM (Traum-Modus)
hailo-ollama + Qwen2.5-1.5B, Nacht-Reflexion, Offline-Reasoning
Basis: NPU_DREI_SCHICHTEN_ARCHITEKTUR.md, night_cycle.py

## Gate 8 — Raeumliche Intelligenz
3D-Raumverstaendnis, Objekt-Persistenz, Segmentierung
Basis: room_map.py, yolov5n_seg.hef, PTZ-Positionen

## Gate 9 — Tentakel-Netzwerk
WLED, MQTT, Home Assistant, Multi-Sensor
Basis: AGENT_TENTACLE.md, ESP32 WiFi-Mic

## Gate 10 — Emergente Autonomie
Selbst-Modifikation, Praeferenz-Lernen, Persoenlichkeits-Drift
Basis: behavior_rules.py, decision_engine.py, CoreIntegrator

## Hardware-Inventar (alles vorhanden)
- Pi5 4GB RAM, 2x NVMe SSD (ext4 + NTFS)
- Hailo-10H 8GB NPU (98% frei, 180MB/8192MB genutzt)
- Sonoff CAM-PT2 (RTSP 1080p, ONVIF PTZ)
- ReSpeaker Lite ESP32-S3 WiFi (16/48kHz, UDP, RGB-LED)
- PiSugar UPS (Batterie, Lade-Status, Power-Monitoring)
- Noctua Fan (GPIO PWM, Temperatur-gesteuert)
- eWeLink Cloud (LED, IR, Alarm)
- HDMI Audio (Piper TTS + Spotify via PipeWire)
