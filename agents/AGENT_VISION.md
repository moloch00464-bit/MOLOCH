# AGENT_VISION.md — Vision-Pipeline & NPU
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der VISION-AGENT. Alles was mit Bilderkennung, NPU-Inferenz, GStreamer und TAPPAS zu tun hat ist DEIN Revier.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/perception/tappas_pipeline.py    829 LOC  — TAPPAS GStreamer Pipeline, _on_buffer, Face-Match
core/perception/hailo_postprocess.py           — HEF Output Parsing, BBox Decode
core/perception/perception_buffer.py           — Frame-Buffer zwischen Pipeline und Service
core/perception/perception_frame.py            — PFrame Datenstruktur
core/perception/perception_manager.py          — Perception State Machine
core/perception/model_health.py                — NPU Model Health Checks
core/perception/spatial_learning.py            — Raeumliches Lernen
core/perception_engine.py                      — Modell-Scoring, Stages, force_models()
core/inference_engine.py                       — Legacy Inference (wenn USE_TAPPAS=0)
core/model_orchestrator.py                     — Legacy Model Loading
core/hardware/hailo_manager.py                 — HailoRT VDevice, HEF Loading
scripts/train_faces_batch.py                   — Face-Training Script
```

## Dein Wissen
- TAPPAS 5.1.0 mit GStreamer 1.26.2
- Feature-Flag: MOLOCH_USE_TAPPAS=1 in moloch.service
- HEF-Modelle: /mnt/moloch-data/hailo/models/ (NIEMALS auf SSD1!)
- NPU: Hailo-10H, 8GB LPDDR4, HailoRT 5.1.1
- Ein VDevice fuer alles (shared), NIEMALS zweites erstellen → Error 74
- GStreamer ArcFace-Embeddings sind INKOMPATIBEL mit HailoRT-direkt Embeddings (BLOCKER!)
- Letterbox-Preprocessing macht TAPPAS automatisch — KEIN cv2.resize
- H8L HEFs NICHT kompatibel mit H10 (Error 93)

## Bekannte Bugs in deinem Bereich
- Face-ID sim=0.200 statt >0.60 → Embedding-Inkompatibilitaet Training vs Live
- ArcFace Threshold zu niedrig (0.45) → erkennt alles als Markus

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. Nur DEINE Dateien anfassen
4. Nach Aenderung: Service restart + verify
5. Filter-Thresholds und Pan-Vorzeichen NICHT anfassen (bereits gefixt!)
6. HailoRT API: configured.run([bindings], timeout=10000) — Bindings als LISTE

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
