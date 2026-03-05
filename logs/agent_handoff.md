# AGENT HANDOFF — Gate 0.5
# Geschrieben: 2026-03-05 20:16 UTC
# Naechste Instanz: Lies dies NACH CLAUDE.md und GATE_05_KONTEXT.md

## AKTUELLER STAND

Gate 0.5 | Phase 5 (Stabilitaet) | LAEUFT — 6h Monitor aktiv

## WAS ERLEDIGT WURDE

### Phase 2 (vorige Session): ALLE 4 TEST-SCRIPTS PASS
```
Test 1 (RTSP basic):     20.7 FPS | 1080p       | PASS
Test 2 (YOLO single):    20.0 FPS | 162 MB RAM  | 512 Persons   | PASS
Test 3 (SCRFD single):   20.0 FPS | 160 MB RAM  | 563 Faces     | PASS
Test 4 (Multi 3-Modell): 20.0 FPS | 199 MB RAM  | 567 Embeddings | PASS
```

### Phase 3 (diese Session): SERVICE-INTEGRATION KOMPLETT
- Poll-Thread `_tappas_perception_loop()` deployed (5 Hz)
- PFrame -> PerceptionEngine/CoreIntegrator/LED/DailyLearner
- `_write_status_json` TAPPAS-kompatibel (getattr + PFrame-Daten im IPC)
- Feature-Flag `MOLOCH_USE_TAPPAS=1` in ~/.profile aktiviert
- Threshold-Propagation: Panel-Slider -> Detection-Filter in _on_buffer
- ArcFace nutzt `arcface_thresh_val` statt hartcodiert 0.5

### Phase 4 (diese Session): NPU-STUFENLOGIK KOMPLETT (Option C)
- PerceptionEngine.tick(context) statt nicht-existierendes update(pframe)
- Context-Dict aus PFrame-Attributen gebaut
- Stage-Machine: idle->person->face korrekt getriggert
- Alle TAPPAS-Modelle permanent aktiv, Stages nur fuer Tracking/Logging
- face_streak, scores, decision_count incrementieren korrekt

### Phase 5 (AKTIV): STABILITAETSTEST LAEUFT
- Monitor PID: laeuft als nohup Hintergrund-Prozess
- Logfile: `logs/stability_phase5.log` (alle 5 Minuten)
- Baseline: 855 MB RSS, 62.25°C CPU, 20.6 FPS, 2.4 GB RAM belegt

## LIVE-SYSTEM STATUS (20:16 UTC)

```
Service:        ACTIVE (TAPPAS)
FPS:            20.6 (steady)
Mode:           tappas
NPU Stage:      face
Person:         detected
Face:           detected (conf ~0.62)
Face ID:        None (zu weit weg oder nicht Markus)
Tracker:        tracking (aktiv)
CoreIntegrator: zone=guardian, tension=0.04
LED:            markus_off
RAM:            2.4 GB / 3.9 GB
CPU Temp:       62.25°C
Active Models:  scrfd, arcface, yolov8m
```

## WAS ALS NAECHSTES KOMMT

### Phase 5 auswerten (nach 6h)
1. `cat ~/moloch/logs/stability_phase5.log` — Monitor-Daten pruefen
2. Kriterien: FPS >= 10, RAM < 3.5 GB, Temp < 70°C, 0 Crashes
3. Wenn PASS: 24h laufen lassen
4. Dann: `python3 ~/moloch/moloch_audit.py --auto`

### Offene Items (nicht blockierend)
- Face-ID Match testen (Markus direkt vor Kamera)
- Whisper + TAPPAS Parallel-Test (Voice-Command waehrend Detection)
- Panel-GUI: TAPPAS-Status korrekt anzeigen pruefen
- Pose-Modell Integration (yolov8s_pose_h10.hef) — spaeter

## GEAENDERTE DATEIEN (seit letztem Handoff)

- `core/moloch_service.py` — Poll-Thread, Status-JSON Fix, PerceptionEngine tick()
- `core/perception/tappas_pipeline.py` — Threshold-Filterung, ArcFace Threshold
- `~/.profile` — MOLOCH_USE_TAPPAS=1

## GIT COMMITS (diese Session)

- `043a67d` Phase 3 KOMPLETT: Poll-Thread + Status-JSON + Threshold-Propagation
- `06f70df` Phase 4: PerceptionEngine Stage-Tracking via tick() + Option C

## WICHTIG FUER NAECHSTE INSTANZ

1. Service laeuft MIT TAPPAS — NICHT resetten ohne Grund
2. Stability Monitor laeuft im Hintergrund — Log checken
3. Pan-Vorzeichen in camera.py NICHT ANFASSEN
4. Filter-Thresholds NICHT ANFASSEN
5. Definition of Done: siehe GATE_05_ARBEITSPAKET.md (unten)
