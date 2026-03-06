# AGENT HANDOFF — Gate 0.5
# Geschrieben: 2026-03-05 20:37 UTC
# Naechste Instanz: Lies dies NACH CLAUDE.md und GATE_05_KONTEXT.md

## AKTUELLER STAND

Gate 0.5 | Phase 5 (Stabilitaet) + Face-ID BLOCKER | Service LAEUFT

## KRITISCHER BLOCKER: Face-ID Match = sim=0.000

ArcFace-Embeddings aus der GStreamer TAPPAS-Pipeline (hailofilter + libface_recognition_post.so)
sind INKOMPATIBEL mit Embeddings aus direkter HailoRT-Inference (train_faces_batch.py).
Cosine-Similarity ist EXAKT 0.000 — das bedeutet die Embeddings sind orthogonal/anders formatiert.

### Ursache (Hypothesen):
1. GStreamer arcface_hailofilter Postprocess normalisiert anders als direkte NPU-Inference
2. Face-Align im GStreamer-Pfad (libvms_face_align.so) veraendert das Crop anders
3. Preprocessing-Unterschied: GStreamer Pipeline skaliert auf 112x112 intern vs. Script macht cv2.resize

### Loesung (naechste Instanz muss dies fixen):
Option A: Face-Training INNERHALB der TAPPAS-Pipeline machen:
  - Fotos durch die GStreamer-Pipeline schicken (als file-source statt rtspsrc)
  - Embeddings aus dem Pad-Probe extrahieren
  - So sind Training und Inference identisch

Option B: Embedding-Format aus GStreamer-Pipeline reverse-engineeren:
  - Live ein Embedding aus _on_buffer extrahieren und loggen (Form, Range, Norm)
  - Mit dem Training-Embedding vergleichen
  - Transformation finden (z.B. L2 vs Cosine, andere Dimensionsreihenfolge)

Option C: Die _match_face Methode anpassen:
  - Statt face_embeddings.json aus dem Batch-Script zu nutzen,
    die Live-Pipeline ein paar Sekunden laufen lassen und Embeddings sammeln
  - "Einpraegen per Live-Kamera" statt aus Fotos

### EMPFEHLUNG: Option A oder C — Training muss den gleichen Pfad wie Inference nutzen!

### Debug-Info die schon da ist:
- Temporaeres INFO-Log in _on_buffer: alle 50 Frames loggt es sim, emb_dim, db_size
- Face-DB hat 1 Person (markus) aus 308 Embeddings (Durchschnitt)
- db_size=1 ist korrekt, emb_dim=512 ist korrekt
- sim=0.000 → Embeddings sind KOMPLETT inkompatibel

## WAS DIESE SESSION ERLEDIGT HAT

### Phase 3 (Service-Integration) — DONE
- Poll-Thread `_tappas_perception_loop()` deployed (5 Hz)
- `_write_status_json` TAPPAS-kompatibel (getattr + PFrame-Daten)
- Feature-Flag `MOLOCH_USE_TAPPAS=1` in ~/.profile
- Threshold-Propagation: Panel-Slider → `_on_buffer` Detection-Filter
- ArcFace nutzt `arcface_thresh_val` statt hartcodiert

### Phase 4 (NPU-Stufenlogik) — DONE (Option C)
- PerceptionEngine.tick(context) statt nicht-existierendes update()
- Stage-Machine: idle→person→face korrekt getriggert
- Alle TAPPAS-Modelle permanent aktiv, Stages nur fuer Tracking/Logging

### Phase 5 (Stabilitaet) — LAEUFT
- Monitor in `logs/stability_phase5.log` (alle 5 Min)
- Baseline: 855 MB RSS, 62°C CPU, 20.0 FPS

### Face Training — DONE aber INKOMPATIBEL
- Script: `scripts/train_faces_batch.py` (stoppt Service, NPU direkt, startet Service)
- 369 Bilder → 287 Markus-Embeddings gespeichert
- Face-DB: 21 → 308 Embeddings
- TAPPAS lädt DB korrekt: 1 Person aus 308 Embeddings (Durchschnitt)
- ABER: sim=0.000 → Embeddings aus unterschiedlichen Pfaden inkompatibel!

### Face-DB Lade-Logik verbessert
- Jetzt Durchschnitt pro Person (Name vor '#') statt alle einzeln
- face_embeddings.json: {key: embedding} → gruppiert nach Person → ein Mean-Embedding

## GEAENDERTE DATEIEN

- `core/moloch_service.py` — Poll-Thread, Status-JSON Fix, PerceptionEngine tick()
- `core/perception/tappas_pipeline.py` — Threshold-Filter, Face-DB Gruppierung, Debug-Log
- `scripts/train_faces_batch.py` — NEU: Batch Face Training (HailoRT direkt)
- `~/.profile` — MOLOCH_USE_TAPPAS=1

## GIT COMMITS (diese Session)

- `c4dca13` BACKUP vor Phase 3 TAPPAS Service-Integration (vorige Session)
- `043a67d` Phase 3 KOMPLETT: Poll-Thread + Status-JSON + Threshold-Propagation
- `06f70df` Phase 4: PerceptionEngine Stage-Tracking via tick() + Option C
- `d094497` Handoff aktualisiert

## SERVICE-STATUS

- Moloch Service: ACTIVE (TAPPAS, 20 FPS)
- MOLOCH_USE_TAPPAS=1: AKTIV
- Face-Detection: FUNKTIONIERT (conf ~0.77)
- Face-ID Match: KAPUTT (sim=0.000 — Embedding-Inkompatibilitaet)
- Tracker: FUNKTIONIERT (tracking/searching)
- Person Detection: FUNKTIONIERT

## TEMPORAERE DEBUG-LOGS (ENTFERNEN WENN GEFIXT)

- `tappas_pipeline.py` Zeile ~551: alle 50 Frames ein FACE-MATCH INFO-Log
- Zeile ~691-694: debug-log fuer Match/Kein-Match

## WICHTIG FUER NAECHSTE INSTANZ

1. BLOCKER: Face-ID sim=0.000 → Embeddings Training vs Live inkompatibel
2. Service laeuft MIT TAPPAS — 20 FPS, alles ausser Face-ID funktioniert
3. Face-DB hat 308 Embeddings aber sie sind NUTZLOS fuer TAPPAS-Matching
4. Pan-Vorzeichen in camera.py NICHT ANFASSEN
5. Stability Monitor laeuft noch im Hintergrund
6. Lies GATE_05_ARBEITSPAKET.md fuer Definition of Done
