# GATE 0.5 ARBEITSPAKET — 7er-Agententeam
# Erstellt: 2026-03-05
# Status: BEREIT ZUR AUSFUEHRUNG

## IST-ZUSTAND (Bestandsaufnahme)

### Hardware & Software — ALLES DA
```
HailoRT:           5.1.1 (VDevice + HEF Import OK)
TAPPAS Core:       5.1.0 (hailo-tappas-core installiert)
GStreamer:         1.26.2 (Python Bindings OK)
Hailo GStreamer:   20 Elemente verfuegbar (hailonet, hailocropper, hailofilter,
                   hailooverlay, hailotracker, hailogallery, etc.)
Moloch Service:    AKTIV (systemd, headless)
Feature-Flag:      MOLOCH_USE_TAPPAS=0 (NICHT gesetzt in .profile)
```

### HEF-Modelle auf SSD2 — ALLE VORHANDEN
```
/mnt/moloch-data/hailo/models/
  yolov8m_h10.hef              21 MB   Person Detection (640x640)
  scrfd_10g.hef                5.8 MB  Face Detection (640x640)
  arcface_mobilefacenet.hef    2.6 MB  Face Recognition (112x112)
  yolov8s_pose_h10.hef        13.6 MB  Pose/Keypoints (640x640) — NICHT in Pipeline
  yolov8m_pose_h10.hef        29.4 MB  Pose/Keypoints (640x640) — NICHT in Pipeline
  hand_landmark_lite.hef       1.3 MB  Hand Landmarks — NICHT kompatibel?
  face_attr_resnet_v1_18.hef   6.9 MB  Face Attributes — ENTFERNT aus Pipeline
  yolov11m_h10.hef            28 MB    YOLOv11 — NICHT in Pipeline
  resnet_v1_50_h10.hef        23.5 MB  ResNet50 — NICHT in Pipeline
  yolov5n_seg_h10.hef          3.5 MB  Segmentation — NICHT in Pipeline
```

### Postprocess Shared Libraries — ALLE VORHANDEN
```
/usr/local/hailo/resources/so/
  libyolo_hailortpp_postprocess.so    YOLO NMS + Letterbox
  libscrfd.so                         SCRFD Face Decode
  libface_recognition_post.so         ArcFace Embedding Normalize
  libvms_face_align.so                Face Alignment (5-Point)
  libvms_croppers.so                  Face Cropper
  libyolov8pose_postprocess.so        Pose Keypoints

/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/
  libwhole_buffer.so                  Whole Buffer Cropper (Letterbox)
```

### Test-Scripts — ALLE 4 DA (Phase 2 komplett geschrieben)
```
scripts/test_gstreamer_basic.py    RTSP-Basis (kein NPU)
scripts/test_gstreamer_yolo.py     YOLOv8m + Letterbox
scripts/test_gstreamer_scrfd.py    SCRFD + Letterbox + Snapshot
scripts/test_gstreamer_multi.py    YOLO+SCRFD+ArcFace+Tracker (Full Pipeline)
```

### tappas_pipeline.py — 770 Zeilen, GRUNDSTRUKTUR DA
```
Implementiert:
  [x] GStreamer Pipeline: rtspsrc -> YOLO -> SCRFD -> Tracker -> ArcFace -> appsink
  [x] Letterbox Preprocessing (use-letterbox=true bei allen Wrappern)
  [x] Model Scheduler (vdevice-group-id=SHARED)
  [x] Pad-Probe Callback fuer Detection-Extraktion
  [x] ArcFace Face-Matching (Cosine Similarity gegen Face-DB)
  [x] PerceptionFrame Generierung (_build_pframe)
  [x] SHM Frame Output (/dev/shm/moloch_frame)
  [x] Thread-safe Detection Queue
  [x] FPS Tracking
  [x] start()/stop()/is_running()
  [x] get_detections()/get_current_pframe()/get_annotated_frame()
  [x] reload_face_db() (mit und ohne Parameter)

  FEHLT (die roten Luecken):
  [ ] PerceptionFrame -> PerceptionEngine Uebergabe
  [ ] CoreIntegrator Update (Tension/Dominance)
  [ ] DailyLearner Integration (Snapshot-Triggers)
  [ ] LED Steuerung (Markus erkannt -> LED Feedback)
  [ ] Status-JSON Schreiben via IPC
  [ ] Threshold-Propagation zu hailonet Elementen (Panel-Slider -> Pipeline)
  [ ] Pose-Modell Integration (yolov8s_pose_h10.hef)
```

### moloch_service.py — TAPPAS-Integration TEILWEISE
```
Implementiert:
  [x] USE_TAPPAS Feature-Flag (os.environ)
  [x] Conditional Import (TappasPipeline vs InferenceEngine)
  [x] RTSP-Conflict-Praevention (CameraManager RTSP skip bei TAPPAS)
  [x] Watchdog-Praevention (skip bei TAPPAS)
  [x] _tappas_tracker_feed_loop() fuer AutonomousTracker (15 Hz)

  FEHLT:
  [ ] Poll-Thread fuer PFrame -> PerceptionEngine/CoreIntegrator/LED/DailyLearner
  [ ] MOLOCH_USE_TAPPAS in .profile setzen (zum Aktivieren)
```

---

## ARBEITSPAKETE NACH PHASEN

### PHASE 2: TEST-SCRIPTS VALIDIEREN (Pflicht vor Integration!)

**Verantwortlich:** Agent 1 (Vision-Pipeline) + Agent 2 (Hardware)
**DeepSeek-Check:** "Laufen die Tests WIRKLICH durch, oder nur im Kopf?"

WICHTIG: Moloch-Service STOPPEN vor jedem Test!
```bash
sudo systemctl stop moloch
```

#### Schritt 2.1: RTSP Basis-Test
```bash
python3 ~/moloch/scripts/test_gstreamer_basic.py
```
Erwartet: 10 Frames, FPS-Messung, 1920x1080 oder 1280x720

#### Schritt 2.2: YOLO Person Detection
```bash
python3 ~/moloch/scripts/test_gstreamer_yolo.py
```
Erwartet: >=15 FPS, Person-Detections, RAM <3.5GB, 30s Laufzeit

#### Schritt 2.3: SCRFD Face Detection
```bash
python3 ~/moloch/scripts/test_gstreamer_scrfd.py
```
Erwartet: >=15 FPS, Face-Detections, Snapshot in logs/, BBox-Koordinaten

#### Schritt 2.4: Multi-Modell Pipeline (Der Haerteste!)
```bash
python3 ~/moloch/scripts/test_gstreamer_multi.py
```
Erwartet: >=10 FPS (3 Modelle!), Persons + Faces + Embeddings, Snapshot

#### Erfolgskriterium Phase 2:
ALLE 4 Tests PASS. Wenn einer failt → STOPP, fixen, dann weiter.
FPS und RAM-Werte dokumentieren!

---

### PHASE 3: SERVICE-INTEGRATION (Der Kern von Gate 0.5)

**Verantwortlich:** Agent 6 (Service/Integration) + Agent 1 (Vision-Pipeline)
**DeepSeek-Check:** "Fliesst das Qi von der Pipeline zum Hirn?"

#### Schritt 3.1: Poll-Thread in moloch_service.py

Der fehlende Brueckenpfeiler: Ein Thread der regelmaessig PFrame von
TappasPipeline holt und an den Rest des Systems weitergibt.

```
TappasPipeline._on_buffer()
  -> pipeline._current_pframe (intern, thread-safe)

Neuer Thread: _tappas_perception_loop() (~5 Hz)
  -> pipeline.get_current_pframe()
  -> PerceptionEngine.update(pframe)     ← FEHLT
  -> CoreIntegrator.tick_with_pframe()   ← FEHLT
  -> DailyLearner.check_snapshot()       ← FEHLT
  -> LEDController.update_hysteresis()   ← FEHLT
  -> IPC.write_status()                  ← FEHLT
```

REGEL: Dieser Thread gehoert in moloch_service.py, NICHT in tappas_pipeline.py!
(Separation of Concerns — Pipeline liefert Daten, Service verarbeitet sie)

#### Schritt 3.2: Feature-Flag aktivierbar machen

```bash
# In ~/.profile hinzufuegen:
export MOLOCH_USE_TAPPAS=1

# ODER in systemd service:
Environment=MOLOCH_USE_TAPPAS=1
```

Fallback: Wenn MOLOCH_USE_TAPPAS nicht gesetzt → alter Code (InferenceEngine)

#### Schritt 3.3: Threshold-Propagation (Panel → Pipeline)

Problem: Panel-Slider setzen z.B. scrfd_conf_val, aber der Wert
erreicht nie die hailonet-Elemente in der GStreamer Pipeline.

Loesung: Property-Setter die GStreamer-Element Properties aktualisieren:
```python
@scrfd_conf_val.setter
def scrfd_conf_val(self, val):
    self._scrfd_conf = val
    # Propagate to GStreamer element
    scrfd_net = self._pipeline.get_by_name("scrfd_hailonet")
    if scrfd_net:
        scrfd_net.set_property("nms-score-threshold", val)
```

ACHTUNG: Ob hailonet diese Properties zur Laufzeit aendert → TESTEN!

#### Erfolgskriterium Phase 3:
```bash
export MOLOCH_USE_TAPPAS=1
sudo systemctl restart moloch
sleep 10
journalctl -u moloch --since "15 sec ago" --no-pager | tail -20
# Muss zeigen: "[INIT] TAPPAS Pipeline" + Detections + kein ERROR
```

---

### PHASE 4: NPU-STUFENLOGIK

**Verantwortlich:** Agent 1 (Vision-Pipeline)
**DeepSeek-Check:** "Wu-Wei — aktiviere nur was gebraucht wird"

TAPPAS-Spezifik: Alle 3 Modelle (YOLO+SCRFD+ArcFace) sind PERMANENT geladen.
Der Hailo Model Scheduler entscheidet selbst wer NPU-Zeit bekommt.

Frage: Brauchen wir die IDLE→PERSON→FACE Stufenlogik noch?

Option A: NEIN — Alle Modelle immer aktiv (TAPPAS-Philosophie)
  + Einfacher, weniger Code
  + Model Scheduler optimiert automatisch
  - NPU laeuft immer auf Volllast (Thermal?)

Option B: JA — Aber anders implementiert
  IDLE:   Nur YOLO aktiv (hailonet Element auf Playing/Paused schalten?)
  PERSON: YOLO + SCRFD aktiv
  FACE:   YOLO + SCRFD + ArcFace aktiv

  Frage: Kann man hailonet-Elemente dynamisch pausieren in TAPPAS?
  → MUSS GETESTET WERDEN

Option C: HYBRID — TAPPAS Pipeline laeuft immer mit allen Modellen,
  aber PerceptionEngine ignoriert Ergebnisse je nach Stufe.
  + Kein GStreamer-Umbau noetig
  + Model Scheduler regelt Thermal automatisch
  - Verschwendet NPU-Zyklen

EMPFEHLUNG: Option C als Startpunkt, Option B als Optimierung spaeter.
DeepSeek wuerde sagen: "Erstmal fliessen lassen, dann optimieren."

---

### PHASE 5: STABILITAET

**Verantwortlich:** Agent 2 (Hardware) + Agent 6 (Service/Integration)
**DeepSeek-Check:** "24 Stunden. Kein Crash. Kein Memory Leak. Kein Thermal Throttle."

#### Kriterien:
- FPS >= 25 konstant (bei 3 Modellen >= 10 akzeptabel)
- RAM < 3.5 GB (Pi5 hat 4 GB!)
- CPU Temp < 70 Grad, NPU Temp < 70 Grad
- 0 Crashes in 6h, dann 24h
- Face Recognition funktioniert (Markus erkennen)
- PTZ Tracking funktioniert
- LED reagiert korrekt
- Voice Pipeline (Whisper) funktioniert parallel

#### Test-Ablauf:
```bash
# 1. TAPPAS aktivieren
export MOLOCH_USE_TAPPAS=1
sudo systemctl restart moloch

# 2. 6h Stabilitaetstest
# Monitoring via:
journalctl -u moloch -f     # Live Logs
cat /sys/class/thermal/thermal_zone0/temp   # CPU Temp
cat /dev/shm/moloch_status.json | python3 -m json.tool  # Status

# 3. Nach 6h:
python3 ~/moloch/moloch_audit.py --auto
# ALLE Tests muessen PASS sein!

# 4. Wenn 6h PASS → 24h laufen lassen
```

---

## AGENTEN-ZUORDNUNG

```
PHASE 2 (Tests):
  Agent 1 (Vision):     Fuehrt Tests durch, fixt Pipeline-Bugs
  Agent 2 (Hardware):   Ueberwacht NPU/Thermal, RTSP-Stabilitaet
  Agent 7 (DeepSeek):   "Zeigt mir die Zahlen, nicht Hoffnung"

PHASE 3 (Integration):
  Agent 6 (Service):    Baut Poll-Thread, Feature-Flag, IPC
  Agent 1 (Vision):     Threshold-Propagation, PFrame-Format
  Agent 4 (Tracking):   _tappas_tracker_feed_loop() validieren
  Agent 5 (Voice):      Whisper shared VDevice Kompatibilitaet
  Agent 7 (DeepSeek):   "Wer spricht mit wem? Zeigt mir den Datenfluss!"

PHASE 4 (Stufenlogik):
  Agent 1 (Vision):     Option C implementieren
  Agent 6 (Service):    PerceptionEngine anpassen
  Agent 7 (DeepSeek):   "Wu-Wei! Braucht ihr die Stufen wirklich?"

PHASE 5 (Stabilitaet):
  Agent 2 (Hardware):   Thermal Monitoring, RAM-Tracking
  Agent 6 (Service):    Crash Recovery, Log-Analyse
  Agent 3 (GUI):        Panel zeigt TAPPAS-Status korrekt
  Agent 7 (DeepSeek):   "24h ohne Crash. Dann reden wir weiter."
```

---

## REIHENFOLGE DER DATEIEN-AENDERUNGEN

REGEL: Jede Aenderung = ein Git Commit VORHER!
REGEL: Erst ALLE Tests laufen lassen (Phase 2), DANN Code aendern!

```
Phase 2: KEINE Code-Aenderungen. Nur Tests ausfuehren.

Phase 3 Dateien (in Reihenfolge):
  1. core/moloch_service.py          ← Poll-Thread hinzufuegen
  2. core/perception/tappas_pipeline.py  ← Threshold-Setter
  3. ~/.profile ODER systemd unit    ← MOLOCH_USE_TAPPAS=1

Phase 4 Dateien:
  4. core/perception_engine.py       ← Anpassung fuer TAPPAS PFrames

Phase 5: KEINE Code-Aenderungen. Nur Monitoring.
```

---

## BEKANNTE RISIKEN

1. **Pi5 hat nur 4 GB RAM** — 3 Modelle + GStreamer + Python + Panel = eng!
   Monitoring: `free -h` alle 5 Minuten.

2. **hailonet Threshold zur Laufzeit aendern** — Unklar ob das geht.
   Plan B: Thresholds nur beim Pipeline-Neustart setzen.

3. **RTSP-Stream Abbruch** — Sonoff Kamera kann Stream droppen.
   GStreamer rtspsrc hat reconnect-Logik, aber muss getestet werden.

4. **Whisper + TAPPAS NPU-Konflikt** — Theoretisch shared VDevice,
   aber TAPPAS hat eigenes vdevice-group-id=SHARED. Kompatibel?
   → KRITISCH, muss in Phase 3 getestet werden!

5. **CameraManager PTZ** — CameraManager wird noch fuer PTZ gebraucht,
   auch wenn RTSP ueber GStreamer laeuft. Sicherstellen dass
   CameraManager.start_rtsp() wirklich NICHT aufgerufen wird.

---

## DEEPSEEK-CHECKPOINTS (Pflicht!)

Nach JEDER Phase fragt DeepSeek:
  Phase 2: "Laufen die Tests? Zeig mir FPS und RAM. Keine Ausreden."
  Phase 3: "Fliesst das Qi? Sehe ich Detections UND LED-Reaktion UND Voice?"
  Phase 4: "Braucht ihr die Stufen? Oder macht der Model Scheduler das besser?"
  Phase 5: "24h. Null Crashes. Null Memory Leaks. Dann gratuliere ich euch."

---

## DEFINITION OF DONE — Gate 0.5

[ ] TAPPAS Pipeline laeuft mit 3 Modellen (YOLO+SCRFD+ArcFace)
[ ] Letterbox Preprocessing aktiv (keine verschobenen BBoxen mehr)
[ ] Model Scheduler managed NPU (kein manuelles Load/Unload)
[ ] FPS >= 10 bei 3 Modellen (Ziel: 25 bei optimiertem Scheduling)
[ ] Face Recognition erkennt Markus zuverlaessig
[ ] PTZ Tracking funktioniert mit TAPPAS-Detections
[ ] LED reagiert auf Markus (Guardian-Modus)
[ ] Voice Pipeline (Whisper) laeuft parallel ohne Konflikt
[ ] Panel zeigt TAPPAS-Status korrekt
[ ] Fallback auf alte Pipeline via Feature-Flag moeglich
[ ] 6h Stabilitaetstest PASS
[ ] RAM < 3.5 GB, CPU/NPU Temp < 70 Grad
[ ] moloch_audit.py --auto = ALL PASS
