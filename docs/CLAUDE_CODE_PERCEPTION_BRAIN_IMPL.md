# CLAUDE_CODE_PERCEPTION_BRAIN_IMPL
## M.O.L.O.C.H. — Perception Brain Implementierungsplan
**Erstellt:** 2026-03-09 | **Autor:** Claude Sonnet (Architekt)  
**Für:** Claude Code (Implementierung)  
**Basis:** MOLOCH_PERCEPTION_BRAIN_SYSTEM + MOLOCH_AGENT_TOOLBOX v2.1

---

## WICHTIGE REGELN VOR DEM START

```
Lies CLAUDE.md und AGENT_TEAM_README.md.
Pi5 hat 4GB RAM. NPU hat 8GB — getrennte Welten.
NICHTS umbauen was läuft. Ergänzen, nicht ersetzen.
Git Backup vor jeder Änderung.
```

---

## MODUL 1: ONVIF Event Listener
**Datei:** `core/perception/onvif_event_listener.py`  
**Abhängigkeiten:** `pip3 install onvif-zeep --break-system-packages`  
**RAM:** ~15MB

### Was es tut
Abonniert ONVIF Motion/Human Events von der Sonoff CAM-PT2.
Gibt Wake-Signal auf Event Bus → IDLE → WAKE_UP Trigger.

### Input / Output
- Input: Sonoff CAM-PT2 ONVIF (IP aus settings.json: `camera_ip`)
- Output: Event Bus Events:
  - `camera.motion_detected` — Bewegung erkannt
  - `camera.human_detected` — Person erkannt (stärkerer Trigger)
  - `camera.ptz_moved` — PTZ hat sich bewegt

### Implementierung
```python
# Kamera-IP aus config/settings.json lesen
# ONVIF Subscribe auf:
#   tns1:RuleEngine/CellMotionDetector
#   tns1:RuleEngine/MyRuleDetector/People
# Bei Event → Event Bus publish (Priority 2)
# Reconnect-Loop alle 10s wenn Verbindung weg
# Timeout: 5s für ONVIF-Verbindungsaufbau
```

### Event Bus Integration
```python
await event_bus.publish(MolochEvent(
    topic="camera.human_detected",
    data={"confidence": ..., "source": "onvif"},
    priority=2
))
```

---

## MODUL 2: Tracking Quality Monitor
**Datei:** `core/perception/tracking_quality_monitor.py`  
**Abhängigkeiten:** numpy (bereits vorhanden)  
**RAM:** ~5MB

### Was es tut
Bewertet kontinuierlich wie gut die Kamera-interne Smart-Tracking-Funktion arbeitet.
Score 0.0–1.0. Unter 0.4 → MOLOCH übernimmt.

### Input / Output
- Input: PFrame aus Vision Pipeline (bbox, keypoints, PTZ-Position)
- Output: Event `tracking.quality_update` {score, reason, state}

### Score-Berechnung
```
bbox_stability    = 1.0 - (bbox_jitter / bbox_size)    # Gewicht 0.3
frame_centering   = 1.0 - (center_offset / frame_size) # Gewicht 0.3
jitter_penalty    = 1.0 wenn PTZ-Wechsel < 3x/2s       # Gewicht 0.2
camera_confidence = aus ONVIF oder geschätzt aus bbox   # Gewicht 0.2

score = weighted_sum(alle 4 Komponenten)
```

### Schwellwerte
- `< 0.4` → publish `tracking.takeover_required`
- `> 0.8` → publish `tracking.camera_sufficient`
- PTZ Richtungswechsel > 3x in 2s → `tracking.jitter_detected`

### Ringbuffer
```python
# PTZ-Position History: 20 Frames (2s bei 10Hz Sampling)
# BBox History: 30 Frames
```

---

## MODUL 3: PTZ Handover Controller
**Datei:** `core/perception/ptz_handover_controller.py`  
**Abhängigkeiten:** Bestehender ptz_arbiter.py  
**RAM:** ~8MB

### Was es tut
Steuert den sauberen Wechsel zwischen Kamera-Smart-Tracking und MOLOCH-PTZ.
Verhindert PTZ-Konflikte und Ruckler beim Handover.

### Handover camera → MOLOCH
```
1. Aktuellen PTZ-Pan/Tilt per ONVIF lesen
2. MOLOCH PTZ-Controller mit dieser Position initialisieren
3. Kamera Smart-Tracking deaktivieren (ONVIF Command)
4. NPU-Modelle hochfahren (signal an homeostasis.py)
5. Smooth transition: ersten 10 Frames PTZ-Bewegung dämpfen
```

### Handover MOLOCH → camera
```
1. MOLOCH PTZ-Kommandos stoppen (ptz_arbiter pause)
2. Kamera Smart-Tracking reaktivieren
3. 2s warten (Kamera stabilisiert sich)
4. Schwere NPU-Modelle entladen (signal an homeostasis.py)
```

### Event Bus Integration
```python
# Subscribed auf:
#   tracking.takeover_required  → handover_to_moloch()
#   tracking.camera_sufficient  → handover_to_camera()
#   tracking.jitter_detected    → emergency_takeover()
```

---

## MODUL 4: NPU Power Manager
**Datei:** `core/autonomy/npu_power_manager.py`  
**Abhängigkeiten:** homeostasis.py (bereits vorhanden)  
**RAM:** ~3MB

### Was es tut
Steuert welche NPU-Modelle je nach System-State geladen sind.
Schaltet NPU-Clock runter wenn Person stillsteht.

### State → Modelle Mapping
```
IDLE:                  NPU aus (alle Modelle entladen)
SEARCH_MODE:           NPU aus
WAKE_UP:               person_detection_light laden (leichtes YOLO)
CAMERA_TRACKING:       tracking_quality_monitor (low power)
MOLOCH_TRACKING_FAR:   YOLO + SCRFD + ArcFace + ptz_controller (full)
MOLOCH_TRACKING_NEAR:  SCRFD + pose_estimation + head_detection (full)
```

### Standing-Still-Detection
```python
# Wenn bbox_velocity < 0.02 für 5+ Sekunden:
#   → publish npu.reduce_clock (homeostasis reagiert)
# Wenn bbox_velocity > 0.05:
#   → publish npu.full_clock
```

### Event Bus Integration
```python
# Subscribed auf:
#   perception_brain.state_changed → load/unload models
#   motion_state_changed (standing_still) → reduce clock
```

---

## MODUL 5: Near Field Handler
**Datei:** `core/perception/near_field_handler.py`  
**Abhängigkeiten:** Bestehender PerceptionEngine  
**RAM:** ~3MB

### Was es tut
Erkennt wenn Person zu nah an der Kamera ist → Smart Tracking verliert Kopf.
Triggert MOLOCH_TRACKING_NEAR State.

### Trigger-Bedingungen
```python
# ALLE müssen True sein für Near-Field-Trigger:
bbox_height_ratio = person_bbox_height / frame_height
near_field = (
    bbox_height_ratio > 0.70 AND
    head_keypoints_missing AND          # Pose: keine Schulter/Kopf-Keypoints
    bbox_center_y > frame_height * 0.5  # Bbox-Mitte unten im Bild
)

# Rückkehr zu CAMERA_TRACKING:
return_condition = bbox_height_ratio < 0.40
```

### Event Bus Integration
```python
# Subscribed auf: perception.person_detected (PFrame)
# Publisht:
#   tracking.near_field_enter  → ptz_handover_controller
#   tracking.near_field_exit   → ptz_handover_controller
```

---

## MODUL 6: Smart Tracking Improvements (in bestehende Module integrieren)

### 6.1 PTZ_JITTER_DETECTION
**In:** `core/perception/tracking_quality_monitor.py` (Modul 2 oben)  
Bereits beschrieben: PTZ-Position 10Hz samplen, 3x Richtungswechsel in 2s = Jitter.

### 6.2 MULTI_PERSON_SCENE
**In:** `core/tracking/ptz_arbiter.py` (bestehend erweitern)
```python
# Wenn Vision Pipeline > 1 Person erkennt:
#   → Ziel-Auswahl nach Priorität:
#     1. face_id == "markus" (bekannte Person)
#     2. Größte Bounding Box
#     3. Zuletzt getracktes Target
```

### 6.3 SEATED_PERSON_DETECTION
**In:** `core/awareness/motion_analyzer.py` (bestehend erweitern)
```python
# Neue activity: "seated"
# Trigger: hip_keypoints vorhanden, knee_keypoints < hip_keypoints Y-Position
# Bei "seated": PTZ-Ankerpunkt → head_center statt body_center
# Event: motion_state_changed {state: "seated"}
```

### 6.4 SMART_SEARCH_RETURN
**In:** `core/tracking/ptz_arbiter.py` (bestehend erweitern)
```python
# Bei target_lost Event:
#   1. last_known_position speichern (pan, tilt)
#   2. PTZ fährt zu last_known_position
#   3. 3 Sekunden warten
#   4. Erst dann patrol_scan aktivieren
```

### 6.5 FACE_LOCK_MODE
**In:** `core/tracking/ptz_arbiter.py` (bestehend erweitern)
```python
# Wenn SCRFD face_confidence > 0.8 für 5+ consecutive Frames:
#   → face_lock_active = True
#   → PTZ Zielkoordinaten: face_bbox_center statt body_bbox_center
# face_lock_active = False wenn face_confidence < 0.5
```

---

## MODUL 7: Perception Brain State Machine
**Datei:** `core/perception/perception_brain.py`  
**RAM:** ~10MB

### Was es tut
Zentrale State Machine die alle 6 Zustände koordiniert.
Aggregiert Signale von Modul 1-5 und triggert Übergänge.

### States
```
IDLE → WAKE_UP:           camera.motion_detected OR camera.human_detected
WAKE_UP → CAMERA_TRACKING: camera.smart_tracking_active (nach NPU Boot)
CAMERA_TRACKING → MOLOCH_TRACKING_FAR: tracking.takeover_required OR tracking.jitter_detected
CAMERA_TRACKING → MOLOCH_TRACKING_NEAR: tracking.near_field_enter
MOLOCH_TRACKING_NEAR → CAMERA_TRACKING: tracking.near_field_exit
MOLOCH_TRACKING_FAR → CAMERA_TRACKING: tracking.camera_sufficient
CAMERA_TRACKING → IDLE: no_motion_timeout (120s) AND camera.home_position
```

### Event Bus Output
```python
# Bei jedem State-Wechsel:
event_bus.publish(MolochEvent(
    topic="perception_brain.state_changed",
    data={"from": old_state, "to": new_state, "reason": trigger},
    priority=1
))
```

---

## INTEGRATION IN moloch_service.py

```python
# Neue Imports (NUR hinzufügen, nichts entfernen):
from core.perception.onvif_event_listener import ONVIFEventListener
from core.perception.tracking_quality_monitor import TrackingQualityMonitor
from core.perception.ptz_handover_controller import PTZHandoverController
from core.autonomy.npu_power_manager import NPUPowerManager
from core.perception.near_field_handler import NearFieldHandler
from core.perception.perception_brain import PerceptionBrain

# In _start_services():
self.onvif_listener = ONVIFEventListener(settings)
self.tracking_quality = TrackingQualityMonitor()
self.ptz_handover = PTZHandoverController()
self.npu_power = NPUPowerManager()
self.near_field = NearFieldHandler()
self.perception_brain = PerceptionBrain()
```

---

## IMPLEMENTIERUNGSREIHENFOLGE FÜR CLAUDE CODE

**Priorität 1 (sofort):**
1. `tracking_quality_monitor.py` — Score-Berechnung, Jitter-Detection
2. `near_field_handler.py` — Near-Field-Trigger
3. Smart Tracking Improvements in ptz_arbiter.py (6.2-6.5)

**Priorität 2 (danach):**
4. `onvif_event_listener.py` — ONVIF Events als Wake-Trigger
5. `ptz_handover_controller.py` — Smooth Handover
6. `npu_power_manager.py` — State-basiertes Power Management

**Priorität 3 (zum Schluss):**
7. `perception_brain.py` — State Machine die alles zusammenhält

---

## RAM-BUDGET

| Modul | RAM |
|-------|-----|
| onvif_event_listener | ~15MB |
| tracking_quality_monitor | ~5MB |
| ptz_handover_controller | ~8MB |
| npu_power_manager | ~3MB |
| near_field_handler | ~3MB |
| perception_brain | ~10MB |
| **Gesamt neu** | **~44MB** |
| Bestehendes System | ~2900MB |
| **Gesamtverbrauch** | **~2944MB / 4096MB** |
| **Puffer** | **~1150MB ✅** |

---

## TESTPLAN FÜR JEDEN MODUL

```
Nach jedem Modul:
1. import testen: python3 -c "from core.perception.X import X"
2. Unit-Test: Modul isoliert mit Mock-Daten
3. Integration: Service neu starten, Diagnostics API prüfen
4. RAM-Check: psutil.virtual_memory() nach Service-Start
5. Git commit mit Tag
```
