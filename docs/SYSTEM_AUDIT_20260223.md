# M.O.L.O.C.H. System-Audit — 2026-02-23 02:30 CET

> Gnadenlos. Alles gefunden.

## Gesamtbewertung

**Service läuft stabil**, alle 6 NPU-Modelle aktiv, Kamera erreichbar, Tracker locked.
Aber unter der Haube: **2 kritische Race Conditions**, **1 Klartext-Passwort in Git**,
ein kaputter GestureDetector, und ~15 Warnungen.

---

## KRITISCH — Sofort fixen

### K1. RTSP-Passwort im Klartext im Source-Code

**Datei**: `moloch_service.py:71`
```python
RTSP_URL = os.environ.get(
    "MOLOCH_RTSP_URL",
    "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
)
```
Username `Moloch_4.5`, Passwort `Auge666` steht im Code UND in der Git-History.
Auch wenn die Zeile geändert wird — `git log -p` zeigt es für immer.

**Fix**: Kamera-Passwort JETZT ändern. Default-Fallback durch leeren String ersetzen,
echte URL nur noch via Env-Variable `MOLOCH_RTSP_URL` (in `~/.profile` oder systemd Unit).

---

### K2. Race Condition: `_conversation` in voice_pipeline.py (kein Lock)

**Datei**: `voice_pipeline.py:426-483`

`self._conversation` wird von mehreren Threads gleichzeitig geschrieben:
- `_process_recording` Thread (PTT → Whisper → Claude)
- `_process_text` Thread (Chat-Eingabe → Claude)
- `reset_conversation()` (GUI-Thread)

Kein Lock. Konkretes Szenario: Markus tippt Text WÄHREND PTT verarbeitet wird →
zwei `user`-Messages hintereinander → Claude API Error → Rollback `pop()` löscht
die falsche Message → Konversations-Korruption.

**Fix**: `self._lock` (existiert bereits) um alle `_conversation`-Zugriffe in `_chat()`.
API-Call AUSSERHALB des Locks (sonst blockiert alles):
```python
with self._lock:
    self._conversation.append({"role": "user", "content": user_text})
    if len(self._conversation) > 10:
        self._conversation = self._conversation[-10:]
    msgs = list(self._conversation)  # Kopie fuer API-Call
response = self._claude_client.messages.create(..., messages=msgs)
with self._lock:
    self._conversation.append({"role": "assistant", "content": text})
```

---

### K3. Race Condition: `_run_model()` ohne Lock nach ctx-Lookup

**Datei**: `moloch_service.py:600-610`

```python
with self._ctx_lock:
    ctx = self._active_ctx.get(name)    # Lock nur fuer Lookup
if not ctx:
    return {}
bindings = ctx["bindings"]              # KEIN LOCK!
bindings.input().set_buffer(...)        # KEIN LOCK!
ctx["configured"].run([bindings], ...)  # KEIN LOCK! — 21ms blockierend
```

Wenn `_unconfigure_model()` (aus PanelCmdPoll-Thread) gleichzeitig läuft:
```python
ctx = self._active_ctx.pop(name, None)  # Entfernt ctx
ctx["ctx_mgr"].__exit__(...)            # Gibt NPU-Ressourcen frei!
```

→ `_run_model()` greift auf freigegebene NPU-Hardware zu → **NPU-Crash möglich**.

**Fix**: Separaten `_inference_lock` einführen, der `_run_model()` UND
`_unconfigure_model()` synchronisiert. Oder: `_configuring` Event konsequent
prüfen VOR dem `ctx`-Zugriff.

---

## WARNUNG — Sollte gefixt werden

### W1. GestureDetector bekommt falsche Keypoints

**Datei**: `moloch_service.py:1109`
```python
gesture = self._gesture_detector.detect(hand_result["landmarks"])
```
`hand_result["landmarks"]` = 21 MediaPipe Hand-Landmarks (Finger).
`GestureDetector.detect()` erwartet 17 COCO Body-Keypoints (Schultern, Knie...).
21 > 17, also kein Early-Return — aber Finger werden als Körperteile interpretiert.

**Ergebnis**: Gesten-Erkennung liefert Nonsens. `hand_gesture` im PerceptionFrame ist unbrauchbar.

**Fix**: Eigenen Hand-Gesture-Detektor schreiben (Finger-Geometrie: Daumen hoch, Peace, Zeigen)
ODER GestureDetector mit Pose-Keypoints füttern statt Hand-Landmarks.

---

### W2. `self._processing` wird nie gesetzt — Doppel-TTS möglich

**Datei**: `voice_pipeline.py`

`self._processing` wird in `__init__` auf `False` gesetzt, aber NIRGENDS auf `True`.
Sollte Doppelverarbeitung (PTT + Spontan-Kommentar gleichzeitig) verhindern.

**Ergebnis**: Zwei TTS-Ausgaben können gleichzeitig über HDMI laufen → Audio-Matsch.

**Fix**: In `_process_recording()` und `_process_text()` am Anfang `self._processing = True`,
im `finally` zurücksetzen.

---

### W3. `_write_face_state()` — kein Atomic Write

**Datei**: `moloch_service.py:2126-2127`
```python
with open(FACE_STATE_PATH, "w") as f:
    json.dump(state, f)
```
Direct Write. `voice_pipeline.py:763` liest dieselbe Datei in anderem Thread.
Reader kann halbfertige JSON erwischen → `json.JSONDecodeError`.

Im Gegensatz dazu: `_write_status_json()` macht es RICHTIG mit `.tmp` + `os.rename()`.

**Fix**: Gleicher Pattern — `.tmp` schreiben, dann `os.rename()`.

---

### W4. `_saved_mic_gain` ohne Default — AttributeError möglich

**Datei**: `moloch_service.py:2852`
```python
cmd.get('mic_gain', self._saved_mic_gain)
```
`_saved_mic_gain` hat keinen Default in `__init__`. Wenn `settings.json` keinen
`audio`-Key hat → `AttributeError` beim ersten `set_audio` IPC-Command.

**Fix**: In `__init__`: `self._saved_mic_gain = 1.0` etc.

---

### W5. Face-State Name-Case-Mismatch

**Datei**: `voice_pipeline.py:768`
```python
if fs.get("name") == "Markus" and time.time() - fs.get("timestamp", 0) < 30:
```
Case-sensitiver Vergleich auf `"Markus"`. In `face_state.json` steht aktuell `"Markus"` (Großbuchstabe),
aber `face_id` im PerceptionFrame ist `name.lower()` = `"markus"`.

**Risiko**: Wenn sich der Schreib-Code ändert, kommen nie mehr spontane Kommentare.

**Fix**: `fs.get("name", "").lower() == "markus"`

---

### W6. `_write_status_json()` hat zwei Writer ohne Mutex

**Datei**: `moloch_service.py:2777-2779`

Wird aus ZWEI Threads aufgerufen:
- InferenceLoop (via `_write_shm()`, ~5-30x/Sek)
- CamStatusLoop (alle 1.5s)

Beide schreiben `moloch_status.tmp` → `os.rename()`. `os.rename()` ist atomar,
aber die `.tmp`-Datei kann von beiden gleichzeitig beschrieben werden.

**Fix**: `_status_write_lock = threading.Lock()` um den Body.

---

### W7. `_write_status_json()` liest Shared State ohne Locks

**Datei**: `moloch_service.py:2735-2781`
```python
"active_models": list(self._active_ctx.keys()),  # _ctx_lock nicht benutzt
"fps": {k: round(v, 1) for k, v in self._fps.items()},  # _fps_lock nicht benutzt
```
`_fps_lock` und `_ctx_lock` existieren, werden aber beim Lesen ignoriert.

**Fix**: Locks beim Lesen verwenden.

---

### W8. 24x `except:` ohne Typ (Bare Excepts)

Verteilt über:
- `autonomous_tracker.py:1086` (main loop!)
- `camera_cloud_bridge.py:415, 710, 1066`
- `voice_pipeline.py` (diverse)
- `moloch_service.py` (diverse)

Verschlucken ALLE Fehler inklusive `KeyboardInterrupt`, `SystemExit`, `MemoryError`.

**Fix**: Mindestens `except Exception:` mit Logging. In main loops: `except Exception as e: logger.error(...)`.

---

## INFO — Zur Kenntnis

### I1. Singleton-Races bei Erst-Initialisierung

Ohne Lock: `get_personality_engine()`, `get_daily_learner()`, `get_model_health()`, `get_perception_buffer()`
Mit Lock (korrekt): `get_memory()`, `get_core_integrator()`

**Risiko**: Nur beim allerersten Aufruf. In der Praxis durch sequentiellen Start in `init()` entschärft.

### I2. `head_roll` — nicht in PerceptionFrame

`estimate_head_pose()` liefert (pitch, yaw, roll). `roll` wird in `draw_name()` angezeigt,
aber PerceptionFrame hat kein `head_roll` Feld. Kein Bug — einfach nie implementiert.

### I3. `face_bbox` und `distance_ratio` nicht in `to_dict()`

Beide Felder existieren im Dataclass, werden befüllt, aber nicht via IPC exportiert.
Kein Crash, aber zukünftige JSON-Consumer sehen die Daten nicht.

### I4. `mic_test` IPC-Command — nicht implementiert

```python
# moloch_service.py:2857
elif action == 'mic_test':
    logger.info("[IPC] Mic Test angefordert (noch nicht implementiert)")
```
Panel-Button "MIC TEST" tut nichts.

### I5. IPC via `/tmp/moloch_cmd_*.json` — keine Authentifizierung

Jeder lokale Prozess kann Commands senden. Auf Single-User-Pi OK,
aber bei Web-Interface oder Remote-Zugriff ein Angriffsvektor.

### I6. `w1_bus_master` Kernel-Thread frisst 80% CPU

Nicht Moloch-Code, sondern ein 1-Wire Bus-Treiber. Falls kein DS18B20
Temperatursensor angeschlossen ist: `sudo modprobe -r w1_therm w1_gpio` spart einen CPU-Kern.

### I7. Dead Code — nicht aktiv verwendete Module

| Datei | Status |
|-------|--------|
| `core/vision/hailo_analyzer.py` | Nicht importiert |
| `core/vision/vision_worker.py` | Nicht importiert |
| `core/vision/unified_pipeline.py` | Nicht importiert |
| `core/vision/gst_hailo_detector.py` | Nicht importiert |
| `core/vision/gst_hailo_pose_detector.py` | Nicht importiert |
| `core/perception/perception_manager.py` | Nicht importiert |
| `core/tts/tts_manager.py` | PREPARATION ONLY |
| `core/tts/selection/` (Package) | PREPARATION ONLY |
| `core/dashboard.py` | Nicht importiert |
| `core/environment_watcher.py` | Nicht importiert |

~5.600 Zeilen toter Code. Nicht schädlich, aber Ballast.

### I8. Import-Konflikt `core/tts.py` vs `core/tts/` (Package)

Beides existiert. Python bevorzugt das Package. Die einzelne `tts.py` wird ignoriert.
Funktioniert zufällig, aber ist ein Wartungs-Risiko.

### I9. TODO/FIXME/HACK Markierungen im Code

```
core/personality/personality_engine.py: # TODO: Shadow-Mode vollständig implementieren
core/calibration_engine.py:             # HACK: sleep statt proper callback
core/mpo/autonomous_tracker.py:         # FIXME: kann stuck bleiben wenn Kamera nicht antwortet
```
Plus ~15 weitere. Keine davon crash-kritisch.

### I10. Swap-Nutzung bei 577 MB

RAM: 2.1 GB used / 3.9 GB total. 1.8 GB available (inkl. Cache).
577 MB Swap aktiv. Nicht kritisch, aber zeigt dass das System gelegentlich
unter Speicherdruck steht. Claude-Prozess allein frisst 400 MB.

---

## CLEAN — Alles gut

| Bereich | Status |
|---------|--------|
| Service | `active (running)`, 0 Errors in 5 Min |
| NPU | Alle 6 Modelle aktiv, FPS stabil (SCRFD 42, ArcFace 88, YOLOv8m 35, Hand 99, Pose 17, Total 9) |
| SSD1 | 6% belegt, 414 GB frei |
| SSD2 | 4% belegt, 461 GB frei, NTFS3 stabil gemountet |
| Kamera | Ping 8ms, RTSP aktiv, Tracker locked |
| CPU-Temp | 64°C (Throttling erst bei 80-85°C) |
| Langzeitgedächtnis | identity.json, facts.json (54 Fakten), core_state.json — alle aktuell |
| Conversations | 2 Dateien (22.02 + 23.02), werden geschrieben |
| Shared Memory | `/dev/shm/moloch_frame` (691 KB) + `moloch_status.json` (2.4 KB) — aktuell |
| eWeLink Cloud | Verbunden, LED-Steuerung funktioniert |
| Perception Weights | 370k Entscheidungen, Gewichte plausibel |
| Thresholds | SCRFD 0.4, ArcFace 0.6, YOLO 0.7 — sinnvoll |

---

## Zusammenfassung nach Priorität

| Prio | # | Was | Aufwand |
|------|---|-----|---------|
| **KRITISCH** | K1 | RTSP-Passwort aus Code + Kamera-PW ändern | 10 Min |
| **KRITISCH** | K2 | `_conversation` Lock in voice_pipeline.py | 15 Min |
| **KRITISCH** | K3 | `_run_model()` Lock für NPU-Safety | 30 Min |
| WARNUNG | W1 | GestureDetector fixen (falsche Keypoints) | 2h |
| WARNUNG | W2 | `_processing` Flag aktivieren | 5 Min |
| WARNUNG | W3 | `_write_face_state()` atomic write | 5 Min |
| WARNUNG | W4 | `_saved_mic_gain` Default | 2 Min |
| WARNUNG | W5 | Face-Name case-insensitiv | 1 Min |
| WARNUNG | W6 | Status-Writer Lock | 5 Min |
| WARNUNG | W7 | Status-Reader Locks benutzen | 10 Min |
| WARNUNG | W8 | Bare excepts eingrenzen | 30 Min |
| INFO | I1-I10 | Singletons, Dead Code, Cleanup | Gelegentlich |
