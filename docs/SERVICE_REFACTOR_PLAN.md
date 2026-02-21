# SERVICE_REFACTOR_PLAN.md

## M.O.L.O.C.H. Service Refactoring — Regel 10 Compliance

**Datum:** 2026-02-22
**Grundlage:** System-Audit vom 22.02.2026, CLAUDE.md Regel 10 (Christian-Prinzip)
**Ziel:** `core/moloch_service.py` von 2539 Zeilen auf ~800 Zeilen reduzieren

---

## 1. DIAGNOSE

### IST-Zustand

`moloch_service.py` = **2539 Zeilen**, **15+ Verantwortlichkeiten** in einer Datei.

| Verantwortlichkeit | Zeilen | Bereich |
|---|---|---|
| CloudController Klasse | 45 | 102–146 |
| RTSP Capture + Auto-Reconnect | 117 | 376–492 |
| NPU Model Management (configure/run) | 94 | 498–591 |
| Inference Loop (SCRFD/ArcFace/YOLO/Hand) | 543 | 597–1139 |
| Cross-Process NPU Coordination | 127 | 1144–1270 |
| Tentakel-Logik (Takeover/Release) | 234 | 1276–1509 |
| Kamera-Status Polling + Bewegungserkennung | 105 | 1515–1619 |
| Frozen Frame Watchdog | 33 | 1625–1657 |
| Autonomous Mode Enable/Disable | 61 | 1663–1723 |
| Cloud/Camera Connect + Smart Tracking | 59 | 1729–1787 |
| LED Signaling (on/off/blink/indicator) | 67 | 1793–1859 |
| Face Recognition (DB/State/Announce) | 29 | 1865–1893 |
| Lifecycle (init/start/stop) | 86 | 1899–1984 |
| Public API (toggle_model, toggle_autonomous) | 111 | 1990–2100 |
| Panel IPC (SHM + Commands) | 266 | 2150–2415 |
| Settings Persistence (load/save) | 109 | 2419–2527 |
| Helper: load_face_db | 18 | 82–99 |

### Regel-10-Verstoesse

```
a) Separation of Concerns   — VERLETZT: 15+ Aufgaben in 1 Datei
b) Fail Isolation            — TEILWEISE: try/except vorhanden, aber alles im selben Prozess-Scope
c) Interface Contract        — VERLETZT: Module greifen direkt auf 60+ self._xxx Attribute zu
d) Pipeline-Architektur      — VERLETZT: Inference Loop mischt Input/Processing/Output/Control
e) Pre/Post-Conditions       — TEILWEISE: einige Checks vorhanden
f) Handshake                 — OK: Cloud-Calls pruefen Ergebnis
g) State Validation          — TEILWEISE: connected-Checks, aber kein formales State-System
h) Health Monitoring         — TEILWEISE: Watchdog existiert, aber kein Heartbeat-System
i) Mux/Exklusiver Zugriff   — OK: HailoManager, _ctx_lock, _transition_lock
j) Atomic Changes            — OK: Settings mit tmp+rename
```

---

## 2. ZIEL-ARCHITEKTUR

```
core/moloch_service.py          (~800 Zeilen, Orchestrator)
  |
  +-- core/cloud_controller.py    CloudController           (async eWeLink Bridge)
  +-- core/rtsp_reader.py         RTSPReader                (Stream + Watchdog)
  +-- core/npu_pipeline.py        NPUPipeline               (Model Lifecycle + IPC)
  +-- core/led_controller.py      LEDController             (Status-LED Signaling)
  +-- core/settings_manager.py    SettingsManager           (Load/Save Config)
  +-- core/panel_ipc.py           PanelIPC                  (SHM Frame + Commands)
  +-- core/tentakel_controller.py TentakelController        (Guardian/Takeover/Release)
  |
  +-- core/perception_engine.py   PerceptionEngine          (EXISTIERT, keine Aenderung)
  +-- core/mpo/autonomous_tracker.py AutonomousTracker      (EXISTIERT, keine Aenderung)
  +-- core/hardware/hailo_manager.py HailoManager           (EXISTIERT, keine Aenderung)
  +-- core/hardware/camera.py     CameraController          (EXISTIERT, keine Aenderung)
  +-- core/hardware/camera_cloud_bridge.py CameraCloudBridge (EXISTIERT, keine Aenderung)
```

### Datenfluss nach Refactoring

```
RTSPReader ──frame──> MolochService (Inference Loop)
                          |
                          +── NPUPipeline.run_model()
                          +── PerceptionEngine.tick()
                          +── LEDController.set() / blink_markus()
                          +── TentakelController.on_detection()
                          |
                          v
                      PanelIPC.write_frame(annotated)
                      PanelIPC.write_status(state)

PanelIPC.poll_commands() ──cmd──> MolochService._dispatch_cmd()
                                      |
                                      +── NPUPipeline.configure/unconfigure
                                      +── TentakelController.takeover/release
                                      +── SettingsManager.save()
                                      +── CloudController.run()
```

---

## 3. MODULE IM DETAIL

### 3.1 `core/cloud_controller.py` — Phase 1

**Extrahiert aus:** Zeile 102–146 (CloudController) + 1729–1787 (connect/toggle)

**Klasse:** `CloudController`

**Interface:**
```python
class CloudController:
    connected: bool

    def start(self) -> None
    def run(self, coro) -> Any
    def connect(self) -> None
    def toggle_smart_tracking(self) -> None
    def set_smart_tracking(self, on: bool) -> bool
```

**Eigene Abhaengigkeiten:** `CameraCloudBridge`, `CloudConfig`, `asyncio`, `threading`
**Groesse:** ~100 Zeilen

---

### 3.2 `core/rtsp_reader.py` — Phase 1

**Extrahiert aus:** Zeile 376–492 (_start_rtsp) + 1625–1657 (_frozen_frame_watchdog)

**Klasse:** `RTSPReader`

**Interface:**
```python
class RTSPReader:
    frozen_restart_count: int

    def __init__(self, url: str, width: int, height: int)
    def start(self) -> None
    def stop(self) -> None
    def get_frame(self) -> Optional[np.ndarray]
    def get_frame_age(self) -> float
```

**Details:**
- Eigener Reader-Thread + Watchdog-Thread
- Frozen-Frame-Detection (Hash-basiert, 10 identische Frames)
- Auto-Reconnect mit Backoff
- Thread-safe Frame-Zugriff via Lock

**Eigene Abhaengigkeiten:** `cv2`, `threading`, `numpy`
**Groesse:** ~170 Zeilen

---

### 3.3 `core/led_controller.py` — Phase 1

**Extrahiert aus:** Zeile 1793–1859

**Klasse:** `LEDController`

**Interface:**
```python
class LEDController:
    def __init__(self, cloud: CloudController)
    def on(self) -> None
    def off(self) -> None
    def blink(self, count: int = 6, interval: float = 0.3) -> None
    def set_indicator(self, on: bool) -> None
    def blink_markus(self) -> None
```

**Details:**
- State-Tracking (vermeidet API-Spam bei unveraendertem Zustand)
- Markus-Blink mit 10s Cooldown
- Blink-Lock (nur eine Sequenz gleichzeitig)
- Thread-safe

**Eigene Abhaengigkeiten:** `CloudController`, `threading`
**Groesse:** ~80 Zeilen

---

### 3.4 `core/settings_manager.py` — Phase 1

**Extrahiert aus:** Zeile 2419–2527

**Klasse:** `SettingsManager`

**Interface:**
```python
class SettingsManager:
    def __init__(self, path: str)
    def load(self) -> dict          # Gibt parsed dict zurueck
    def save(self, data: dict) -> bool  # Atomic write (tmp + rename)
```

**Details:**
- Pure Load/Save, keine Businesslogik
- Atomic Write via tmp + os.replace
- Fehlerbehandlung: korrupte Datei -> leeres dict + Warning

**Eigene Abhaengigkeiten:** `json`, `os`
**Groesse:** ~60 Zeilen

---

### 3.5 `core/panel_ipc.py` — Phase 1

**Extrahiert aus:** Zeile 2150–2228

**Klasse:** `PanelIPC`

**Interface:**
```python
class PanelIPC:
    def write_frame(self, frame: np.ndarray) -> None   # /dev/shm/moloch_frame
    def write_status(self, status: dict) -> None        # /dev/shm/moloch_status.json
    def poll_commands(self) -> List[dict]                # /tmp/moloch_cmd_*.json
    def cleanup(self) -> None                            # IPC Dateien loeschen
```

**Details:**
- SHM Frame: Header (H, W, C, SEQ) + Raw Bytes
- SHM Status: JSON mit Model-State, FPS, Thresholds
- Command Polling: Nummerierte JSON-Dateien + Legacy-Support
- Sequence Counter fuer Frame-Ordering

**Eigene Abhaengigkeiten:** `struct`, `json`, `numpy`, `glob`
**Groesse:** ~100 Zeilen

---

### 3.6 `core/npu_pipeline.py` — Phase 2

**Extrahiert aus:** Zeile 498–591 + 1144–1270 + 1930–1953

**Klasse:** `NPUPipeline`

**Interface:**
```python
class NPUPipeline:
    active_models: List[str]        # Namen der konfigurierten Modelle

    def __init__(self, model_paths: dict, hailo_manager)
    def load_models(self) -> None                       # VDevice + HEF laden
    def configure(self, name: str) -> None              # Modell auf NPU konfigurieren
    def unconfigure(self, name: str) -> None            # Modell freigeben
    def run(self, name: str, input_data: np.ndarray) -> dict  # Inference ausfuehren
    def pause_for_voice(self) -> None                   # VDevice freigeben
    def resume_after_voice(self, models: List[str]) -> None  # VDevice neu aufbauen
    def sync_flags(self) -> dict                        # {scrfd: bool, arcface: bool, ...}
    def watchdog_tick(self) -> None                     # Max-2 + Anti-Oszillation
    def release_all(self) -> None                       # Shutdown
```

**Details:**
- Kapselt VDevice, _models, _output_names, _active_ctx komplett
- _ctx_lock und _configuring Event bleiben intern
- Cross-Process: NPU_VOICE_REQUEST / NPU_VISION_PAUSED Handling
- Auto-Recovery bei leeren Models (3 Versuche)

**Eigene Abhaengigkeiten:** `hailo_platform`, `HailoManager`, `numpy`, `threading`
**Groesse:** ~300 Zeilen

**ACHTUNG:** Das ist die riskanteste Extraktion. Die Inference Loop nutzt `_run_model` im
Hot Path (~15 FPS). Die Indirektion ueber ein Objekt statt `self._run_model()` darf
KEINE messbare Latenz einfuehren. Python-Methodenaufruf auf Objekt ~50ns = irrelevant
bei 20ms Inference-Time.

---

### 3.7 `core/tentakel_controller.py` — Phase 2

**Extrahiert aus:** Zeile 1276–1509 + 1515–1619 + 1663–1723

**Klasse:** `TentakelController`

**Interface:**
```python
class TentakelController:
    moloch_has_control: bool
    manual_mode: bool
    autonomous_mode: bool
    tentakel_enabled: bool

    def __init__(self, cloud: CloudController, npu: NPUPipeline, led: LEDController)
    def takeover(self, reason: str) -> None
    def release(self) -> None
    def check_timeout(self) -> None
    def update_position(self, pan: float, tilt: float) -> None   # Bewegungserkennung
    def toggle_autonomous_manual(self) -> None
    def enable_autonomous(self) -> None
    def disable_autonomous(self) -> None
    def on_detection(self, detection_type: str) -> None   # face/person erkannt
    def get_cam_status(self) -> dict                      # Mode/PTZ/ST Status
```

**Details:**
- Guardian-Modus: Kamera-Bewegung erkennen -> Takeover
- Fliessender Uebergang: ST bleibt AN bis Detection, dann ST AUS + Tracker AN
- Release: Home Position -> ST AN -> Tracker STOP
- Progressive Cooldown (1.5x, max 180s)
- Idle Pre-Load: NPU Modelle vorladen wenn Kamera still steht

**Eigene Abhaengigkeiten:** `CloudController`, `NPUPipeline`, `LEDController`,
`AutonomousTracker`, `CameraController`, `threading`
**Groesse:** ~350 Zeilen

---

## 4. REFACTORING-REIHENFOLGE

### Phase 1: Standalone-Module (kein Risiko)

Module ohne Rueckabhaengigkeiten. Koennen einzeln extrahiert und getestet werden.

| # | Modul | Aus Service entfernen | Risiko |
|---|---|---|---|
| 1.1 | `cloud_controller.py` | Zeile 102–146 + 1729–1787 | Niedrig |
| 1.2 | `rtsp_reader.py` | Zeile 376–492 + 1625–1657 | Niedrig |
| 1.3 | `led_controller.py` | Zeile 1793–1859 | Niedrig |
| 1.4 | `settings_manager.py` | Zeile 2419–2527 | Niedrig |
| 1.5 | `panel_ipc.py` | Zeile 2150–2228 | Niedrig |

**Nach Phase 1:** moloch_service.py ~1900 Zeilen (−640)

### Phase 2: Logik-Module (braucht Interface-Design)

Module mit Querverbindungen. Erfordern definierte Interfaces.

| # | Modul | Aus Service entfernen | Risiko |
|---|---|---|---|
| 2.1 | `npu_pipeline.py` | Zeile 498–591 + 1144–1270 + 1930–1953 | Mittel |
| 2.2 | `tentakel_controller.py` | Zeile 1276–1509 + 1515–1619 + 1663–1723 | Mittel |

**Nach Phase 2:** moloch_service.py ~800 Zeilen (−1740 gesamt)

### Phase 3: Aufraeum-Arbeiten

| # | Aufgabe |
|---|---|
| 3.1 | `_execute_panel_cmd()` in Dispatch-Table umbauen (dict statt elif-Kette) |
| 3.2 | Inference Loop intern strukturieren (Kommentar-Bloecke in Helper-Methoden) |
| 3.3 | Observer-Pattern formalisieren (Typed Events statt String-Keys) |
| 3.4 | `load_face_db()` nach `core/vision/face_database.py` verschieben (existiert bereits!) |

---

## 5. MIGRATIONS-PROTOKOLL PRO MODUL

Fuer JEDES Modul gilt:

```
1. git add -A && git commit -m "BACKUP vor [Modulname] Extraktion"
2. Neue Datei anlegen unter core/
3. Klasse mit Interface schreiben
4. Code aus moloch_service.py in neue Datei verschieben
5. In moloch_service.py: import + Instanz erzeugen + self._xxx Zugriffe umleiten
6. Service starten: sudo systemctl restart moloch
7. Logs pruefen: journalctl -u moloch --since "1 min ago" --no-pager
8. 60s laufen lassen, auf Fehler/Crashes pruefen
9. Wenn OK: git add -A && git commit -m "Refactor: [Modulname] extrahiert"
10. Wenn NICHT OK: git checkout -- . && git clean -fd
```

---

## 6. INTERFACE-KONTRAKTE

### MolochService (Orchestrator) -> Module

```python
# === Konstruktor ===
class MolochService:
    def __init__(self):
        self._settings = SettingsManager(SETTINGS_PATH)
        self._cloud = CloudController()
        self._rtsp = RTSPReader(RTSP_URL, self.PREVIEW_W, self.PREVIEW_H)
        self._npu = NPUPipeline(MODEL_PATHS, get_hailo_manager())
        self._led = LEDController(self._cloud)
        self._ipc = PanelIPC()
        self._tentakel = TentakelController(self._cloud, self._npu, self._led)
        # ... Perception, DailyLearner, GestureDetector bleiben direkt

    def init(self):
        self._npu.load_models()
        self._face_db = load_face_db(FACE_DB_PATH)
        self._rtsp.start()
        self._cloud.connect()

    def start(self, blocking=True):
        # Threads starten:
        #   InferenceLoop      (self._inference_loop)
        #   CamStatusLoop      (self._tentakel.cam_status_loop)  <-- NEU: delegiert
        #   PanelCmdPoll       (self._poll_panel_cmds)
        #   FrozenWatchdog     (self._rtsp.watchdog_loop)        <-- NEU: in RTSPReader
        ...

    def stop(self):
        self._rtsp.stop()
        self._npu.release_all()
        self._ipc.cleanup()
```

### Modul -> Modul Kommunikation

**KEINE direkten Querverbindungen!** Alle Kommunikation laeuft ueber MolochService.

```
FALSCH:  TentakelController importiert LEDController direkt
RICHTIG: TentakelController ruft self._on_takeover_callback() auf,
         MolochService registriert Callback der LEDController.on() aufruft
```

Oder pragmatischer (da wir kein RPC-Framework brauchen):
```
OK:      TentakelController bekommt LEDController im Konstruktor injected
         (Dependency Injection, keine zirkulaeren Imports)
```

---

## 7. RISIKO-BEWERTUNG

| Risiko | Auswirkung | Gegenmassnahme |
|---|---|---|
| NPU-Latenz durch Indirektion | FPS-Einbruch | Benchmark vor/nach: `time.perf_counter()` um `npu.run()` |
| Thread-Safety bei Modul-Grenzen | Race Conditions | Locks bleiben im jeweiligen Modul, nicht ueber Grenzen teilen |
| Import-Reihenfolge | Zirkulaere Imports | Module importieren NUR Standard-Lib + ihre eigenen Dependencies |
| Service-Startup Reihenfolge | Crash beim Boot | `init()` Reihenfolge beibehalten: NPU -> FaceDB -> RTSP -> Cloud |
| RTSP Reconnect waehrend Extraktion | Stream-Ausfall | RTSPReader komplett eigenstaendig testen bevor Integration |
| Tentakel-State Inkonsistenz | Kamera blockiert | TentakelController uebernimmt ALLE State-Variablen komplett |

---

## 8. ERFOLGSKRITERIEN

Nach Abschluss MUSS gelten:

- [ ] `moloch_service.py` < 1000 Zeilen
- [ ] Jedes neue Modul hat genau EINE Verantwortlichkeit
- [ ] `sudo systemctl restart moloch` startet fehlerfrei
- [ ] Inference FPS unveraendert (±2 FPS Toleranz)
- [ ] Tentakel-Takeover/Release funktioniert
- [ ] Panel IPC (Frame + Commands) funktioniert
- [ ] LED-Signaling funktioniert
- [ ] Settings Save/Load funktioniert
- [ ] Kein Modul importiert ein anderes Modul direkt (ausser injected Dependencies)
- [ ] `journalctl -u moloch --priority=err` zeigt keine neuen Fehler

---

## 9. WAS BLEIBT IN `moloch_service.py`

Nach Phase 2 verbleibt:

```python
# ~800 Zeilen
class MolochService:
    # __init__:        Module instanziieren, State-Variablen (~80 Zeilen)
    # init():          Hardware initialisieren (~30 Zeilen)
    # start()/stop():  Thread-Orchestrierung (~50 Zeilen)
    # _inference_loop: SCRFD/ArcFace/YOLO/Hand Processing (~550 Zeilen)
    # Public API:      toggle_model, _dispatch_cmd (~90 Zeilen)
```

Die Inference Loop bleibt im Service weil:
1. Sie ALLE Module orchestriert (NPU + Perception + LED + Tracker + IPC)
2. Sie ~30 State-Variablen im Hot Path liest/schreibt
3. Extraktion wuerde ein komplexes State-Objekt erfordern ohne echten Gewinn
4. Die Loop IST die Kernaufgabe des Service — das ist seine Verantwortlichkeit

---

## 10. NICHT IM SCOPE

Folgende Arbeiten gehoeren NICHT zu diesem Refactoring:

- **Perception Engine umbauen** — laeuft bereits sauber in eigener Datei
- **AutonomousTracker aendern** — eigene Datei, eigene Logik
- **GUI/Panel-Module anfassen** — sind bereits modular (Regel 10 compliant)
- **Neue Features einfuegen** — NUR Struktur-Umbau, keine Funktionsaenderungen
- **Python-Version upgrade** — irrelevant fuer Refactoring
- **Test-Framework einfuehren** — separater Auftrag
- **78 Patch-Dateien aufraeumen** — separater Auftrag
- **Git HEAD detached fixen** — separater Auftrag (KRITISCH, aber anderer Scope)
