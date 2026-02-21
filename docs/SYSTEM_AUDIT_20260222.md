# M.O.L.O.C.H. System-Audit

**Datum:** 2026-02-22, 00:29 Uhr
**Erstellt von:** Claude Opus 4.6 (automatisiert)
**Zweck:** Komplette Bestandsaufnahme — NUR Dokumentation, NICHTS geaendert

---

## 1. HARDWARE

### CPU
| Eigenschaft | Wert |
|---|---|
| Architektur | ARM Cortex-A76 (aarch64) |
| Kerne | 4 |
| Max Frequenz | 2.4 GHz |
| BogoMIPS | 108.00 |
| Throttled | `0x0` (kein Throttling, kein Under-Voltage) |

### RAM
| | Total | Benutzt | Frei | Verfuegbar |
|---|---|---|---|---|
| RAM | 3.9 GB | 2.2 GB | 904 MB | 1.7 GB |
| Swap | 2.0 GB | 190 MB | 1.8 GB | — |

Swap-Typ: zram (2 GB, Partition), kein Loop-Swap aktiv (CLAUDE.md sagt "2 GB zram + 2 GB loop" — Loop fehlt)

### Speicher (NVMe SSDs)
| Device | Groesse | Mount | FS | Benutzt | Frei | Use% |
|---|---|---|---|---|---|---|
| sdb2 (SSD1) | 458 GB | `/` | ext4 | 26 GB | 415 GB | 6% |
| sdb1 (Boot) | 510 MB | `/boot/firmware` | vfat | 78 MB | 433 MB | 16% |
| sda2 (SSD2) | 477 GB | `/mnt/moloch-data` | NTFS | 9.6 GB | 468 GB | 3% |
| zram0 | 2 GB | [SWAP] | swap | 190 MB | — | — |

SSD2 Verzeichnisse:
```
7.6G   /mnt/moloch-data/backups/
662M   /mnt/moloch-data/hailo/
1.0G   /mnt/moloch-data/reference/
197M   /mnt/moloch-data/qdrant/
648K   /mnt/moloch-data/daily/
```

### Temperatur
| Sensor | Wert |
|---|---|
| SoC | 66.4°C |

### Hailo NPU
| Eigenschaft | Wert |
|---|---|
| Device | `/dev/hailo0` (crw-rw-rw-) |
| Architektur | HAILO10H |
| Firmware | 5.1.1 (release, app) |
| HailoRT | 5.1.1 |
| Aktueller Nutzer | python3 (PID 20210, moloch.service) |
| TOPS | 40 |

---

## 2. OS

| Eigenschaft | Wert |
|---|---|
| Distribution | Debian GNU/Linux 13 (trixie) |
| Kernel | 6.12.62+rpt-rpi-v8 (aarch64) |
| Kernel Datum | 2025-12-18 |
| Uptime | 3h 05min (seit 21:24 am 21.02.2026) |
| Load Average | 5.43 / 4.01 / 3.82 |
| Angemeldete User | 4 |
| Display Manager | LightDM |
| VNC | wayvnc + vncserver-x11-serviced |

### Aktive Services (Auswahl)
```
moloch.service          active  running   M.O.L.O.C.H. Service (Headless)
moloch-fan.service      active  exited    Fan Control (Trip-Points gesetzt)
moloch-ap.service       FAILED            MOLOCH_DIRECT WiFi AP (dnsmasq crash)
docker.service          active  running   Docker Engine
containerd.service      active  running   containerd
ssh.service             active  running   OpenSSH
bluetooth.service       active  running   Bluetooth
hostapd.service         active  running   WiFi AP (trotz moloch-ap Fehler)
NetworkManager.service  active  running   Network Manager
cups.service            active  running   CUPS Drucker
wayvnc.service          active  running   VNC Server
```

---

## 3. NETZWERK

### Interfaces
| Interface | IP | Status | Hinweis |
|---|---|---|---|
| wlan0 | 192.168.178.24/24 | UP | Primaere MOLOCH IP (CLAUDE.md) |
| eth0 | 192.168.178.30/24 | UP | Ethernet (NICHT in CLAUDE.md!) |
| docker0 | 172.17.0.1/16 | UP | Docker Bridge |
| lo | 127.0.0.1/8 | UP | Loopback |

**ACHTUNG:** CLAUDE.md dokumentiert Pi5 = 192.168.178.24 (SSH). Das ist wlan0. Ethernet (eth0) hat .30 — wird nirgends dokumentiert.

### Routen
```
default via 192.168.178.1 (eth0, metric 100)    <- Fritz!Box
default via 192.168.178.1 (wlan0, metric 600)   <- Fritz!Box (Backup)
172.17.0.0/16 via docker0
```

### DNS
```
nameserver 192.168.178.1 (fritz.box)
```

### Offene Ports
| Port | Dienst | Bind |
|---|---|---|
| 22 | SSH | 0.0.0.0 + [::] |
| 111 | rpcbind | 0.0.0.0 + [::] |
| 631 | CUPS | 127.0.0.1 |
| 5900 | VNC | * |
| 6333 | Qdrant (HTTP) | 0.0.0.0 + [::] |
| 6334 | Qdrant (gRPC) | 0.0.0.0 + [::] |
| 31284 | Node (VSCode) | 127.0.0.1 |
| 40699 | VSCode | 127.0.0.1 |

**ACHTUNG:** Qdrant (6333/6334) bindet auf 0.0.0.0 — aus dem LAN erreichbar!

---

## 4. PYTHON

| Eigenschaft | Wert |
|---|---|
| Version | Python 3.13.5 |
| Packages gesamt | 410 |

### Relevante Packages
| Package | Version | Hinweis |
|---|---|---|
| hailort | 5.1.1 | NPU Runtime |
| hailo-tappas-core-python-binding | 5.1.0 | Tappas Bindings |
| hailo-apps | 25.12.0 | Editable install |
| opencv-python | 4.13.0.90 | |
| opencv | 4.10.0 | Doppelte Installation! |
| numpy | 2.2.4 | |
| torch | 2.10.0 | PyTorch (auf ARM!) |
| onnxruntime | 1.23.2 | Emotion/Age Detection |
| faster-whisper | 1.2.1 | STT |
| ctranslate2 | 4.6.3 | Whisper Backend |
| anthropic | 0.76.0 | Claude API |
| Flask | 3.1.1 | |
| pydantic | 2.12.5 | |
| onvif-zeep | 0.2.12 | ONVIF PTZ |
| qdrant-client | 1.16.2 | Vector DB |
| scikit-learn | 1.8.0 | |
| scipy | 1.17.0 | |
| pillow | 11.1.0 | |

**ACHTUNG:** `opencv` (4.10.0) UND `opencv-python` (4.13.0.90) installiert — moeglicher Versionskonflikt.

---

## 5. MOLOCH CORE — Python-Dateien

**Gesamt:** 227 .py Dateien, 67.994 Zeilen

### Top 30 nach Groesse
| Zeilen | Datei |
|---|---|
| 2539 | `core/moloch_service.py` |
| 2500 | `core/gui/moloch_unified_panel.py` |
| 1342 | `scripts/moloch_vision_lab.py` |
| 1207 | `core/hardware/camera_cloud_bridge.py` |
| 1149 | `core/hardware/camera.py` |
| 1085 | `core/console/moloch_console.py` |
| 1070 | `core/mpo/autonomous_tracker.py` |
| 1061 | `core/vision/gst_hailo_pose_detector.py` |
| 927 | `core/calibration_engine.py` |
| 913 | `config/eye_live_scan.py` |
| 859 | `core/hardware/thermal_manager.py` |
| 853 | `core/gui/camera_control_panel.py` |
| 835 | `core/speech/audio_pipeline.py` |
| 833 | `context/system_autonomy.py` |
| 800 | `config/eye_test_protocol.py` |
| 780 | `core/vision/unified_pipeline.py` |
| 777 | `core/personality/personality_engine.py` |
| 763 | `core/gui/hailo_control_panel.py` |
| 746 | `core/gui/popups/popup_audio.py` |
| 735 | `core/status.py` |
| 717 | `core/audio/audio_manager.py` |
| 704 | `core/hardware/hailo_manager.py` |
| 689 | `core/perception/hailo_postprocess.py` |
| 684 | `patch_panel_bridge.py` |
| 663 | `scripts/onvif_capability_scan.py` |
| 650 | `core/perception/perception_manager.py` |
| 628 | `core/perception_engine.py` |
| 620 | `scripts/self_diagnosis.py` |
| 609 | `config/eye_auto_test.py` |
| 599 | `core/vision/vision_worker.py` |

---

## 6. MOLOCH SERVICES (systemd)

### moloch.service
| Eigenschaft | Wert |
|---|---|
| Status | **active (running)** |
| Unit File | `/etc/systemd/system/moloch.service` |
| Enabled | ja |
| PID | 20210 |
| Start | 2026-02-21 23:45:58 |
| CPU Time | 48min 35s (in 43min Laufzeit!) |
| Tasks | 34 |
| Command | `python3 -c "from core.moloch_service import MolochService; s = MolochService(); s.init(); s.start(blocking=True)"` |
| Letzte Fehler | **Keine** (journalctl --priority=err leer) |

### moloch-fan.service
| Eigenschaft | Wert |
|---|---|
| Status | **active (exited)** — OK, One-Shot |
| Trip Points | 50°C / 55°C / 65°C / 75°C |

### moloch-ap.service
| Eigenschaft | Wert |
|---|---|
| Status | **FAILED** |
| Fehler | dnsmasq.service konnte nicht starten |
| Enabled | ja (startet bei jedem Boot und failt) |

### Docker Container
| Container | Image | Status | Ports |
|---|---|---|---|
| qdrant | qdrant/qdrant | Up 3h | 6333-6334 |

---

## 7. MODELLE

### Hailo HEF Modelle (`/mnt/moloch-data/hailo/models/`)
| Modell | Groesse | Aktiv in Service | Datum |
|---|---|---|---|
| scrfd_10g.hef | 5.8 MB | Ja (Face Detection) | 2026-02-09 |
| arcface_mobilefacenet.hef | 2.6 MB | Ja (Face Recognition) | 2026-02-09 |
| yolov8m_h10.hef | 21 MB | Ja (Person Detection) | 2026-02-05 |
| hand_landmark_lite.hef | 1.3 MB | Ja (Hand Landmarks) | 2026-01-05 |
| yolov8s_pose_h10.hef | 14 MB | Nein (nicht in MODEL_PATHS) | 2026-02-05 |
| yolov8m_pose_h10.hef | 28 MB | Nein | 2026-02-05 |
| yolov11m_h10.hef | 27 MB | Nein | 2026-02-05 |
| resnet_v1_50_h10.hef | 23 MB | Nein | 2026-02-05 |
| yolov5n_seg_h10.hef | 3.4 MB | Nein | 2026-02-05 |

**Hinweis:** `yolov8s_pose_h10.hef` steht in CLAUDE.md als aktiv, ist aber NICHT in `MODEL_PATHS` von `moloch_service.py`.

### Piper TTS Voices (`~/moloch/models/voices/`)
| Voice | Groesse | Rolle |
|---|---|---|
| de_DE-thorsten-high.onnx | 109 MB | Guardian/Shadow Stimme |
| de_DE-karlsson-low.onnx | 61 MB | Kobold Stimme |
| de_DE-thorsten-low.onnx | 61 MB | |
| de_DE-thorsten-medium.onnx | 61 MB | |
| de_DE-kerstin-low.onnx | 61 MB | |
| de_DE-pavoque-low.onnx | 61 MB | |
| de_DE-ramona-low.onnx | 61 MB | |
| de_DE-eva_k-x_low.onnx | 20 MB | |

**Gesamt:** 8 Voices, ~495 MB

---

## 8. DATEN

### Face-DB (`data/face_embeddings.json`)
| Eigenschaft | Wert |
|---|---|
| Personen in DB | 1 (nur **Markus**) |
| Embedding-Dimensionen | 512 |

### Face-Fotos (`faces/`)
| Person | Fotos |
|---|---|
| markus | 5 |
| franzi | 0 |
| lilly | 0 |
| meise | 0 |
| ray | 0 |
| sven | 0 |
| unknown | 0 |

**PROBLEM:** 6 von 7 Crew-Mitgliedern haben KEINE Fotos. Face-DB kennt nur Markus.

### Snapshots
| Verzeichnis | Anzahl |
|---|---|
| `~/moloch/snapshots/` | 161 |
| `~/moloch/data/snapshots/` | 18 |

**ACHTUNG:** Zwei Snapshot-Verzeichnisse! CLAUDE.md dokumentiert nur `snapshots/`.

### Daily Learner
| Eigenschaft | Wert |
|---|---|
| Bilder | 0 |
| Verzeichnis | `data/daily_learner/` existiert NICHT |

### Weitere Daten
```
data/
  calibration_results.json    7.8 KB
  diagnosis_history.json      204 B
  face_embeddings.json        11 KB
  last_diagnosis.json         988 B
  perception_history.json     49 KB
  faces/                      (aeltere Face-DB Kopie?)
  memory/
  snapshots/                  18 Bilder
```

---

## 9. GIT

| Eigenschaft | Wert |
|---|---|
| Branch | **HEAD detached from 0f32398** |
| main Branch | vorhanden (lokal + remote) |
| Remote | origin/main |

**ACHTUNG:** HEAD ist DETACHED! Nicht auf main.

### Uncommitted Changes
```
modified:   config/perception_weights.json
modified:   config/settings.json
modified:   core/moloch_service.py
```

### main vs HEAD
HEAD (556ed10) ist 5 Commits VOR main (0f32398):
```
556ed10  BACKUP vor Hand-Landmark Inference Block
74c1cbf  BACKUP vor Hand-Detection Score-Boost
fa1d3ae  BACKUP vor LED-Button Umbenennung
cb5273c  BACKUP vor Alarm-Toggle Fix
b799e1a  BACKUP vor Alarm-Toggle Fix
```

main (0f32398):
```
0f32398  BACKUP vor Tracking-Redesign
63ed969  Fix: Tracking-Release faehrt erst auf Home
c2bd0f8  Fix: Save Settings speichert jetzt Audio, Camera, Thresholds, Hand-Occlusion
4eaede4  Fix: frame_w/frame_h -> fw/fh in estimate_head_pose
57559a1  Fix: toggle_autonomous schaltet auch tentakel_enabled
```

### Letzte 20 Commits
```
556ed10  BACKUP vor Hand-Landmark Inference Block in moloch_service.py
74c1cbf  BACKUP vor Hand-Detection Score-Boost in perception_engine.py
fa1d3ae  BACKUP vor LED-Button Umbenennung in panel_ewelink.py
cb5273c  BACKUP vor Alarm-Toggle Fix in moloch_service.py
b799e1a  BACKUP vor Alarm-Toggle Fix in moloch_service.py
13bcb70  BACKUP vor popup_gallery.py Tab-Korrektur
f95db1c  BACKUP vor LED nightVision Fix
343350e  BACKUP vor Snapshot-Galerie Popup + GALERIE Button
4b6f457  BACKUP vor DailyLearner maybe_snapshot Integration
f57a0ad  BACKUP vor Snapshot-Galerie Popup + Button
0b09e32  BACKUP vor Snapshot Handler Fix
bf88989  Service: Cloud Handler — LED, Alarm, Snapshot, Status LED, Sync
6e748ec  Fix: Position-Button bleibt aktiv eingefaerbt
d1f74bb  BACKUP vor Position Toggle Fix
da409d9  Positionen: Werkstatt + Wohnzimmer mit Farb-Feedback
990d2c7  Fix: AUTONOM Button liest manual_mode statt autonomous_mode
6852de7  BACKUP vor AUTONOM Key Fix
573484f  Panel: Modi vereinfacht — nur AUTONOM + TEACHEN
7221e96  Fix: Status Keys in allen Modulen an echten Service-Status angepasst
59545b3  BACKUP vor Status Key Mapping
```

---

## 10. HEALTH

### RTSP Stream
| Eigenschaft | Wert |
|---|---|
| Erreichbar | **JA** |
| Codec | H.264, 1920x1080 @ 20fps |
| Audio | PCM A-Law |
| URL | `rtsp://...@192.168.178.25:554/av_stream/ch0` |

### eWeLink Cloud
| Eigenschaft | Wert |
|---|---|
| Verbunden | **JA** (Logs zeigen LED toggle) |
| Letzte Aktivitaet | 00:28:36 (sledOnline toggle) |

### NPU Device
| Eigenschaft | Wert |
|---|---|
| Device frei | **NEIN** (belegt von PID 20210 / moloch.service) |
| `lsof /dev/hailo0` | python3 PID 20210 |

### Bluetooth SmartMic
| Eigenschaft | Wert |
|---|---|
| Paired | ja |
| Bonded | ja |
| Connected | **NEIN** |

### Qdrant
| Eigenschaft | Wert |
|---|---|
| Health | `healthz check passed` |
| Port | 6333 (HTTP), 6334 (gRPC) |

### IPC (Shared Memory)
| Datei | Groesse |
|---|---|
| `/dev/shm/moloch_frame` | 922 KB |
| `/dev/shm/moloch_status.json` | 1.1 KB |

### System Load
| | Wert |
|---|---|
| Load Average | 5.43 / 4.01 / 3.82 |
| CPU Cores | 4 |

**ACHTUNG:** Load Average (5.43) liegt UEBER CPU-Kern-Anzahl (4). System ist ueberlastet.

---

## 11. CONFIG

### settings.json
```json
{
  "version": 1,
  "thresholds": {
    "scrfd_conf": 0.4,
    "scrfd_nms": 0.4,
    "arcface_thresh": 0.6,
    "yolo_conf": 0.5
  },
  "hand_occlusion": {
    "timeout": 4.0,
    "streak": 8,
    "recency": 2.5
  },
  "audio": {
    "mic_gain": 1.08,
    "agc_enabled": true,
    "noise_gate_db": -36.0
  },
  "camera": {
    "ptz_speed": 15.0,
    "led_enabled": false,
    "ir_mode": "Aus"
  }
}
```

### perception_weights.json
```json
{
  "version": 1,
  "total_decisions": 312700,
  "weights": {
    "scrfd": -0.1,
    "arcface": -0.1,
    "yolov8m": -0.05,
    "pose": -0.1,
    "hand_landmark": 0.0
  },
  "effective_scores": {
    "scrfd": 0.5,
    "arcface": 0.4,
    "yolov8m": 0.35,
    "hand_landmark": 0.2
  }
}
```

**Hinweis:** 312.700 Decisions aufgezeichnet. Alle Weights sind negativ (Modelle wurden abgestraft). `hand_landmark` hat 0.0 (neutral).

### CLAUDE.md
- Vorhanden: **JA** (344 Zeilen)
- Letztes Update: Enthaelt Hailo-10H Architektur, Panel-Struktur, Regel 10

### Environment Variables (redacted)
```
EWELINK_APP_ID_1=***
EWELINK_APP_SECRET_1=***
EWELINK_USERNAME=***
EWELINK_PASSWORD=***
MOLOCH_CAMERA_USER=***
MOLOCH_CAMERA_PASS=***
MOLOCH_CAMERA_HOST=***
MOLOCH_RTSP_URL=***
```

### Config-Dateien Inventar (35 JSON-Dateien in `config/`)
```
api_keys.json                 camera_cloud.json
camera_home.json              controlled_autonomy.json
display_labels.json           eye_capabilities.json
eye_discovery_fresh.json      eye_presets.json
eye_presets_raw.json          hailo10h_capabilities.json
ha_phase1_raw.json            hardware_autonomy.json
hardware_identity.json        identity_registry.json
model_registry.json           moloch_context.json
moloch_identity.json          moloch_vision_pipeline.json
npu_config.json               onvif_phase2_raw.json
perception.json               perception_weights.json
ptz_limits.json               rec_alarm_taste.json
rec_alarm_taste_2.json        rec_calibrate_check.json
rec_indikator_led.json        rec_infrarot_led.json
rec_ptz_kalibrierung.json     rec_ptz_kalibrierung_2.json
settings.json                 sonoff_camera.json
vision_modes.json             vision_pipeline.json
vision_runtime_config.json
```

---

## 12. PROBLEME

### KRITISCH

1. **Git HEAD DETACHED** — HEAD ist nicht auf main. 5 Commits existieren nur im detached state. Bei einem `git checkout main` wuerden diese Commits verloren gehen falls sie nicht gemerged/rebased werden.

2. **Load Average 5.43 bei 4 Kernen** — System ist dauerhaft ueberlastet. CPU-Time des Service: 48min in 43min Laufzeit = mehr als 1 Kern dauerbeschlagnahmt. Docker + Qdrant + Service + VNC + Desktop fressen CPU.

3. **Face-DB nur Markus** — 6 Crew-Mitglieder (Ray, Lilly, Meise, Sven, Franzi) haben KEINE Fotos in `faces/` und KEINEN Eintrag in `face_embeddings.json`. ArcFace erkennt nur Markus, alle anderen sind "Unbekannt".

4. **moloch-ap.service failt bei jedem Boot** — Enabled aber dnsmasq crashed. Erzeugt Fehlerlogs bei jedem Systemstart.

### MITTEL

5. **SmartMic nicht verbunden** — Bluetooth paired/bonded aber `Connected: no`. Kein aktives Mikrofon.

6. **Zwei Snapshot-Verzeichnisse** — `~/moloch/snapshots/` (161 Bilder) und `~/moloch/data/snapshots/` (18 Bilder). Unklar welches die primaere Quelle ist. CLAUDE.md dokumentiert nur `snapshots/`.

7. **78 Patch/Fix-Dateien im Wurzelverzeichnis** — `patch_*.py` und `fix_*.py` Dateien (~720 KB) liegen im Repo-Root. Diese scheinen historische Einmal-Patches zu sein und sind vermutlich alle bereits angewendet.

8. **Doppelte OpenCV-Installation** — `opencv` 4.10.0 und `opencv-python` 4.13.0.90. Versionskonflikt moeglich.

9. **Daily Learner Verzeichnis fehlt** — `data/daily_learner/` existiert nicht, obwohl der Code DailyLearner referenziert und der Service ihn initialisiert.

10. **Ethernet-IP nicht dokumentiert** — eth0 hat 192.168.178.30, CLAUDE.md dokumentiert nur wlan0 (192.168.178.24). Bei Ethernet-Verbindung aendert sich die erreichbare IP.

11. **Temperatur 66.4°C** — Nicht kritisch, aber warm. Fan-Trip-Points starten bei 50°C, also Fan laeuft bereits.

### NIEDRIG

12. **Qdrant bindet auf 0.0.0.0** — Ports 6333/6334 sind aus dem gesamten LAN erreichbar. Kein Auth konfiguriert.

13. **Perception Weights alle negativ** — Nach 312.700 Decisions sind scrfd (-0.1), arcface (-0.1), yolov8m (-0.05), pose (-0.1) alle abgestraft. System bevorzugt weniger Modelle (was bei 2-Slot-Limit sinnvoll sein kann).

14. **panel_crash.log zeigt historische Fehler:**
    - `FileNotFoundError` bei `/tmp/moloch_npu_voice_request` (Race Condition, 2026-02-14) — wurde inzwischen mit `try/except FileNotFoundError` gefixt
    - `NameError: name 'scrfd' is not defined` (2026-02-15) — String-Literal vergessen, wurde gefixt

15. **system_check.log zeigt HailoRT Fehler** — `HAILO_DRIVER_OPERATION_FAILED(36)` und `HAILO_CONNECTION_REFUSED(89)`. Passiert wenn mehrere Prozesse gleichzeitig auf den NPU zugreifen wollen. Hailo Manager sollte das verhindern.

16. **yolov8s_pose_h10.hef Inkonsistenz** — CLAUDE.md listet dieses Modell als "aktiv", aber es ist nicht in `MODEL_PATHS` im Service. Dokumentation veraltet.

17. **`data/faces/` vs `faces/`** — Zwei Face-Verzeichnisse existieren. `faces/` (Root) hat die Crew-Ordner, `data/faces/` ist ein aelteres Verzeichnis.

18. **3 uncommitted Changes** — `perception_weights.json`, `settings.json`, `moloch_service.py` haben ungesicherte Aenderungen.

19. **Swap-Diskrepanz** — CLAUDE.md sagt "2 GB zram + 2 GB loop (4 GB total)". Tatsaechlich nur zram aktiv (2 GB). Loop-Swap fehlt.

20. **CUPS laeuft** — Druckdienst aktiv auf einem Headless-Surveillance-System. Unnoetig.

---

## Zusammenfassung

| Kategorie | Status |
|---|---|
| Hardware | OK (66°C, kein Throttle, NPU aktiv) |
| OS | Debian 13 Trixie, Kernel 6.12, aktuell |
| Netzwerk | OK (wlan0 + eth0, Fritz!Box) |
| Service | **LAEUFT** (seit 43min, keine Fehler) |
| NPU | Belegt von Service, FW 5.1.1 |
| RTSP | **AKTIV** (1080p/20fps) |
| Cloud | **VERBUNDEN** |
| Face-DB | **NUR MARKUS** (Rest fehlt) |
| Git | **DETACHED HEAD** (5 ungemergte Commits) |
| Load | **HOCH** (5.43 bei 4 Kernen) |
| Bluetooth | **GETRENNT** |
| Codebase | 227 .py, ~68k Zeilen, 78 Patch-Dateien |
