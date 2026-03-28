# M.O.L.O.C.H. — TOOLS PLAN
# Scripts und Werkzeuge fuer sicherere Entwicklung
# Stand: 2026-03-28

---

## Uebersicht

8 Scripts in 3 Prioritaetsstufen. Alle in `~/moloch/scripts/`.
Geschaetzter Gesamtaufwand: ~1000 LOC Python.

| Prio | Script | LOC | Zweck |
|------|--------|-----|-------|
| P0 | preflight.py | ~80 | Baseline VOR Aenderung |
| P0 | postflight.py | ~120 | Vergleich NACH Aenderung |
| P0 | smoke_test.py | ~100 | 8-Punkt-Checkliste nach Reboot |
| P1 | gst_lint.py | ~200 | GStreamer-String Validator |
| P1 | danger_check.py | ~150 | Pre-Commit Hook |
| P2 | baseline_capture.py | ~80 | Cron-basiertes Metrik-Logging |
| P2 | valve_test.py | ~120 | TAPPAS Valve Isolation |
| P2 | config_guard.py | ~150 | Runtime-State aus Git entfernen |

---

## P0 — Sofort bauen (Fundament)

### 1. scripts/preflight.py — Pre-Change Baseline

**Zweck**: System-Zustand VORHER erfassen. Liefert die "Vorher"-Werte fuer postflight.py.

**Input**: Keine Argumente noetig.

**Datenquellen**:
- `/dev/shm/moloch_status.json` → fps, ram_mb, cpu_temp, thread_count, frame_age
- `systemctl is-active moloch` → Service-Status
- `git status --porcelain` → Git clean/dirty
- `/sys/class/thermal/thermal_zone0/temp` → CPU-Temperatur
- `/proc/meminfo` → MemAvailable

**Output**:
- JSON nach `/tmp/moloch_preflight.json` mit Timestamp
- Human-readable Summary auf stdout:
```
[PREFLIGHT] 2026-03-28 14:30:00
  Service:    active          OK
  Git:        clean           OK
  FPS:        20.6            OK (min: 10)
  RAM:        855 MB          OK (max: 3500)
  CPU:        62.0 C          OK (max: 80)
  Frame Age:  0.3s            OK (max: 5)
  NPU:        reachable       OK
  Threads:    28              OK (max: 50)
```

**Exit-Code**: 0 = alles gruen, 1 = Service down oder Git dirty

**Abhaengigkeiten**: Nur stdlib (json, subprocess, pathlib, datetime)

**Geschaetzte LOC**: ~80

---

### 2. scripts/postflight.py — Post-Change Verification

**Zweck**: Aktuellen Zustand mit Preflight-Baseline vergleichen. Delta-Tabelle anzeigen. Audit ausfuehren.

**Input**: Keine Argumente. Liest `/tmp/moloch_preflight.json` automatisch.

**Ablauf**:
1. Preflight-JSON lesen (Fehler wenn nicht vorhanden = preflight wurde uebersprungen)
2. Aktuelle Werte erfassen (gleiche Quellen wie preflight)
3. Deltas berechnen: RAM_delta, FPS_delta, CPU_delta, Thread_delta
4. `python3 ~/moloch/moloch_audit.py --auto` ausfuehren und Exit-Code pruefen
5. Vergleichstabelle anzeigen

**Output**:
```
[POSTFLIGHT] 2026-03-28 14:45:00

Metrik      | Vorher | Nachher | Delta  | Status
------------|--------|---------|--------|-------
RAM (MB)    | 855    | 870     | +15    | OK
FPS         | 20.6   | 20.1    | -0.5   | OK
CPU (C)     | 62.0   | 63.2    | +1.2   | OK
Threads     | 28     | 28      | 0      | OK
Service     | active | active  | --     | OK
Audit       | --     | 40/40   | --     | PASS

Ergebnis: PASS
```

**Schwellwerte**:
- WARN: RAM +50MB, FPS -5, CPU +5C, Threads +5
- FAIL: RAM +200MB, FPS <10, Service inactive, Audit FAIL

**Exit-Code**: 0 = PASS, 1 = FAIL

**Geschaetzte LOC**: ~120

---

### 3. scripts/smoke_test.py — 8-Punkt Post-Reboot Checklist

**Zweck**: Schneller Sanity-Check nach `sudo reboot`. Ersetzt manuelle Pruefung.

**8 Checks**:

| # | Check | Methode | Schwelle |
|---|-------|---------|----------|
| 1 | Service laeuft | `systemctl is-active moloch` | == "active" |
| 2 | FPS OK | `/dev/shm/moloch_status.json` → fps | > 10 |
| 3 | RAM OK | `/proc/meminfo` → MemAvailable | > 500 MB frei |
| 4 | Frame frisch | Status-JSON → frame_age | < 5 Sekunden |
| 5 | NPU erreichbar | `hailortcli fw-control identify` (timeout=10) | Exit 0 |
| 6 | IPC funktioniert | `/dev/shm/moloch_frame` existiert + size > 0 | Datei vorhanden |
| 7 | Kein SEGV | `dmesg \| tail -60 \| grep -c SEGV` | == 0 |
| 8 | Audit PASS | `python3 moloch_audit.py --auto` | Exit 0 |

**Output**:
```
[SMOKE TEST] 2026-03-28 14:50:00

  1. Service:     active         PASS
  2. FPS:         20.6           PASS
  3. RAM:         855 MB used    PASS
  4. Frame Age:   0.3s           PASS
  5. NPU:         Hailo-10H OK   PASS
  6. IPC:         moloch_frame OK PASS
  7. SEGV:        0 gefunden     PASS
  8. Audit:       40/40 PASS     PASS

  Ergebnis: 8/8 PASS
```

**Log**: `~/moloch/logs/smoke_test_last.json`

**Exit-Code**: 0 = alle 8 PASS, 1 = mindestens 1 FAIL

**Geschaetzte LOC**: ~100

---

## P1 — Danach bauen (Praevention)

### 4. scripts/gst_lint.py — GStreamer Pipeline Validator

**Zweck**: GStreamer Pipeline-String validieren OHNE die Pipeline zu starten.
Fangt Typos, fehlende Properties, nicht-existente SO/HEF-Dateien ab.

**Input**: Kein Argument (liest automatisch aus tappas_pipeline.py) oder
`--string "rtspsrc ! queue ! ..."` fuer manuellen Test.

**Ablauf**:
1. Pipeline-String aus `tappas_pipeline.py::_build_pipeline_string()` extrahieren
   (Methode aufrufen ohne Pipeline zu starten, oder String aus Source parsen)
2. String an `!` splitten → Element-Liste
3. Fuer jedes Element:
   - `Gst.ElementFactory.find(name)` → existiert die Factory?
   - Fuer jedes `key=value`: Property auf der Factory pruefen
4. SO-Pfade pruefen: Alle `.so` Dateien die im String referenziert werden → `os.path.exists()`
5. HEF-Pfade pruefen: Alle `.hef` Dateien → `os.path.exists()`
6. Queue-Positionen pruefen: Warnung wenn zwei schwere Elemente ohne Queue dazwischen
7. Valve/Selector-Paare auf Konsistenz pruefen

**WICHTIG**: `Gst.parse_launch()` wird NICHT aufgerufen — das wuerde die Pipeline instanziieren.
Nur `Gst.ElementFactory.find()` fuer statische Validierung.

**Output**:
```
[GST LINT] tappas_pipeline.py

  Elements: 14 gefunden
  rtspsrc .................. OK
  rtph264depay ............. OK
  queue .................... OK
  avdec_h264 ............... OK
  hailonet (yolo) .......... OK
    hef-path: /mnt/moloch-data/hailo/models/yolov8m_h10.hef ... EXISTS
  hailofilter (postproc) ... OK
    so-path: /opt/hailo/tappas/... ... EXISTS
  ...

  SO-Dateien: 4/4 vorhanden
  HEF-Dateien: 3/3 vorhanden
  Queue-Gaps: 0 Warnungen

  Ergebnis: VALID
```

**Abhaengigkeiten**: `gi.repository.Gst` (bereits im System installiert)

**Geschaetzte LOC**: ~200

---

### 5. scripts/danger_check.py — Pre-Commit Hook

**Zweck**: Git Pre-Commit Hook der gefaehrliche Patterns VOR dem Commit abfaengt.

**Installation**:
```bash
ln -sf ~/moloch/scripts/danger_check.py ~/moloch/.git/hooks/pre-commit
chmod +x ~/moloch/scripts/danger_check.py
```

**Checks auf staged Files** (via `git diff --cached --name-only`):

| Check | Aktion | Schwere |
|-------|--------|---------|
| 2+ ROT-Dateien im selben Commit | BLOCK | FAIL |
| Runtime-State Dateien gestaged | BLOCK | FAIL |
| `subprocess.Popen` ohne `timeout` in neuen Zeilen | WARNUNG | WARN |
| `shell=True` in neuen Zeilen | WARNUNG | WARN |
| `pan_delta` Zeile in camera.py geaendert | WARNUNG | WARN |
| ArcFace Threshold-Konstante geaendert | WARNUNG | WARN |
| Mehr als 5 Dateien in einem Commit | WARNUNG | WARN |

**ROT-Dateien-Liste** (eingebettet im Script, gleich wie DANGER_MAP.md):
```python
ROT_FILES = {
    "core/moloch_service.py",
    "core/perception/tappas_pipeline.py",
    "core/hardware/camera.py",
    "core/hardware/hailo_manager.py",
    "core/core_integrator.py",
    "core/voice_pipeline.py",
    "core/mpo/autonomous_tracker.py",
    "core/gui/moloch_unified_panel.py",
    "core/speech/audio_pipeline.py",
    "core/inference_engine.py",
    "core/camera_manager.py",
    "core/model_orchestrator.py",
    "core/perception_engine.py",
    "core/ipc_router.py",
    "core/hardware/thermal_manager.py",
    "config/settings.json",
}

RUNTIME_STATE_FILES = {
    "config/last_face_position.json",
    "config/learned_patrol_positions.json",
    "config/kontext.json",
}
```

**Output bei FAIL**:
```
[DANGER CHECK] Pre-Commit Hook

  FAIL: 2 ROT-Dateien im selben Commit:
    - core/moloch_service.py
    - core/perception/tappas_pipeline.py
  Regel: Maximal 1 ROT-Datei pro Commit.

  WARN: subprocess.Popen ohne timeout in:
    - core/audio/wifi_mic.py:45

  Commit BLOCKIERT. Bitte aufteilen.
```

**Exit-Code**: 0 = Commit erlaubt, 1 = Commit blockiert

**Abhaengigkeiten**: Nur stdlib (subprocess fuer git-Befehle, re fuer Pattern-Matching)

**Geschaetzte LOC**: ~150

---

## P2 — Wenn Zeit (Monitoring)

### 6. scripts/baseline_capture.py — Periodischer Metrik-Snapshot

**Zweck**: Alle 5 Minuten via Cron: Systemmetriken in JSONL-Datei schreiben fuer Trendanalyse.

**Cron-Eintrag**:
```
*/5 * * * * python3 ~/moloch/scripts/baseline_capture.py
```

**Output**: Eine JSON-Zeile pro Aufruf in `~/moloch/logs/baseline_history.jsonl`:
```json
{"ts": "2026-03-28T14:30:00", "fps": 20.6, "ram_mb": 855, "cpu_c": 62.0, "threads": 28, "npu": "ok", "uptime_s": 3600}
```

**Optional**: `--plot` Flag fuer ASCII-Chart der letzten 24h.

**Geschaetzte LOC**: ~80

---

### 7. scripts/valve_test.py — TAPPAS Valve Isolation

**Zweck**: Einzelne GStreamer-Valves (SCRFD, Pose, ReID) isoliert oeffnen/schliessen
und FPS-Impact messen. Ohne Pipeline-Restart.

**Ablauf**:
1. Fuer jede Valve (scrfd_valve, pose_valve, reid_valve):
   - `drop=True` setzen (via IPC-Kommando)
   - 3 Sekunden warten
   - FPS messen (aus Status-JSON)
   - `drop=False` setzen
   - 3 Sekunden warten
   - FPS messen
   - Delta berechnen
2. Report: Welche Valves sicher gleichzeitig aktiv sein koennen

**Voraussetzung**: Service-seitiger IPC-Handler fuer Valve-Steuerung.
Teilweise vorhanden in `tappas_pipeline.py:259-293`.

**Geschaetzte LOC**: ~120

---

### 8. scripts/config_guard.py — Runtime-State Migration

**Zweck**: Einmalige Migration + laufende Ueberwachung. Trennt echte Config
von Runtime-State.

**Einmalige Migration**:
1. `config/last_face_position.json` → `/dev/shm/last_face_position.json`
2. `config/learned_patrol_positions.json` → `/dev/shm/learned_patrol_positions.json`
3. Pfad-Konstanten in `autonomous_tracker.py` (Zeile 42-43) anpassen
4. `.gitignore` erweitern:
   ```
   config/last_face_position.json
   config/learned_patrol_positions.json
   config/kontext.json
   ```
5. tmpfiles.d Config fuer /dev/shm Init nach Reboot

**Laufende Ueberwachung** (als Pre-Commit Check oder standalone):
- Pruefen ob Runtime-State-Dateien im Git-Index sind
- Warnung wenn ja

**Dateien die ECHTE Config sind** (gehoeren in Git):
- `config/settings.json`
- `config/keywords.json`
- `config/display_labels.json`
- `config/ptz_limits.json`
- `config/perception.json`
- `config/api_keys.json` (mit .gitignore fuer Credentials!)

**Dateien die Runtime-State sind** (gehoeren NICHT in Git):
- `config/last_face_position.json` — geschrieben bei jedem Tracking-Zyklus
- `config/learned_patrol_positions.json` — geschrieben waehrend Patrol-Lernen
- `config/kontext.json` — geschrieben von moloch_sprache.py

**Geschaetzte LOC**: ~150

---

## Abhaengigkeiten zwischen Scripts

```
preflight.py ──────── schreibt ──────→ /tmp/moloch_preflight.json
                                              |
postflight.py ─────── liest ─────────────────-+
       |
       +───── ruft auf ──→ moloch_audit.py --auto

smoke_test.py ─────── ruft auf ──→ moloch_audit.py --auto
                                   hailortcli fw-control identify

gst_lint.py ────────── unabhaengig (liest tappas_pipeline.py Source)

danger_check.py ────── unabhaengig (Git Hook, liest staged files)

baseline_capture.py ── unabhaengig (Cron, schreibt JSONL)

valve_test.py ──────── braucht ──→ Service-seitigen IPC-Handler

config_guard.py ────── aendert ──→ autonomous_tracker.py, .gitignore
```

---

## Reihenfolge der Implementierung

1. **preflight.py** (Grundlage fuer postflight)
2. **smoke_test.py** (sofort nutzbar nach Reboot)
3. **postflight.py** (braucht preflight als Basis)
4. **danger_check.py** (sofort als Git Hook installierbar)
5. **gst_lint.py** (braucht Gst-Import, komplexer)
6. **config_guard.py** (einmalige Migration)
7. **baseline_capture.py** (Cron-Setup)
8. **valve_test.py** (braucht Service-Erweiterung)
