---
name: moloch-dev
description: Entwicklungs-Skill fuer M.O.L.O.C.H. — Pre/Post-Flight Checks, Danger Zones, Code Templates, NEVER-DO Regeln
---

# M.O.L.O.C.H. Entwicklungs-Skill
# Wird bei jeder Claude Code Session geladen
# Stand: 2026-03-28

---

## 1. SESSION-START PROTOKOLL

Jede Session MUSS mit diesen Schritten beginnen:

1. `~/moloch/CLAUDE.md` lesen (Systemregeln)
2. `~/moloch/logs/agent_handoff.md` lesen (falls vorhanden — letzte Session)
3. Relevantes `~/moloch/agents/AGENT_*.md` lesen (je nach Domain)
4. `git status` pruefen — bei dirty tree STOPPEN und User fragen
5. `~/moloch/docs/DANGER_MAP.md` als Referenz laden

---

## 2. DATEI-RISIKO-KLASSIFIKATION

Vor dem Edit einer Datei: Risiko-Stufe bestimmen.

### ROT — System-Crash Risk (User-Bestaetigung PFLICHT)

```
core/moloch_service.py          # 2725 LOC, 28 Imports, zentraler Orchestrator
core/perception/tappas_pipeline.py  # 1876 LOC, GStreamer Gst.parse_launch
core/hardware/camera.py         # 1160 LOC, Pan-Inversion Zeile 732
core/hardware/hailo_manager.py  # 704 LOC, VDevice-Singleton
core/core_integrator.py         # 868 LOC, 8+ Consumer
core/voice_pipeline.py          # 2072 LOC, VDevice + Threading
core/mpo/autonomous_tracker.py  # 1964 LOC, 30+ Case-Branches
core/gui/moloch_unified_panel.py # 2504 LOC, 30+ tk.after()
core/speech/audio_pipeline.py   # 835 LOC, subprocess ohne timeout
core/inference_engine.py        # 1239 LOC, Legacy (deprecated)
core/camera_manager.py          # 1055 LOC, RTSP-Routing
core/model_orchestrator.py      # 711 LOC, HEF-Lifecycle
core/perception_engine.py       # 719 LOC, Stage-Machine
core/ipc_router.py              # 163 LOC, SHM Frame-Exchange
core/hardware/thermal_manager.py # 859 LOC, Fan/Thermal
core/ptz_tracker.py             # 160 LOC, CoreIntegrator-Ref
core/perception/model_scheduler.py # 152 LOC, kein Heartbeat
core/memory/episodic_memory.py  # 286 LOC, Qdrant ohne Retry
core/memory/person_reid.py      # 340 LOC, Threshold = Tracking
config/settings.json            # Kein Schema, Crash bei Malformed
core/perception/tappas_pipeline.py  # ROT: GStreamer + alle Worker-Starts
core/perception/super_res_worker.py # ROT: SHARED VDevice, NPU-Konflikt moeglich
core/perception/low_light_processor.py # ROT: SHARED VDevice, in jedem Frame
```

**Regel**: Bei ROT-Dateien IMMER fragen:
"Diese Datei ist ROT (System-Crash Risk). Soll ich fortfahren?"

### GELB — Bug Risk (Vorsicht, Post-Flight Pflicht)

Alle `core/personality/*.py`, `core/autonomy/*.py`, `core/awareness/*.py`,
`core/gui/popups/*.py`, `core/gui/panel_*.py` (ausser panel_styles.py),
`core/spotify_controller.py`, `core/speech/hailo_whisper.py`,
`core/console/moloch_console.py`, `core/audio/*.py`,
`core/vision/face_database.py`, `core/vision/identity_manager.py`,
`core/longterm_memory.py`, `core/ptz_arbiter.py`, `core/action_bridge.py`.

### GRUEN — Sicher editierbar

`core/gui/panel_styles.py`, `core/vision/emotion_detector.py`,
`core/vision/gesture_detector.py`, `core/vision/age_gender_detector.py`,
`core/net/internet_bridge.py`, `core/tts/config/voices.py`,
`core/eye_viewer.py`, `scripts/*`, `docs/*`.

---

## 3. NEVER-DO REGELN (8 Stueck)

Aus der Analyse von 1200 Commits, 377 davon BACKUP/Fix/Revert:

### NEVER 1: GStreamer-String blind aendern
**Datei**: `tappas_pipeline.py`, Methode `_build_pipeline_string()` (Zeile ~254)
**Warum**: Gst.parse_launch() crashed bei jedem Typo mit SEGV. Kein Syntax-Check.
**Stattdessen**: `scripts/gst_lint.py` nutzen (sobald gebaut), oder Element-Properties
einzeln via `pipeline.get_by_name("x").set_property("key", value)` aendern.

### NEVER 2: Pan-Vorzeichen aendern
**Datei**: `camera.py`, Zeile 732: `pan_delta = -error_x`
**Warum**: Das MINUS ist KORREKT. Sonoff Pan ist physisch invertiert.
Wurde 6x in der Git-Historie "gefixt" und jedes Mal wieder zurueckgedreht.
**Regel**: Diese Zeile ist TABU. Kein Edit. Keine Diskussion.

### NEVER 3: ArcFace-Threshold aendern
**Datei**: `config/settings.json` oder `face_database.py`
**Warum**: Root Cause ist GStreamer/HailoRT Embedding-Inkompatibilitaet (sim=0.000).
Threshold-Aenderung ist Symptombehandlung. Wurde 4x geaendert (0.50→0.60→0.55→0.70).
**Loesung**: Enrollment muss durch denselben GStreamer-Pfad laufen wie Live-Inference.

### NEVER 4: Mehrere ROT-Dateien in einem Commit
**Warum**: Git-Historie zeigt: Multi-File "Shotgun Surgery" Commits machen Rollback
unmoeglich. Wenn 3 Subsysteme gleichzeitig geaendert werden und es crashed,
weiss niemand welche Aenderung schuld war.
**Regel**: 1 Commit = 1 Datei (bei ROT-Dateien). Immer.

### NEVER 5: subprocess.Popen ohne timeout
**Warum**: Zombie-Prozesse. audio_pipeline.py (Zeile 158) und music_visualizer.py
(Zeile 228) haben dieses Problem bereits. Pi5 hat nur 4GB RAM.
**Pattern**: `subprocess.run([...], capture_output=True, timeout=30, text=True)`

### NEVER 6: JSON direkt schreiben
**Warum**: Partial-Write bei Crash/Stromausfall korrumpiert die Datei. Mehrere
Dateien im Codebase haben dieses Problem (calibration_engine.py:97, identity_manager.py:72).
**Pattern**: Immer atomic — tempfile schreiben, dann os.replace():
```python
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
with os.fdopen(fd, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
os.replace(tmp, path)  # Atomic auf ext4, Fallback fuer NTFS noetig
```

### NEVER 7: Runtime-State in Git committen
**Dateien**: `config/last_face_position.json`, `config/learned_patrol_positions.json`
**Warum**: Diese Dateien werden bei jedem Tracking-Zyklus geschrieben und verschmutzen
jeden BACKUP-Commit. Sie gehoeren nach `/dev/shm/` oder `/tmp/`.
**Regel**: Vor git add pruefen ob State-Dateien im Staging sind.

### NEVER 8: shell=True in subprocess
**Dateien**: `moloch_console.py:514,518`, `audio_manager.py:552`, `tts_manager.py:552`
**Warum**: Command Injection Risiko. Immer Liste statt String:
```python
# FALSCH: subprocess.run("cmd " + user_input, shell=True)
# RICHTIG: subprocess.run(["cmd", user_input])
```

---

## 4. PRE-FLIGHT CHECKS (VOR jeder Code-Aenderung)

```bash
# 1. Git muss clean sein
git status  # Keine uncommitted Changes

# 2. Syntax-Check des Zielmoduls
python3 -c "import core.[modul]; print('Syntax OK')"

# 3. Baseline erfassen (sobald preflight.py existiert)
python3 ~/moloch/scripts/preflight.py
# Alternativ manuell:
cat /dev/shm/moloch_status.json | python3 -m json.tool | grep -E "fps|ram|cpu"

# 4. Service laeuft
systemctl is-active moloch  # Muss "active" sein

# 5. Risiko-Stufe bestimmen (ROT/GELB/GRUEN)
# Bei ROT: User fragen

# 6. BACKUP Commit
git add -A && git commit -m "BACKUP vor [kurze Beschreibung]"
```

---

## 5. POST-FLIGHT CHECKS (NACH jeder Code-Aenderung)

```bash
# 1. Syntax-Check
python3 -c "import core.[modul]; print('Syntax OK')"

# 2. Service neu starten
sudo systemctl restart moloch
sleep 5

# 3. Service laeuft noch?
systemctl is-active moloch  # Muss "active" sein

# 4. Audit (40 Auto-Tests)
python3 ~/moloch/moloch_audit.py --auto

# 5. Baseline-Vergleich (sobald postflight.py existiert)
python3 ~/moloch/scripts/postflight.py
# Manuell: RAM ±50MB, FPS ±5, CPU ±5C akzeptabel

# 6. Bei FAIL:
# SOFORT revert: git checkout -- [datei]
# NICHT weiter-patchen!
# Root-Cause analysieren, dann neuer Versuch.
```

---

## 6. CODE-TEMPLATES

### Singleton Pattern (5+ Module nutzen dieses Pattern)

```python
import threading

_instance = None
_lock = threading.Lock()

def get_thing() -> "Thing":
    """Singleton-Zugriff. Thread-safe."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = Thing()
    return _instance
```

Genutzt von: `longterm_memory.py`, `hailo_whisper.py`, `thermal_manager.py`,
`einpraegen.py`, `power_monitor.py`.

### Safe JSON Write (Atomic + NTFS-Fallback)

```python
import tempfile, os, json

def safe_json_write(path: str, data) -> None:
    """Atomarer JSON-Write. NTFS-Fallback fuer SSD2."""
    dir_path = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # Atomic auf ext4
    except OSError:
        # NTFS-Fallback (kein atomic rename moeglich)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.unlink(tmp)
        except OSError:
            pass
```

Referenz-Implementierungen: `moloch_sprache.py:570-579`, `persistent_memory.py:87-106`.

### Subprocess mit Timeout

```python
import subprocess

def safe_run(cmd: list, timeout: int = 30) -> subprocess.CompletedProcess:
    """Subprocess mit Timeout. Kein shell=True!"""
    return subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        text=True
    )
```

### GStreamer Property-Aenderung (ohne Pipeline-Rebuild)

```python
# RICHTIG: Einzelne Property aendern
element = self._pipeline.get_by_name("scrfd_valve")
if element:
    element.set_property("drop", True)  # Valve schliessen

# FALSCH: Gesamte Pipeline neu bauen wegen einer Property
# self._build_pipeline_string()  # NICHT fuer Property-Aenderungen!
```

---

## 7. HANDOFF-PROTOKOLL (bei ~85% Kontext)

Bei hoher Kontext-Auslastung: Handoff-Datei schreiben BEVOR es zu spaet ist.

**Datei**: `~/moloch/logs/agent_handoff.md`

**Inhalt**:
```markdown
# Agent Handoff — [Datum] [Uhrzeit]

## Aktueller Task
[Was wird gerade gemacht]

## Erledigt
- [x] Punkt 1
- [x] Punkt 2

## Offen
- [ ] Punkt 3
- [ ] Punkt 4

## Geaenderte Dateien
- core/xxx.py (was geaendert)

## Service-Status
- moloch.service: active/inactive
- USE_TAPPAS: 1/0
- Letzte Audit-Ergebnisse: PASS/FAIL

## Blocker
- [Beschreibung falls vorhanden]

## Baseline-Metriken
- RAM: XXX MB
- FPS: XX.X
- CPU: XX.X C
```

---

## 8. DEBUGGING-LEITFADEN

### Service crashed nach Aenderung
1. `journalctl -u moloch -n 50` — Letzte 50 Log-Zeilen
2. `dmesg | tail -20` — Kernel-Meldungen (SEGV, OOM)
3. `python3 -c "import core.[modul]"` — Syntax-Fehler?
4. `git diff HEAD~1` — Was wurde geaendert?
5. `git checkout -- [datei]` — Revert zur letzten Version

### Pipeline startet nicht
1. `cat /dev/shm/moloch_status.json` — Status-JSON pruefen
2. `ps aux | grep gst` — GStreamer-Prozesse?
3. `hailortcli fw-control identify` — NPU erreichbar?
4. RTSP-Stream pruefen: `ffprobe rtsp://192.168.178.25:554/stream1`

### RAM steigt ueber 3GB
1. `cat /proc/meminfo | grep MemAvailable` — Aktueller Stand
2. `ps aux --sort=-rss | head -5` — Top RAM-Verbraucher
3. Service neustarten: `sudo systemctl restart moloch`
4. Wenn nach Restart wieder hoch: Memory-Leak in letzter Aenderung

### NPU Error 74 (OUT_OF_PHYSICAL_DEVICES)
1. **NICHT** zweites VDevice erstellen
2. Pruefen ob alter Prozess noch laeuft: `ps aux | grep hailo`
3. Service komplett stoppen: `sudo systemctl stop moloch`
4. NPU Reset: `sudo systemctl restart moloch`

---

## 9. HAILO-10H NPU REFERENZ (Stand Maerz 2026)

### Software-Versionen
- Installiert: HailoRT 5.1.1, TAPPAS 5.1.0
- Verfuegbar: HailoRT 5.2.0, Dataflow Compiler 5.2.0
- Model Zoo Master-Branch = NUR Hailo-10H/15H. Hailo-8 = v2.x Branch.
- Debian Trixie (13) fuer volle H10-Unterstuetzung

### NPU RAM Budget (8GB LPDDR4X)
```
Vision (YOLO+SCRFD+ArcFace+Pose): ~585 MB
Whisper Base:                      ~155 MB
Qwen2.5-1.5B + KV-Cache:         ~1750 MB
---
Belegt:                           ~2490 MB
Reserve:                          ~5510 MB
```
- KV-Cache waechst: ~256MB pro 1000 Tokens Kontext (max 2048 Tokens)
- LLM laeuft = Vision pausiert 5-8s (Time-Slicing noetig)

### Leistungsdaten
- Vision: 3.5-4.5W, 20+ FPS
- LLM: bis 8W Peak, 6-10 Tokens/Sekunde, First Token <1s
- Gesamt mit Pi5: ~20W unter Volllast → 27W Netzteil MUSS stabil sein
- PCIe: Pi5 hat nur Gen2 x1 (~500MB/s), H10 will Gen3 x4 = Mismatch

### Verfuegbare Modelle (noch nicht eingebaut)
- **SigLIP/SigLIP2**: Zero-Shot Image Classification ("Was sehe ich?" ohne Training)
- **Qwen2.5-VL-3B**: Vision-Language (Kamerabild → Textbeschreibung)
- **PaddleOCR v5**: Text im Bild erkennen (Schilder, Bildschirme)
- **YOLOv11/v12**: Neueste YOLO-Generationen
- **Stable Diffusion**: Bildgenerierung am Edge
- **DeepSeek-R1-Distill-Qwen-1.5B**: Reasoning
- **Qwen2.5-Coder-1.5B**: Code-Generierung

### hailo-ollama (fuer Gate 5.1)
- Ollama-kompatibel + OpenAI-kompatibel auf Port 8000
- Modelle: Qwen2.5-1.5B, Llama-3.2-1B/3B, DeepSeek-R1
- Max ~3B Parameter mit 8GB RAM
- Setup: hailo_model_zoo_genai → cmake build → hailo-ollama Binary

### NEVER (NPU-spezifisch)
- NEVER zweites VDevice erstellen (Error 74)
- NEVER LLM und Vision gleichzeitig ohne Time-Slicing
- NEVER Whisper Small verwenden (Bug: ignoriert language=de)
- NEVER mehr als ~2.5GB NPU-RAM fuer Vision+Speech reservieren wenn LLM geplant
