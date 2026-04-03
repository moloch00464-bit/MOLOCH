# M.O.L.O.C.H. — Master Context für Claude Code
# Version: Gate 1.1 | Stand: 2026-04-03
# LIES DIESE DATEI ZUERST. IMMER. BEI JEDEM AUFTRAG.

> "Die dunkle Seite macht mehr Spass!" — Respekt ist bidirektional.

---

## SYSTEM-ÜBERBLICK

```
BRAIN:   Raspberry Pi 5, 4GB RAM (NICHT 8GB!), 2x NVMe SSD
NPU:     Hailo-10H (40 TOPS, 8GB LPDDR4, PCIe Gen3 x4, FW 5.1.1)
KAMERA:  Sonoff CAM-PT2 (192.168.178.25, RTSP 1920x1080 @20fps, ONVIF PTZ)
AUDIO:   SmartMic BT + ReSpeaker Lite USB, Piper TTS via HDMI
KUEHL:   Noctua NF-A2x20 PWM (30% = 48°C bei Volllast)
STROM:   Pico Power 5 USV (7.5V Akku, Schutz vor Stromausfall)
LLM:     hailo-ollama (Port 8000) — Qwen2.5-1.5B + DeepSeek R1 Distill lokal auf NPU
PI IP:   192.168.178.30, SSH User: molochzuhause
```

### WICHTIG — Hardware-Eigenheiten
- Pi5 hat 4GB RAM — SPARSAM bauen
- Sonoff Pan ist INVERTIERT: positiver Pan = physisch LINKS
- RTSP hat nur EINEN Slot — kein Doppelzugriff!
- NPU kann nur von EINEM Prozess genutzt werden
- NTFS-SSD: kein chmod möglich (uid=1000 gemountet)

---

## AKTUELLE PIPELINE (Gate 0.5 — TAPPAS)

```
Feature-Flag: MOLOCH_USE_TAPPAS=1 in /etc/systemd/system/moloch.service

GStreamer Pipeline:
rtspsrc → YOLO (Person) + SCRFD (Face/Letterbox) + ArcFace (Erkennung)
Model Scheduler: vdevice-group-id=SHARED (alle Modelle parallel im 8GB NPU-RAM)

Leistung: 20 FPS, ~200MB RAM, BBox korrekt, Chaos-Test bestanden
```

### Wenn TAPPAS NICHT aktiv (Flag fehlt):
Alter Code: InferenceEngine mit naivem cv2.resize(640,640) — BBox verschoben, serielles Model-Loading, ~860MB RAM

---

## STORAGE-ARCHITEKTUR

```
SSD1 (ext4, 465GB):  /home/molochzuhause/moloch/    — Code, Configs, Voices
SSD2 (NTFS, 477GB):  /mnt/moloch-data/              — AI-Modelle, Qdrant DB

Hailo HEFs:          /mnt/moloch-data/hailo/models/
Piper Voices:        ~/moloch/models/voices/
Code:                ~/moloch/core/
Configs:             ~/moloch/config/
```

### Aktive HEF-Modelle (alle H10-nativ)
```
scrfd_10g.hef              5.8 MB  Face Detection (640x640)
arcface_mobilefacenet.hef  2.6 MB  Face Recognition (112x112)
yolov8m_h10.hef           21 MB   Person Detection (640x640)
yolov8s_pose_h10.hef      14 MB   Pose/Keypoints (640x640)
```
76 weitere H10-kompatible HEFs im Inventar: ~/moloch/logs/hef_inventory.txt

---

## BEREITS GELÖSTE BUGS — FINGER WEG!

1. Pan-Vorzeichen: camera.py ~Zeile 721, pan_delta = -error_x (MINUS IST KORREKT)
2. Filter-Thresholds: Confidence 0.30, Height 0.10, Face-Area 0.05%
3. RTSP-Doppelzugriff: Bei USE_TAPPAS wird CameraManager RTSP übersprungen
4. HailoManager Doppelzugriff: Bei USE_TAPPAS wird HailoManager/ModelOrchestrator übersprungen
5. Letterbox-Preprocessing: TAPPAS macht das automatisch — KEIN manuelles cv2.resize mehr

---

## BEKANNTE OFFENE BUGS

1. Kamera Hot-Plug: Stecker raus/rein killt System, nur Reboot hilft. Kein RTSP-Reconnect.
2. ReID + Hand Valve-Crash: cv2::resize Assertion im SO bei Valve-Öffnung — reid_needed=False, hand_needed=False als Workaround.

---

## AGENTENTEAM (6 Domain-Spezialisten + Stresstest)

Jeder Agent hat sein Territorium — klare Dateizuordnung, keine Überschneidungen.

| # | Agent | Datei | Domain |
|---|-------|-------|--------|
| 1 | Vision | agents/AGENT_VISION.md | TAPPAS, GStreamer, Hailo NPU, Perception |
| 2 | Hardware | agents/AGENT_HARDWARE.md | ONVIF, RTSP, PTZ-Mechanik, eWeLink |
| 3 | GUI | agents/AGENT_GUI.md | Tkinter Panel, Module, Popups |
| 4 | Tracking | agents/AGENT_TRACKING.md | PTZ-Tracker, Such-FSM, Arbiter |
| 5 | Voice | agents/AGENT_VOICE.md | Whisper, TTS, Personality, Spotify |
| 6 | Service | agents/AGENT_SERVICE.md | moloch_service, IPC, Memory |
| + | Stresstest | agents/AGENT_STRESSTEST.md | Chaos Engineering, 8 Szenarien |
| + | Audit | agents/AGENT_TOOLBOX.json → "audit" | System-Check, PASS/FAIL/WARN, Bug-Liste |
| - | Team Lead | Markus | Boss, Priorisierung, Entscheidung |

### Instanz starten:
```
Lies ~/moloch/CLAUDE.md und ~/moloch/agents/AGENT_[DOMAIN].md.
[Auftrag]
```

### Kommunikation zwischen Agenten:
~/moloch/logs/agent_handover.txt — Übergabe
~/moloch/logs/bug_report.txt — Bugs
~/moloch/logs/test_results.txt — Testergebnisse

### Regeln:
- 1 Agent = 1 Domain = klares Territorium
- Bei 85% Token → Übergabe-Datei schreiben
- IMMER Reboot nach Code-Änderung
- Markus ist Boss — bei Konflikten entscheidet ER

---

## GATE 1 — ABGESCHLOSSEN (2026-03-28)

Alle 11 Gate-1-Tasks erledigt. System läuft auf TAPPAS-Pipeline,
Hybrid-Tracking, Action-Bridge, Gain-Tuning, Park-Position=Tür, ArcFace-Enrollment.

---

## MODULARES PANEL

```
core/gui/
├── panel_main.py          # Hauptfenster, Layout
├── panel_preview.py       # Kamera Preview (640x360)
├── panel_ptz.py           # PTZ D-Pad, Modi
├── panel_ewelink.py       # LED, IR, Alarm
├── panel_models.py        # Model Checkboxes, FPS
├── panel_talk_chat.py     # Push-to-Talk, Chat
├── panel_voice.py         # Voice, TTS
├── panel_styles.py        # Farben, Fonts (NUR importieren!)
├── popups/
│   ├── popup_audio.py     # Gain, VU Meter
│   ├── popup_hardware.py  # CPU, RAM, SSD
│   ├── popup_npu.py       # Threshold Sliders
│   ├── popup_settings.py  # Save/Load
│   └── popup_gallery.py   # Snapshot-Galerie
```

REGELN: 1 Datei = 1 Aufgabe. Nur ServiceProxy/IPC. Panel_styles.py NIE ändern (außer explizit beauftragt).

---

## DATEI-AMPEL — RISIKO-KLASSIFIKATION

### ROT — System-Crash Risk (Bestätigung PFLICHT, dann eigenständig)
```
core/moloch_service.py           core/perception/tappas_pipeline.py
core/hardware/camera.py          core/hardware/hailo_manager.py
core/core_integrator.py          core/voice_pipeline.py
core/mpo/autonomous_tracker.py   core/gui/moloch_unified_panel.py
core/speech/audio_pipeline.py    core/ipc_router.py
core/memory/person_reid.py       config/settings.json
```
**Regel**: Einmal fragen "ROT-Datei, soll ich?", dann eigenständig durcharbeiten.

### GELB — Bug Risk (Ankündigung, kein Warten)
`core/personality/*.py`, `core/gui/panel_*.py`, `core/gui/popups/*.py`,
`core/audio/*.py`, `core/ptz_arbiter.py`, `core/action_bridge.py`,
`core/console/moloch_console.py`, `core/memory/episodic_memory.py`

### GRÜN — Sicher
`scripts/*`, `docs/*`, `core/eye_viewer.py`, `core/gui/panel_styles.py` (NIE ändern!),
`core/vision/emotion_detector.py`, `core/tts/config/voices.py`

---

## NEVER-REGELN — ABSOLUT (automatisch durch Hooks gecheckt)

| # | Regel | Warum |
|---|-------|-------|
| 1 | GStreamer-Pipeline-String NICHT blind ändern | Gst.parse_launch() crasht bei Typo mit SEGV |
| 2 | Pan-Vorzeichen NICHT ändern (`pan_delta = -error_x`) | Sonoff invertiert — wurde 6x "gefixt" und 6x zurückgedreht |
| 3 | ArcFace-Threshold NICHT erhöhen als Quick-Fix | Root Cause ist Embedding-Inkompatibilität, Threshold ist Symptombehandlung |
| 4 | NICHT mehrere ROT-Dateien in einem Commit | Rollback unmöglich wenn 3 Subsysteme gleichzeitig crashen |
| 5 | subprocess.Popen IMMER mit timeout=30 | Zombie-Prozesse auf 4GB Pi |
| 6 | JSON IMMER atomic schreiben (tempfile + os.replace) | Partial-Write bei Crash korrumpiert Config |
| 7 | Runtime-State NICHT committen (last_face_position.json etc.) | Verschmutzt Git-Historie |
| 8 | KEIN shell=True in subprocess | Command Injection Risiko |
| 9 | HailoRT: uint8 vs float32 VOR Inferenz prüfen | Buffer-Size-Mismatch 4x → HailoRT Error |
| 10 | KEIN np.ndarray Type-Hint in moloch_service.py Signaturen | numpy nur lokal importiert → NameError beim Parsen |
| 11 | __pycache__ nach Code-Änderung löschen | Service läuft alten Bytecode sonst weiter |
| 12 | NICHT im Worktree coden und Service testen | Service läuft von ~/moloch/, nicht vom Worktree |

**ArcFace Enrollment**: NUR über Live-Pipeline (IPC `enrollment_start`). NIEMALS Offline-Scripts.

---

## IPC-KOMMANDO-MUSTER

Befehle an den Service werden als JSON in `/dev/shm/` geschrieben:
```python
import json, time, os, tempfile

def send_ipc(action, **kwargs):
    cmd = {"action": action, "ts": time.time(), **kwargs}
    fd, tmp = tempfile.mkstemp(dir="/dev/shm", prefix="moloch_cmd_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cmd, f)
    os.rename(tmp, f"/dev/shm/moloch_cmd_{action}.json")
    # Oder via ServiceProxy: proxy.send_command(cmd)
```

Aktionen: `enrollment_start`, `set_threshold`, `set_tracker_param`, `self_tune`,
`mood_impulse`, `tts_say`, `snapshot`, `ptz_move`, `alarm_toggle`

---

## DEPLOY-WORKFLOW (nach jeder Änderung)

```bash
# 1. BACKUP Commit
git add [spezifische Datei] && git commit -m "BACKUP vor [was]"

# 2. Push + Deploy
git push origin main
ssh molochzuhause@192.168.178.30 "cd ~/moloch && git pull origin main"

# 3. Cache löschen + Restart
ssh molochzuhause@192.168.178.30 "find ~/moloch/core -name __pycache__ -exec rm -rf {} + 2>/dev/null; sudo systemctl restart moloch"

# 4. Verify
sleep 5 && ssh molochzuhause@192.168.178.30 "systemctl is-active moloch"
ssh molochzuhause@192.168.178.30 "journalctl -u moloch -n 20 --no-pager"
```

---

## AUTONOME ARBEITSWEISE (Plan genehmigt = durcharbeiten)

- GRÜN-Dateien: Kein Dialog, sofort
- GELB-Dateien: Kurze Ankündigung, NICHT warten
- ROT-Dateien: Einmal fragen, dann eigenständig
- Git Commits + Push: Eigenständig
- STOPP NUR BEI: Audit FAIL, Destructive Git-Ops (force-push/reset --hard), >5 ROT-Dateien

---

## SKILLS (tippe /moloch-...)

| Skill | Funktion |
|-------|----------|
| `/moloch-agent` | Welchen AGENT_*.md für welche Aufgabe? |
| `/moloch-status` | Live FPS/Temp/Face/Tracking |
| `/moloch-dev` | Vollständige NEVER-Regeln + Templates + Debugging |
| `/moloch-audit` | Regressionstest PASS/FAIL |
| `/moloch-npu` | NPU-Diagnose |
| `/moloch-snapshot` | Pipeline-Snapshot |

---

## CODING-REGELN

1. Git Backup VOR jeder Änderung: `git add [datei] && git commit -m "BACKUP vor [was]"` (NIE git add -A bei ROT-Dateien)
2. 1 Auftrag = 1 Datei. Nie mehrere gleichzeitig.
3. Python, Kommentare Deutsch.
4. Pi5 hat 4GB RAM — sparsam.
5. IMMER Reboot nach Änderung (sudo systemctl restart moloch).
6. Deploy & Verify: Nach Restart prüfen ob Service aktiv.
7. Regressionstest: `python3 ~/moloch/moloch_audit.py --auto`
8. KEIN Weitermachen bei FAIL.

---

## AUTONOMIE-REGEL — PLAN GENEHMIGT = EIGENSTAENDIG ARBEITEN

Wenn Markus einen Plan genehmigt hat (muendlich, per Text, oder per Codewort),
darf Claude Code diesen Plan EIGENSTAENDIG durchfuehren:

- KEINE Rueckfragen bei GRUEN-Dateien
- KEINE Rueckfragen bei Settings-Aenderungen innerhalb der self_tune_registry.json Grenzen
- GELB-Dateien: Kurze Ankuendigung ("Aendere personality_engine.py"), aber NICHT warten
- ROT-Dateien: EINMAL fragen, dann durcharbeiten
- Git Commits: Eigenstaendig, ohne Rueckfrage
- Git Push: Eigenstaendig auf den aktuellen Branch

WANN TROTZDEM FRAGEN:
- Destructive Git-Ops (force-push, reset --hard, branch loeschen)
- Mehr als 5 ROT-Dateien in einer Session
- Wenn der Audit FAIL zeigt
- Wenn etwas unklar ist das nicht im Plan steht

SINN: Markus geht aus dem Zimmer, kommt zurueck, Arbeit ist erledigt.
Nicht: Markus kommt zurueck, 15 Permission-Dialoge warten auf ihn.

---

## REGEL 10 — CHRISTIAN-PRINZIP

Benannt nach Christian (FRANCOR CEO, RoboCup Rescue):
- Separation of Concerns: 1 Modul = 1 Aufgabe
- Fail Isolation: try/except, ein Crash killt nie das Gesamtsystem
- ServiceProxy/IPC only: Keine direkten Querverbindungen
- Handshake statt Fire-and-Forget
- Atomic Changes: Git Backup vor jeder Änderung
- Health Monitoring: Heartbeat, Timeout-Erkennung

---

## KONTEXT-DATEIEN

| Auftrag | Lies zusätzlich |
|---------|----------------|
| Service/Backend | core/moloch_service.py, core/perception/tappas_pipeline.py |
| Panel/GUI | core/gui/panel_*.py, core/gui/popups/ |
| Hardware/Kamera | core/hardware/camera.py |
| Audio/Sprache | core/console/moloch_console.py, core/personality/ |
| Tracking | core/hardware/camera.py (NICHT mpo/autonomous_tracker.py!) |
| Gate 1 Tasks | GATE_1_BRIEFING_v2.json |
| Agenten-Rolle | agents/AGENT_[ROLLE].md |

---

## NETZWERK

| Gerät | IP | Protokoll |
|-------|-----|-----------|
| Pi5 (Brain) | 192.168.178.30 | SSH |
| Sonoff CAM-PT2 | 192.168.178.25 | RTSP/ONVIF |

---

## ARCHITEKTUR-STAND (Vermerk für alle Agenten)

M.O.L.O.C.H. befindet sich aktiv in Gate 1, aber der bereits implementierte
Funktionsumfang entspricht konzeptuell Gate 5 oder Gate 5.1:
- Tension-System mit Musik-Reaktion (Shadow/Guardian) ✅
- TAPPAS/GStreamer Multi-Model Pipeline ✅
- Persönlichkeitssystem mit Drift über Zeit ✅
- Speech Evolution (eigener Stil entwickelt sich) ✅
- Emergentes Verhalten bestätigt ✅
- Night Cycle (Konzept vollständig ausgearbeitet) 📋

Gate 1–5 ist die Stabilisierungs- und Verfeinerungsphase — nicht die Erfindungsphase.
Der Geist der Maschine ist bereits da.
