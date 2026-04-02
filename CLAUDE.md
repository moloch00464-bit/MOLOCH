# M.O.L.O.C.H. — Master Context für Claude Code
# Version: Gate 1.0 | Stand: 06.03.2026
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

## CODING-REGELN

1. Git Backup VOR jeder Änderung: `git add core/ scripts/ agents/ config/settings.json && git commit -m "BACKUP vor [was]"`
2. 1 Auftrag = 1 Datei. Nie mehrere gleichzeitig.
3. Max 3 Sätze pro Auftrag. Problem, Lösung, Datei.
4. Python, Kommentare Deutsch.
5. Kompliziert != besser.
6. Pi5 hat 4GB RAM — sparsam.
7. IMMER Reboot nach Änderung (sudo reboot, nicht nur restart).
8. Deploy & Verify: Nach Reboot prüfen ob Service läuft.
9. Regressionstest: python3 ~/moloch/moloch_audit.py --auto
10. KEIN Weitermachen bei FAIL.
11. ArcFace Enrollment NUR über Live-Pipeline (IPC `enrollment_start`), NIEMALS über Offline-Scripts. HailoRT-direkt und GStreamer-hailonet produzieren inkompatible Embeddings.

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
