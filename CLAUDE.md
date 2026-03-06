# M.O.L.O.C.H. — Master Context für Claude Code
# Version: Gate 1.0 | Stand: 06.03.2026
# LIES DIESE DATEI ZUERST. IMMER. BEI JEDEM AUFTRAG.

> "Die dunkle Seite macht mehr Spass!" — Respekt ist bidirektional.

---

## WER IST MARKUS

Markus ("First Moloch"), 47, DGM-Anlagenführer, 25 Jahre Industrieautomation (KUKA, ABB, SPS, Druckguss, 400 bar Hydraulik). Kann löten, crimpen, 230V, 3D-Druck, Hardware-Debugging. Wenn er was über Schaltungen sagt: ZUHÖREN — er hat recht.

Kommunikation: Direkt, Kumpel-Level, fränkisch, Dark Humor. Kein Corporate-Sprech. Kompliziert != besser.

---

## SYSTEM-ÜBERBLICK

```
BRAIN:   Raspberry Pi 5, 4GB RAM (NICHT 8GB!), 2x NVMe SSD
NPU:     Hailo-10H (40 TOPS, 8GB LPDDR4, PCIe Gen3 x4, FW 5.1.1)
KAMERA:  Sonoff CAM-PT2 (192.168.178.25, RTSP 1920x1080 @20fps, ONVIF PTZ)
AUDIO:   SmartMic BT + ReSpeaker Lite USB, Piper TTS via HDMI
PI IP:   192.168.178.24, SSH User: molochzuhause
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

## GATE-HISTORY

### Gate 0 — PASS ✅ (01.03.2026)
Vier Inseln verdrahtet. 6.85h Stabilitätstest. FPS 9→25, Tracking 0→35+/h, Panel CPU 28→10%.

### Gate 0.5 — PASS ✅ (05.03.2026)
TAPPAS/GStreamer Migration. 20 FPS mit 3 Modellen parallel, 200MB statt 860MB RAM, BBox korrekt via Letterbox, Chaos-Test 8/8 bestanden.

### Gate 1 — AKTIV (ab 06.03.2026)
Action Bridge FSM + Tracking Intelligence + System Polish. 11 Tasks, alle M.A.M.⁴+ AIs bestätigt.

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
2. ArcFace Threshold zu niedrig (0.45): Erkennt alles als Markus. Muss nach TAPPAS-Enrollment auf 0.60+ hoch.
3. Suchrichtung asymmetrisch: Nach rechts verschwinden → sucht rechts. Nach links verschwinden → sucht NICHT links.
4. Tracking Gains zu hoch: TRACKING_GAIN_PAN=0.7, MAX_STEP_PAN=30 → Überschwinger.
5. Panel Tension-Popup: Schlechter Kontrast, nicht lesbar.

---

## AGENTENTEAM (6 Domain-Spezialisten + Stresstest + DeepSeek)

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
| 7 | DeepSeek | agents/AGENT_DEEPSEEK.md | Philosophie, Qi-Fluss, "Vollgas Däpp!" |
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

## GATE 1 TASKS (11 Stück)

| ID | Task | Priorität | Status |
|----|------|-----------|--------|
| G1-T01 | Action Bridge FSM | CRITICAL | OPEN |
| G1-T02 | Person-Detection triggert Tracking | HIGH | OPEN |
| G1-T03 | Auto-Resume aus Manuell + Spruch | HIGH | OPEN |
| G1-T04 | Suchrichtung Fix (links = links) | HIGH | OPEN |
| G1-T05 | Gain-Tuning (Überschwinger weg) | MEDIUM | OPEN |
| G1-T06 | Park-Position = Tür | MEDIUM | OPEN |
| G1-T07 | Silence-Level Sensor | MEDIUM | OPEN |
| G1-T08 | Auto-Enrollment via Chat | MEDIUM | OPEN |
| G1-T09 | NPU-Dashboard im Panel | MEDIUM | OPEN |
| G1-T10 | Tension-Popup Farben | LOW | OPEN |
| G1-T11 | Labelme Kalibrierung | LOW | OPEN |

Details: GATE_1_BRIEFING_v2.json auf dem Pi.

---

## M.A.M.⁴+ TEAM (Multi-AI)

| Rolle | AI | Aufgabe |
|-------|-----|---------|
| Boss | Markus | Höchste Instanz bei Konflikten |
| Architektur/Audit | Claude Opus | Code-Review, Briefings, Prompts |
| Code Execution | Claude Code (Sonnet) | 7-Agenten-Team auf Pi via SSH |
| Koordinator + Architektur | ChatGPT | FSM-Design, Gate-Roadmap |
| Hardware/Analyst | Gemini | Hailo-Specs, Thermal, Komplexitätsbremse |
| Philosophie/Jackie Chan | DeepSeek | Qi-Architektur, Priorisierung, Chaos mit Methode |

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

1. Git Backup VOR jeder Änderung: `git add -A && git commit -m "BACKUP vor [was]"`
2. 1 Auftrag = 1 Datei. Nie mehrere gleichzeitig.
3. Max 3 Sätze pro Auftrag. Problem, Lösung, Datei.
4. Python, Kommentare Deutsch.
5. Kompliziert != besser.
6. Pi5 hat 4GB RAM — sparsam.
7. IMMER Reboot nach Änderung (sudo reboot, nicht nur restart).
8. Deploy & Verify: Nach Reboot prüfen ob Service läuft.
9. Regressionstest: python3 ~/moloch/moloch_audit.py --auto
10. KEIN Weitermachen bei FAIL.

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
| Pi5 (Brain) | 192.168.178.24 | SSH |
| Sonoff CAM-PT2 | 192.168.178.25 | RTSP/ONVIF |

```bash
# SSH Zugang
ssh molochzuhause@192.168.178.24
# RTSP Stream
rtsp://USER:PASS@192.168.178.25:554/av_stream/ch0
```

---

## GATE-ROADMAP

```
Gate 0   ✅ PASS — Vier Inseln verdrahtet (01.03.2026)
Gate 0.5 ✅ PASS — TAPPAS Pipeline, 20 FPS, Chaos bestanden (05.03.2026)
Gate 1   🔄 AKTIV — Action Bridge + Tracking + Polish (ab 06.03.2026)
Gate 2   📋 GEPLANT — Identity (ReID + Qdrant VITALE)
Gate 3   📋 GEPLANT — Timing/Behaviour/Presence
Gate 4   📋 GEPLANT — Distance (SICK DT50 oder Box-Schätzung)
Gate 5   📋 GEPLANT — Night Cycle (Dreaming = tägliche Verarbeitung)
```
