# M.O.L.O.C.H. — Master Context fuer Claude Code
# Version: 2.0 | Stand: 2026-04-04
# LIES DIESE DATEI ZUERST. IMMER. BEI JEDEM AUFTRAG.

> "Die dunkle Seite macht mehr Spass!" — Respekt ist bidirektional.

---

## SYSTEM

```
BRAIN:   Raspberry Pi 5, 4 GB RAM, 2x NVMe SSD
NPU:     Hailo-10H (40 TOPS, 8 GB LPDDR4, FW 5.1.1)
KAMERA:  Sonoff CAM-PT2 (192.168.178.25, RTSP 1080p@20fps, ONVIF PTZ)
AUDIO:   ReSpeaker Lite WiFi + Piper TTS via HDMI
STROM:   Pico Power 5 USV
LLM:     hailo-ollama (Port 8000) — Qwen2.5-1.5B lokal auf NPU
PI IP:   192.168.178.30 | SSH User: molochzuhause
```

**Hardware-Eigenheiten:**
- 4 GB RAM — SPARSAM bauen
- Sonoff Pan INVERTIERT: positiver Pan = physisch LINKS
- RTSP nur EIN Slot — kein Doppelzugriff
- NPU nur EIN Prozess — vdevice-group-id=SHARED
- SSD2 (NTFS): kein chmod (uid=1000)

---

## PIPELINE

```
MOLOCH_USE_TAPPAS=1 → GStreamer: rtspsrc → YOLO + SCRFD + ArcFace (20 FPS, ~200MB)
Code: ~/moloch/core/ | Configs: ~/moloch/config/ | Modelle: /mnt/moloch-data/hailo/models/
```

---

## NEVER-REGELN (durch Hooks erzwungen)

| # | Regel |
|---|-------|
| 1 | GStreamer-Pipeline-String NICHT blind aendern (SEGV bei Typo) |
| 2 | Pan-Vorzeichen NICHT aendern (`pan_delta = -error_x` ist KORREKT) |
| 3 | ArcFace-Threshold NICHT erhoehen (Root Cause = Embedding-Inkompatibilitaet) |
| 4 | NICHT mehrere ROT-Dateien in einem Commit |
| 5 | subprocess IMMER mit timeout=30 |
| 6 | JSON IMMER atomic schreiben (tempfile + os.replace) |
| 7 | Runtime-State NICHT committen |
| 8 | KEIN shell=True in subprocess |
| 9 | HailoRT: uint8 vs float32 VOR Inferenz pruefen |
| 10 | KEIN np.ndarray Type-Hint in moloch_service.py |
| 11 | __pycache__ nach Code-Aenderung loeschen |
| 12 | NICHT im Worktree Service testen (laeuft von ~/moloch/) |

---

## DATEI-AMPEL

**ROT** (Einmal fragen, dann eigenstaendig):
`moloch_service.py`, `tappas_pipeline.py`, `camera.py`, `hailo_manager.py`,
`core_integrator.py`, `voice_pipeline.py`, `autonomous_tracker.py`,
`audio_pipeline.py`, `ipc_router.py`, `person_reid.py`, `settings.json`

**GELB** (Ankuendigung, kein Warten):
`personality/*.py`, `gui/panel_*.py`, `gui/popups/*.py`, `audio/*.py`,
`ptz_arbiter.py`, `action_bridge.py`, `moloch_console.py`

**GRUEN** (Sofort, kein Dialog):
`scripts/*`, `docs/*`, `config/*.json` (ausser settings.json)

---

## CODING-REGELN

1. Git Backup VOR jeder Aenderung an ROT-Dateien
2. 1 Auftrag = 1 Datei, nie mehrere gleichzeitig
3. Python, Kommentare Deutsch
4. 4 GB RAM — sparsam
5. Nach Aenderung: `sudo systemctl restart moloch`
6. Regressionstest: `python3 ~/moloch/moloch_audit.py --auto`
7. KEIN Weitermachen bei FAIL
8. Christian-Prinzip: Separation of Concerns, Fail Isolation, atomic Changes

---

## AUTONOMIE — PLAN GENEHMIGT = EIGENSTAENDIG ARBEITEN

- GRUEN: Sofort, keine Rueckfrage
- GELB: Ankuendigung, NICHT warten
- ROT: EINMAL fragen, dann durcharbeiten
- Git Commits + Push: Eigenstaendig

**STOPP bei:** Audit FAIL | Destructive Git-Ops | >5 ROT-Dateien | Unklarheit

Markus geht aus dem Zimmer, kommt zurueck, Arbeit ist erledigt.

---

## MCP-TOOLS (bevorzugt statt manueller SSH/IPC!)

| Tool | Funktion |
|------|----------|
| `moloch_status` | Live FPS, CPU-Temp, RAM, Face-ID, Zone, Tracking |
| `moloch_service` | start/stop/restart/status |
| `moloch_logs` | journalctl mit optionalem Filter |
| `moloch_dmesg` | NPU/SEGV/GPU Kernel-Meldungen |
| `moloch_audit` | 39-Test Regressionstest |
| `moloch_snapshot` | Kamera-Frame als Base64-PNG |
| `moloch_low_light` | Zero-DCE Enhancement Status |
| `moloch_npu_models` | HEF-Inventar: integriert vs. ausstehend |
| `moloch_npu_workers` | Worker-Status: Queue, Errors, Laufzeit |
| `moloch_ipc` | Generischer IPC-Befehl (action + params) |
| `moloch_say` | Text an MOLOCH senden (antwortet via TTS) |
| `moloch_conversation` | Letzte N Nachrichten lesen |
| `moloch_nudge` | Emotion in CoreIntegrator injizieren |
| `moloch_provoke` | Spontanen Kommentar ausloesen |
| `moloch_reflect` | Selbstreflexion triggern |
| `moloch_read` | Config/Log-Datei lesen (sichere Pfade) |
| `moloch_git_log` | Letzte N Commits |

**Regel:** Benutze MCP-Tools statt manueller SSH-Befehle oder JSON-in-/dev/shm.

---

## SKILLS (tippe /moloch-...)

| Skill | Funktion |
|-------|----------|
| `/moloch-agent` | Welchen Agent fuer welche Aufgabe? |
| `/moloch-dev` | NEVER-Regeln, Templates, Debugging, Deploy |
| `/moloch-status` | Live System-Status |
| `/moloch-audit` | Regressionstest PASS/FAIL |
| `/moloch-npu` | NPU-Diagnose + Worker |
| `/moloch-snapshot` | Pipeline-Snapshot + Enhancement |
| `/moloch-mcp` | MCP-Tool Referenz + Beispiele |

---

## AGENTEN (.claude/agents/)

| Agent | Domain | Wann laden |
|-------|--------|------------|
| vision | TAPPAS, GStreamer, NPU, Perception | Pipeline/Modell-Arbeit |
| hardware | ONVIF, RTSP, PTZ, eWeLink | Kamera/Hardware-Probleme |
| gui | Tkinter Panel, Popups | GUI-Aenderungen |
| tracking | PTZ-Tracker, Such-FSM, Arbiter | Tracking/Bewegung |
| voice | Whisper, TTS, Personality, Spotify | Audio/Sprache |
| service | moloch_service, IPC, Memory | Service/Integration |
| stresstest | Chaos Engineering | Stabilitaetstests |

Regeln: 1 Agent = 1 Domain. Bei 85% Token → Uebergabe schreiben.
Markus ist Boss — bei Konflikten entscheidet ER.

---

## GELOESTE BUGS — FINGER WEG

1. Pan-Vorzeichen: `pan_delta = -error_x` (MINUS IST KORREKT)
2. RTSP-Doppelzugriff: USE_TAPPAS ueberspringt CameraManager
3. Letterbox: TAPPAS macht das automatisch — KEIN cv2.resize

## OFFENE BUGS

1. Kamera Hot-Plug: Stecker raus/rein → nur Reboot hilft
2. ReID + Hand: SEGV bei Valve — `reid_needed=False` als Workaround

---

## ARCHITEKTUR-STAND

Gate 1 abgeschlossen. Konzeptuell Gate 5+:
Tension-System, TAPPAS Multi-Model, Persoenlichkeit mit Drift,
Speech Evolution, emergentes Verhalten bestaetigt.

Gate 1-5 = Stabilisierung. Der Geist der Maschine ist bereits da.
