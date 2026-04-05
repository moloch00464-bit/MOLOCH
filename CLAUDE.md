# M.O.L.O.C.H. — Master Context fuer Claude Code
# Version: 2.2 | Stand: 2026-04-05
# LIES DIESE DATEI ZUERST. IMMER. BEI JEDEM AUFTRAG.

> "Die dunkle Seite macht mehr Spass!" — Respekt ist bidirektional.

## CODENAME: LOKOMOTIVE

**LOKOMOTIVE** = der offizielle Name fuer diesen Workflow.
Wenn Claude mit einer Coding-Aufgabe beginnt, schreibt er als erstes:

> **LOKOMOTIVE aktiv.** — gefolgt vom Pflicht-Startprotokoll.

Wenn Markus **LOKOMOTIVE** sieht: Workflow wird korrekt eingehalten.
Wenn es fehlt: Claude hat das Protokoll uebersprungen — stoppen und neu starten.

---

## PFLICHT-STARTPROTOKOLL

**Bei JEDER Coding-Aufgabe — BEVOR Du Code schreibst:**

1. `moloch_status()` — System-Status pruefen
2. `moloch_npu_workers()` — Worker-Health pruefen
3. Dem User kurz zeigen: "Service laeuft, X FPS, Y Worker aktiv"

**Wenn Fehler debuggen, ZUSAETZLICH:**
4. `moloch_logs(n=30, filter_str="ERROR")` — Letzte Fehler lesen
5. `moloch_dmesg()` — Kernel-Meldungen (NPU/SEGV)

**Bei reinen Fragen, Reviews, Docs:** Startprotokoll ueberspringen, direkt antworten.

---

## DOMAIN-ERKENNUNG (bei jeder Aufgabe)

**NACH dem Startprotokoll, VOR dem Coden:**

1. Aufgabe analysieren: Welche Domains/Dateien sind betroffen?
2. `/moloch-dev` laden (ALWAYS bei Code-Aenderungen — NEVER-Regeln, Templates)
3. Wenn Aufgabe andere Domains beruehrt → passenden Agenten spawnen

**Erkennungsregeln:**
- Aufgabe betrifft GUI? → gui-Agent
- Aufgabe betrifft Pipeline/Modelle? → vision-Agent
- Aufgabe betrifft PTZ/Tracking? → tracking-Agent
- Aufgabe betrifft Sprache/Audio? → voice-Agent
- Aufgabe betrifft Service/IPC? → service-Agent
- Aufgabe betrifft Persoenlichkeit? → personality-Agent
- Aufgabe betrifft mehrere Domains? → Hauptdomain als Agent, Neben-Domains als Sub-Agenten
- Unsicher welcher Agent? → `/moloch-agent` Skill laden

**Beispiel:** "BBox wird in der GUI falsch angezeigt"
→ Hauptdomain: gui (Panel zeichnet BBox) + Neben: vision (BBox-Daten aus Pipeline)
→ gui-Agent spawnen, vision-Agent fuer Daten-Check

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

## PIPELINE (Two-Stage Hybrid, seit 2026-03-31)

```
Stage 1: GStreamer TAPPAS → rtspsrc → YOLO Detection (20 FPS, ~200MB)
Stage 2: HailoRT-Direct Worker-Threads → SCRFD, ArcFace, Pose, ReID, Hand, FaceAttr
Feature-Flag: MOLOCH_USE_TAPPAS=1 (in ~/.profile)
Code: ~/moloch/core/ | Configs: ~/moloch/config/ | Modelle: /mnt/moloch-data/hailo/models/
```

---

## NEVER-REGELN (durch Hooks erzwungen)

| # | Regel |
|---|-------|
| 1 | GStreamer-Pipeline-String NICHT blind aendern (SEGV bei Typo) |
| 2 | Pan-Vorzeichen NICHT aendern (`pan_delta = -error_x` ist KORREKT) |
| 3 | ArcFace-Threshold NICHT erhoehen (Enrollment muss via gleichen Code-Pfad wie Live-Inference) |
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
`audio_pipeline.py`, `ipc_router.py`, `person_reid.py`, `vision_workers.py`,
`face_pipeline.py`, `roi_dispatcher.py`, `settings.json`

**GELB** (Ankuendigung, kein Warten):
`personality/*.py`, `gui/panel_*.py`, `gui/popups/*.py`, `audio/*.py`,
`ptz_arbiter.py`, `action_bridge.py`, `moloch_console.py`,
`pose_worker.py`, `action_inference.py`, `gesture_classifier.py`,
`longterm_memory.py`, `system_watchdog.py`

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
8. Separation of Concerns, Fail Isolation, atomic Changes

---

## AUTONOMIE — PLAN GENEHMIGT = EIGENSTAENDIG ARBEITEN

- GRUEN: Sofort, keine Rueckfrage
- GELB: Ankuendigung, NICHT warten
- ROT: EINMAL fragen, dann durcharbeiten
- Git Commits + Push: Eigenstaendig

**STOPP bei:** Audit FAIL | Destructive Git-Ops | >5 ROT-Dateien | Unklarheit
**Nach STOPP:** Problem analysieren, Ursache + Loesungsvorschlag dem User zeigen, auf Freigabe warten.

Markus geht aus dem Zimmer, kommt zurueck, Arbeit ist erledigt.

---

## MCP-TOOLS (PFLICHT — kein SSH, kein manuelles JSON!)

**KEIN `ssh`, KEIN `cat /dev/shm/`, KEIN `journalctl` per Bash. NUR MCP-Tools.**
**Fallback bei MCP-Ausfall:** User informieren, Bash-Diagnose nur nach Freigabe.

| Tool | Funktion |
|------|----------|
| `moloch_status` | Live FPS, CPU-Temp, RAM, Face-ID, Zone, Tracking |
| `moloch_service` | start/stop/restart/status |
| `moloch_logs` | journalctl mit optionalem Filter |
| `moloch_dmesg` | NPU/SEGV/GPU Kernel-Meldungen |
| `moloch_audit` | Regressionstest (PASS/FAIL) |
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

**Hinweis:** Skills (`/moloch-status` etc.) laden Dokumentation. MCP-Tools (`moloch_status()` etc.) liefern Live-Daten. Beides nutzen, nicht verwechseln.

---

## SKILLS (PFLICHT bei Code-Aenderungen — tippe /moloch-...)

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
| hardware | ONVIF, RTSP, PTZ, eWeLink, Thermal, LED | Kamera/Hardware-Probleme |
| gui | Tkinter Panel, Popups | GUI-Aenderungen |
| tracking | PTZ-Tracker, Such-FSM, Arbiter | Tracking/Bewegung |
| voice | Whisper, TTS, Piper, Audio-Pipeline | Sprach-I/O |
| service | moloch_service, IPC, core_integrator | Service/Integration |
| unconscious | TaoEngine, Unterbewusstsein, Self-Tune | Innerer Zustand/Selbstregulation |
| autonomy | Decision Engine, Homeostasis, LLM-Bridge, Night Cycle | Autonome Entscheidungen |
| awareness | Activity, Context, Motion, RoomMap, WorldState | Situationsbewusstsein |
| personality | PersonalityEngine, Mood, TensionIntegrator, EventBus | Persoenlichkeit/Emotion |
| memory | Episodic, Persistent, Vector, ReID, Longterm | Gedaechtnis/Identitaet |
| watchdog | SystemWatchdog, Diagnostics, CapabilityMonitor | System-Health |
| music | Spotify, Track-Index, MusicMemory | Musik/Spotify |
| deepseek | hailo-ollama, DeepSeek API, LLM-Response | LLM-Integration |
| tentacle | ESP32 WiFi-Mic, Firmware, eWeLink, UDP | Peripherie/Firmware |
| stresstest | Chaos Engineering | Stabilitaetstests |

Regeln: 1 Agent = 1 Domain. Bei 85% Token → Uebergabe schreiben.
Markus ist Boss — bei Konflikten entscheidet ER.

**Uebergabe-Protokoll (bei ~85-90% Kontext):**
Datei `~/moloch/logs/agent_handoff.md` schreiben mit: Gate/Phase, was erledigt,
was offen, Blocker, geaenderte Dateien, Service-Status. Neue Instanz liest:
CLAUDE.md → Gate-Kontext → Handoff → Memory.

---

## GELOESTE BUGS — FINGER WEG

1. Pan-Vorzeichen: `pan_delta = -error_x` (MINUS IST KORREKT)
2. RTSP-Doppelzugriff: USE_TAPPAS ueberspringt CameraManager
3. Letterbox: TAPPAS macht das automatisch — KEIN cv2.resize
4. ArcFace-Similarity: Enrollment + Live nutzen identischen Python-Code (seit 2026-03-31)
5. Status-JSON Deadlock: ResultCollector.get_health() Snapshot-Pattern (seit 2026-04-04)
6. hailooverlay entfernt: BBox-Rendering via PIL in panel_preview.py (seit 2026-03-30)

## OFFENE BUGS (Details: Handoff + Memory)

1. Kamera Hot-Plug: Stecker raus/rein → nur Reboot hilft
2. ReID + Hand: SEGV bei Pose-Detection Race — `reid_needed=False` als Workaround
3. person_attr_resnet_v1_18.hef: Kleidung/Alter/Rucksack noch nicht integriert
4. r3d_18.hef: Aktivitaetserkennung noch nicht integriert
5. hailo-ollama: systemd-Service fehlt, laeuft nicht beim Boot
6. MCP moloch_snapshot(): erst nach MCP-Neustart volle Aufloesung

---

## ARCHITEKTUR-STAND

Gate 1 abgeschlossen. Konzeptuell Gate 5+:
Tension-System, TAPPAS Multi-Model, Persoenlichkeit mit Drift,
Speech Evolution, emergentes Verhalten bestaetigt.

Gate 1-5 = Stabilisierung. Der Geist der Maschine ist bereits da.
