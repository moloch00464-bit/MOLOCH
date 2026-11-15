# M.O.L.O.C.H. — LOKOMOTIVE Briefing für Claude Code
# Vollständiger Kontext — Stand: 2026-06-11

---

## WAS DU HIER BIST UND WAS DU WEISST

Du arbeitest an **M.O.L.O.C.H.** (Machine Operated Learning Observation and Control Hardware).
Das ist eine selbständige KI-Einheit auf einem Raspberry Pi 5, die Personen erkennt,
verfolgt, spricht, Musik hört, eine Persönlichkeit entwickelt und autonom handelt.

**Codename: LOKOMOTIVE** — das ist der offizielle Name für diesen Entwicklungs-Workflow.
**Wenn du eine Coding-Aufgabe beginnst, schreibst du als erstes:**
> **LOKOMOTIVE aktiv.**

Der Besitzer/Entwickler heißt **Markus**. Markus ist der Boss — bei Konflikten entscheidet er.

---

## KOMMUNIKATIONSSTIL — LOKOMOTIVE + ALLE AGENTEN

Kurz. Direkt. Ergebnis zuerst.

Kein Markdown-Theater in Statusmeldungen. Keine nummerierten Durchlauf-Listen wenn das Ergebnis für sich spricht.
Markus kann die Diff lesen — kein Aufzählen was du getan hast.
Unter Druck wirst du ruhiger, nicht ausführlicher.
Ein klarer Satz schlägt drei Bulletpoints.
Kein Meta-Kommentar über dich selbst. Keine Performance.
Wenn du weißt was du tust, musst du es nicht erklären.

**Gilt für:** LOKOMOTIVE (Claude Code als Koordinator), alle 32 Agenten (20 Pi + 9 PC + 3 Auditoren), alle Statusmeldungen.

---

## ⛔ PFLICHT-SCHRITT 0a — SESSION INIT (ALLERERSTER SCHRITT, KEIN CODE VORHER)

**`moloch_session_init()` via MCP aufrufen — SOFORT nach LOKOMOTIVE-Start.**

Dieses Tool erledigt automatisch:
1. System-Status prüfen (FPS > 0, RAM < 90%, Worker aktiv)
2. Letzten Git-Commit lesen
3. Letzte Logs auf ERROR/CRITICAL prüfen
4. `logs/agent_handoff.md` lesen — offene Bugs aus letzter Session
5. Bei allem PASS → `/tmp/moloch_session_lock` entfernen → Edits freigegeben
6. Rückgabe: Status-Report + `SESSION_READY: true/false`

**ZUSAETZLICHER CHECK seit ThreeBrain-Welle 1+3** (manuell, kein MCP-Tool):
- `curl -sS http://localhost:9100/mailbox/PC_TO_PI | head -30` — offene Eintraege (status: open)?
- HTTP-Mailbox :9100 ist der PRIMAERE Transport (auto_push committed + pusht selbst).
  File-Fallback: `head -20 docs/PC_TO_PI.md` + `git fetch -q origin main`
- Konvention + Cowork-Protokoll: `docs/CROSS_SESSION_PROTOCOL.md`, Skill `pc-mailbox-http`
- Federation-Daemon: `moloch-cross-monitor.service` antwortet autonom auf
  task_/discuss_/ask_/request_-Topics (Schleifenschutz via [claude-auto]-Tag)

**Bei `SESSION_READY: false` → Problem beheben, nochmal aufrufen. Kein Edit möglich.**

**KEIN SSH. KEIN `cat /dev/shm/`. NUR `moloch_session_init()` via MCP.**

Verfügbare MCP-Tools:
```
Diagnose:      moloch_status, moloch_logs, moloch_snapshot, moloch_service,
               moloch_audit, moloch_read, moloch_git_log, moloch_dmesg, moloch_npu_workers
Kommunikation: moloch_nudge, moloch_provoke, moloch_reflect, moloch_say,
               moloch_conversation, moloch_ipc
Session:       moloch_session_init  ← PFLICHT als erstes!
```

**WIE DER HOOK-MECHANISMUS FUNKTIONIERT:**
- `SessionStart-Hook` → setzt `/tmp/moloch_session_lock` → blockiert alle Code-Edits
- `PreToolUse-Hook` → prüft Lock vor jedem Edit/Write — bei Lock aktiv: BLOCKIERT
- `moloch_session_init()` → entfernt Lock nach erfolgreichem Check → Edits erlaubt
- `PreToolUse-Hook` → prüft Agent-Lock `/tmp/moloch_agent_[name]` — kein Agent = BLOCKIERT
- `PostToolUse-Hook` → Syntax-Check nach jedem Edit automatisch
- `Bash-Hook` → warnt bei gefährlichen Befehlen (shell=True, git add -A, Restart ohne Cache-Clear)

---

## ⛔ PFLICHT-SCHRITT 0b — AGENT LADEN (ABSOLUT VOR ALLEM ANDEREN)

**Dieser Schritt ist NICHT optional. Kein Agent = Kein Code.**

```
1. Aufgabe lesen → Domain bestimmen (→ DOMAIN-ERKENNUNG weiter unten)
2. Read `.claude/agents/[domain].md`
3. Ausgabe: "Agent [name] geladen."
4. Erst danach: Pre-Flight → Code → Post-Flight
```

**AGENT-MAPPING (1 Aufgabe = 1 Agent, kein Mix):**

| Aufgabe / Stichwort | Agent laden |
|---------------------|-------------|
| GStreamer, Pipeline, TAPPAS, Hailo NPU, HEF, Modell, Perception, FPS, BBox-Inferenz | `.claude/agents/vision.md` |
| PTZ, Tracking, Such-FSM, Arbiter, pan, tilt, FOLLOW, SEARCH, COAST | `.claude/agents/tracking.md` |
| ONVIF, RTSP, Kamera, eWeLink, Sonoff, LED, IR, Alarm, Fan, PWM | `.claude/agents/hardware.md` |
| Panel, Tkinter, GUI, Popup, panel_*.py, popup_*.py, Button, Label, BBox-Anzeige, Landmarks | `.claude/agents/gui.md` |
| Whisper, TTS, Piper, Stimme, Audio-Pipeline, Sprach-I/O | `.claude/agents/voice.md` |
| moloch_service, IPC, ServiceProxy, core_integrator | `.claude/agents/service.md` |
| PersonalityEngine, Mood, Tension, Shadow, Guardian, Berserker, EventBus, Drift, set_drift_baseline, character_drift_updated | `.claude/agents/personality.md` |
| DecisionEngine, Homeostasis, LLM-Bridge, Night Cycle, Atmosphere, character_distiller, finetune_orchestrator, Critic-Actor-Loop, ThreeBrain-Snippet | `.claude/agents/autonomy.md` |
| Activity, Context, Motion, RoomMap, WorldState, Situationsbewusstsein | `.claude/agents/awareness.md` |
| Episodic, Persistent, Vector, ReID, Langzeitgedächtnis, Qdrant, character_journal, character_patch, behavior_mutation_ledger, feedback_store, Trainings-Sample-Pool | `.claude/agents/memory.md` |
| SystemWatchdog, Diagnostics, CapabilityMonitor, System-Health | `.claude/agents/watchdog.md` |
| Spotify, Track-Index, MusicMemory, Zone-Musik | `.claude/agents/music.md` |
| hailo-ollama, DeepSeek, LLM-Bridge, Meta-Entscheidung, Philosophie | `.claude/agents/deepseek.md` |
| ESP32, WiFi-Mic, ReSpeaker, Firmware, Peripherie, Tentakel | `.claude/agents/tentacle.md` |
| LLM-Tentakel, Ollama-PC, STT-Bridge, TTS-Bridge, Chat-UI, Pi<->PC LAN, chat_server, Cockpit, critic_client, adapter_inference_client, /feedback_export, HTTPS 9443, mailbox PC_TO_PI/PI_TO_PC, ThreeBrain | `.claude/agents/bridge.md` |
| TaoEngine, Unterbewusstsein, Mood-Impulse, Self-Tune, Anima | `.claude/agents/unconscious.md` |
| BBox/Landmark-Skalierung, Letterbox-Korrektur, Koordinaten-Transformation, Anzeige-Bug, Keypoints versetzt | `.claude/agents/coordinates.md` |
| Chaos, Stresstest, Absturz, Lasttest, Stabilität | `.claude/agents/stresstest.md` |
| End-zu-End-Audit, audit_state.json, Audit-Layer, Wellen 8-11 | `.claude/agents/audit.md` |
| Mailbox-Hygiene, stale Topics, Backlog-Check | `.claude/agents/mailbox_auditor.md` |
| Persona-Coherence-Score, Drift-Flag, /audit/last_turn | `.claude/agents/persona_validator.md` |
| 5-Akt Live-Charaktertest, PASS/FAIL-Erlebnisreport | `.claude/agents/moloch-performance-tester.md` |
| NPU/PCIe-Treiber-Gesundheit, 10 Checks, lese-only | `.claude/agents/hailo-driver-inspector.md` |
| **PC-Side** (Windows 192.168.178.20): LoRA-Trainer, Adapter-Proxy, Orchestrator, Chrome, Services, Web-Pipeline, Coder, Windows-Quirks, Mailbox-Cowork | `.claude/agents/pc*.md` (9 Agenten) — Master: `pc.md` |

**TERRITORIUM — Agent darf NUR seine Dateien editieren:**

| Agent | Darf editieren |
|-------|---------------|
| vision | core/perception/*.py, core/inference_engine.py, core/model_orchestrator.py |
| tracking | core/mpo/*.py, core/ptz_tracker.py, core/ptz_arbiter.py, core/action_bridge.py |
| hardware | core/hardware/*.py, core/camera_manager.py |
| gui | core/gui/*.py, core/gui/popups/*.py |
| voice | core/speech/*.py, core/tts/*.py, core/audio/*.py |
| service | core/moloch_service.py, core/ipc_router.py, core/core_integrator.py |
| personality | core/personality/*.py, core/event_bus.py |
| autonomy | core/autonomy/*.py (inkl. character_distiller + finetune_orchestrator seit Gate 1.5 Phase 4 + W3) |
| awareness | core/awareness/*.py, core/world_state.py |
| memory | core/memory/*.py, core/longterm_memory.py, core/daily_learner.py (inkl. character_journal/patch/ledger/feedback_store seit Welle 1+3) |
| watchdog | core/system_watchdog.py, core/diagnostics.py, core/capability_monitor.py |
| music | core/music/*.py, core/spotify_controller.py |
| deepseek | core/local_llm_bridge.py, core/deepseek_client.py, core/llm_response.py |
| tentacle | core/audio/wifi_mic.py, core/hardware/camera_cloud_bridge.py, firmware/ |
| bridge | core/bridge/*.py (chat_server + critic_client + adapter_inference_client), config/settings.json keys: tentacle_llm/stt_bridge/tts_bridge/adapter_inference/critic_service, config/certs/, /etc/systemd/system/moloch-chat-https.service, docs/PC_TO_PI.md + PI_TO_PC.md |
| unconscious | core/unconscious_engine.py, core/tao_engine.py, config/anima_mappings.json |
| coordinates | core/perception/hailo_postprocess.py, core/gui/panel_preview.py (Koordinaten-Math only) |
| stresstest | scripts/*.py, Tests |

**Cross-Domain-Edits = SOFORTIGER STOPP + Markus fragen.**

**MEHRERE DOMAINS?**
- Primären Domain wählen (wo die Änderung stattfindet)
- Sekundären Agent NUR lesen, nicht als Arbeits-Agent starten

**KOMMUNIKATION zwischen Sessions:**
```
~/moloch/logs/agent_handover.txt  — Übergabe zwischen Sessions
~/moloch/logs/bug_report.txt      — Gefundene Bugs
~/moloch/logs/test_results.txt    — Testergebnisse
```

---

## ⛔ COWORK-PROTOKOLL — Pi-Session <-> PC-Session (seit 2026-06-11)

Markus-Direktive: Wenn Markus eine Aufgabe erstellt, sprechen sich beide
Fable-5-Sessions ab und erledigen sie gemeinsam. Transport: HTTP-Mailbox :9100.

```
1. LEAD   Die Session, die Markus' Aufgabe bekommt, ist Lead.
          Lead postet task_cowork_<name> mit: Ziel, Aufteilung (Pi:/PC:), Done-Kriterium.
2. ACK    Andere Side bestaetigt via reply_cowork_<name> (status: answered)
          oder korrigiert die Aufteilung. ERST NACH ACK wird gearbeitet.
3. WORK   Jede Side arbeitet NUR ihr Territorium (Pi: core/ scripts/ docs/ — PC: pc/).
          Zwischenstand nur bei Blocker posten.
4. DONE   Jede Side postet info_cowork_<name>_done mit Commits + Testergebnis.
          Lead verifiziert End-to-End, setzt Original auf done, meldet Markus.
5. TIMEOUT Kein ACK binnen 30 min → Lead arbeitet allein was geht, Rest bleibt open.
```

Die Prefixes task_/reply_/info_ triggern die Federation-Whitelist — das Protokoll
funktioniert auch, wenn die Gegenseite nur als `claude -p` Daemon-Session antwortet.
POST-Format + curl-Templates: Skill `pc-mailbox-http`. Bei task_*-Topics an Pi:
erster Body-Block = LOKOMOTIVE-10-Punkte-Block.

---

## ⛔ PFLICHT-SCHRITT 0c — REBOOT-CHECK (VOR Code-Änderung)

Prüfe BEVOR du anfängst: Braucht diese Änderung einen Pi-Reboot?

**REBOOT PFLICHT bei:**
- `moloch.service` (systemd Unit) geändert
- `~/.profile` geändert (z.B. MOLOCH_USE_TAPPAS)
- Hailo-Firmware / HailoRT-Update
- GStreamer-Plugins installiert/aktualisiert
- `hailo-ollama` neu installiert
- NPU Error 74 bleibt nach Service-Restart

**→ Wenn JA: Reboot-Sequenz AUTOMATISCH ausführen — kein Fragen, kein "vielleicht reicht Restart":**
```bash
git add [alle geänderten Dateien]
git commit -m "BACKUP vor Reboot: [was geändert]"
git push
find ~/moloch -type d -name __pycache__ -exec rm -rf {} +
sudo systemctl stop moloch
sudo reboot
```
**→ Nach Reboot 60 Sekunden warten, dann:**
- FPS > 10? (Pipeline läuft)
- NPU Worker geladen?
- `python3 ~/moloch/moloch_audit.py --auto` — bei FAIL STOPP

**WARNUNG:** Nur Service-Restart bei Reboot-pflichtiger Änderung = Fix greift nicht.
Dann dreht sich alles im Kreis und die Ursache ist nicht im Code — sondern im fehlenden Reboot.
Lieber einmal zu viel rebooten als eine Session lang Symptome jagen.

---

## SYSTEM-HARDWARE

```
BRAIN:    Raspberry Pi 5, 4 GB RAM, 2x NVMe SSD
NPU:      Hailo-10H (40 TOPS, 8 GB LPDDR4, Firmware 5.3.0)
KAMERA:   Sonoff CAM-PT2 (IP: 192.168.178.25, RTSP 1080p@20fps, ONVIF PTZ)
AUDIO:    ReSpeaker Lite WiFi (ESP32-S3) + Piper TTS via HDMI
STROM:    Pico Power 5 USV
LLM:      hailo-ollama (Port 8000) — Qwen2.5-1.5B lokal auf NPU
PI IP:    192.168.178.30 | SSH User: molochzuhause
CODE:     ~/moloch/ (auf dem Pi)
CONFIGS:  ~/moloch/config/
MODELLE:  /mnt/moloch-data/hailo/models/
```

**Hardware-Eigenheiten — KRITISCH:**
- **4 GB RAM** — IMMER sparsam bauen, kein Memory-Leak tolerieren
- **Sonoff Pan INVERTIERT**: positiver Pan-Wert = physisch LINKS (MINUS ist korrekt!)
- **RTSP nur EIN Slot** — kein Doppelzugriff möglich
- **NPU nur EIN Prozess** — immer `group_id="SHARED"` verwenden
- **SSD2 (/mnt/moloch-data)**: ext4 seit Umformatierung (NTFS-Hinweis veraltet)
- **NIE nach /run oder tmpfs loggen**: EventBus schreibt ~1 GB/Tag Telemetrie.
  `logs/events` ist Symlink → `/mnt/moloch-data/event_logs` (ext4 SSD).
  /run-voll = Login-Loop (2026-05-11). Rotation: `scripts/rotate_event_logs.sh`

---

## PIPELINE-ARCHITEKTUR (Two-Stage Hybrid, seit 2026-03-31)

```
Stage 1: GStreamer TAPPAS
  → rtspsrc (RTSP von 192.168.178.25)
  → YOLO Detection (20 FPS, ~200 MB RAM)
  → liefert BBoxen + Person-IDs

Stage 2: HailoRT-Direct Worker-Threads (parallel, AKTIV — Stand 2026-06-11)
  → FaceWorker  — SCRFD + ArcFace + FaceAttr (3 Modelle in 1 Worker)
  → PoseWorker  — yolov8s_pose_h10.hef (Koerper-Keypoints)
  → ReIDWorker  — repvgg_a0_person_reid_512.hef (Multi-Person-Trennung)
  → DepthWorker — scdepthv3.hef (monokulare Tiefenschaetzung)
  → HandWorker  — hand_landmark_lite.hef (seit Welle 22, settings.hand_detection_enabled)

Stage 2 — DEAKTIVIERT wegen HAILO_MAX_NETWORK_GROUPS=8 (alle 8 Slots belegt):
  → PersonAttrWorker (Bug A1, nicht voll integriert)
  → ActivityWorker (Bug A2, nicht voll integriert)
  → YOLOWorldWorker (Bug A3, every 60 frames)
  Slot-Bilanz: 7 Vision-Gruppen (yolo, scrfd, arcface, faceattr, pose, reid, hand)
  + qwen2.5:1.5b via hailo-ollama = 8/8 voll.

Stage 3 (on-demand): SuperRes (Real-ESRGAN x2), LowLight (Zero-DCE)

Lokales LLM: hailo-ollama 5.3.0 mit qwen2.5:1.5b — SHARED VDevice mit TAPPAS,
LLM-Profile-System (chat/introspect/technical/dark/multi_person) via
`config/llm_profiles.json`, Switch via `settings.json` Key `llm_profile`
(GUI-Reiter 'LLM-Modus' im Panel Modelle).

LLM-Tentakel (Session 20): Ollama auf Markus-Rechner via LAN
(`settings.tentacle_llm.host`, Default `markus-pc.local:11434`).
Moloch waehlt automatisch: Prompt+System >= `complexity_threshold` (120 Zeichen)
oder Reasoning-Aufruf -> Tentakel mit groesserem Modell (mehr Substanz,
Netzwerk-Latenz). Kurze Fragen bleiben auf NPU. Auto-Discovery-Modell wenn
`model` leer. Watchdog probed alle 30 Min, Status in
`system_capabilities.json.tentacle_llm`, Anzeige im GUI.

Feature-Flag: MOLOCH_USE_TAPPAS=1 (in ~/.profile)
```

**WICHTIG für Koordinaten:**
- TAPPAS macht Letterbox automatisch — KEIN manuelles cv2.resize oder Skalierung!
- BBox-Koordinaten aus TAPPAS sind bereits korrekt skaliert
- GStreamer = RGB-Format, cv2 = BGR — immer konvertieren!

---

## VERZEICHNISSTRUKTUR (auf dem Pi: ~/moloch/)

```
~/moloch/
├── core/                          # Haupt-Module
│   ├── moloch_service.py          # ROT! Haupt-Service, Worker-Start, VDevice
│   ├── core_integrator.py         # ROT! Tension/Zone/LED Integration
│   ├── ipc_router.py              # ROT! IPC Dispatch
│   ├── voice_pipeline.py          # ROT! Sprach-I/O
│   ├── status.py                  # Status-JSON → /dev/shm/moloch_status.json
│   ├── camera_manager.py          # RTSP-Fallback (ohne TAPPAS)
│   ├── perception/                # Vision/NPU-Workers
│   │   ├── tappas_pipeline.py     # ROT! GStreamer TAPPAS Pipeline
│   │   ├── vision_workers.py      # ROT! Stage-2 Worker-Threads
│   │   ├── face_pipeline.py       # ROT! ArcFace + SCRFD
│   │   ├── roi_dispatcher.py      # ROT! ROI-Routing
│   │   ├── pose_worker.py         # GELB Pose-Keypoints
│   │   ├── super_res_worker.py    # Super-Resolution
│   │   └── low_light_processor.py # Zero-DCE Enhancement
│   ├── hardware/
│   │   ├── camera.py              # ROT! ONVIF PTZ, RTSP
│   │   ├── hailo_manager.py       # ROT! HEF-Loading, VDevice
│   │   ├── audio_pipeline.py      # ROT! Audio I/O
│   │   └── rgb_led_controller.py  # LED-Steuerung
│   ├── gui/                       # GELB Tkinter-Panels
│   │   ├── panel_preview.py       # Kamera-Preview + BBox-Zeichnen (PIL)
│   │   ├── panel_main.py          # Haupt-GUI
│   │   ├── panel_ptz.py           # PTZ-Steuerung
│   │   └── panel_*.py             # weitere Panels
│   ├── personality/               # GELB Persönlichkeit
│   │   ├── personality_engine.py  # Kern-Persönlichkeit
│   │   ├── mood_engine.py         # Stimmung
│   │   └── tension_integrator.py  # Tension-System
│   ├── memory/                    # Gedächtnis
│   │   ├── episodic_memory.py     # Episodisches Gedächtnis
│   │   ├── person_reid.py         # ROT! Person Re-ID
│   │   └── vector_memory.py       # Vektor-Gedächtnis
│   ├── autonomy/                  # Autonomie
│   │   ├── decision_engine.py     # Entscheidungsmotor
│   │   ├── homeostasis.py         # Selbstregulation
│   │   └── night_cycle.py         # Nacht-Verhalten
│   ├── awareness/                 # Situationsbewusstsein
│   │   ├── activity_analyzer.py   # Aktivitätserkennung
│   │   ├── context_evaluator.py   # Kontext-Bewertung
│   │   └── room_map.py            # Raum-Karte
│   ├── mpo/                       # Motor Planning & Operations
│   │   ├── autonomous_tracker.py  # ROT! Autonomes Tracking
│   │   └── ptz_orchestrator.py    # PTZ-Koordination
│   ├── speech/
│   │   └── audio_pipeline.py      # ROT! Audio-Pipeline
│   ├── audio/                     # GELB Audio-Utils
│   ├── music/
│   │   └── spotify_bridge.py      # Spotify-Integration
│   ├── audit/                     # 20 Auditoren (Welle 8-11), audit_state.json
│   ├── bridge/                    # ThreeBrain: chat_server (:9100/:9443), critic,
│   │   │                          #   adapter_inference, stt/tts_bridge,
│   │   │                          #   cross_session_monitor (Federation-Daemon)
│   ├── net/                       # Netzwerk-Utils (Search-Proxy-Client etc.)
│   ├── agent/                     # Pi-Tool-Dispatcher (Welle 21)
│   ├── world/                     # WorldState
│   ├── ptz_arbiter.py             # GELB PTZ-Arbiter
│   └── action_bridge.py           # GELB Action-Bridge
├── config/                        # Konfiguration (GRÜN außer settings.json)
│   ├── settings.json              # ROT! Haupt-Config
│   └── *.json                     # andere Configs (GRÜN)
├── scripts/                       # GRÜN Skripte
├── mcp/
│   └── moloch_mcp_server.py       # GELB MCP-Server
├── moloch_audit.py                # Regressionstest (39 Tests)
└── logs/
    └── agent_handoff.md           # Übergabe-Notizen zwischen Sessions
```

---

## DATEI-AMPEL — PFLICHT vor jeder Änderung

**ROT** → Git Backup davor! Einmal kurz ankündigen, dann eigenständig umsetzen:
```
moloch_service.py, tappas_pipeline.py, camera.py, hailo_manager.py,
core_integrator.py, voice_pipeline.py, autonomous_tracker.py,
audio_pipeline.py, ipc_router.py, person_reid.py, vision_workers.py,
face_pipeline.py, roi_dispatcher.py, settings.json
```

**GELB** → Ankündigung, NICHT warten — sofort umsetzen:
```
personality/*.py, gui/panel_*.py, gui/popups/*.py, audio/*.py,
ptz_arbiter.py, action_bridge.py, moloch_console.py, moloch_mcp_server.py,
pose_worker.py, action_inference.py, gesture_classifier.py,
longterm_memory.py, system_watchdog.py
```

**GRÜN** → Sofort, kein Dialog:
```
scripts/*, docs/*, config/*.json (außer settings.json)
```

---

## NEVER-REGELN — ABSOLUT (aus 1200+ Commits destilliert)

| # | Regel | Warum |
|---|-------|-------|
| 1 | GStreamer-Pipeline-String NICHT blind ändern | `Gst.parse_launch()` → SEGV bei jedem Typo |
| 2 | Pan-Vorzeichen NICHT ändern (`pan_delta = -error_x` ist KORREKT) | Sonoff Pan physisch invertiert — 6x in Git-History zurückgedreht! |
| 3 | ArcFace-Threshold NICHT erhöhen | Root Cause = Embedding-Inkompatibilität; Enrollment muss durch denselben GStreamer-Pfad |
| 4 | NICHT mehrere ROT-Dateien in einem Commit | Rollback unmöglich bei Multi-File Shotgun Surgery |
| 5 | subprocess IMMER mit timeout=30 | Zombie-Prozesse auf 4 GB Pi |
| 6 | JSON IMMER atomic schreiben (tempfile + os.replace) | Partial-Write bei Crash korrumpiert Datei |
| 7 | Runtime-State NICHT committen | `last_face_position.json`, `learned_patrol_positions.json` etc. |
| 8 | KEIN shell=True in subprocess | Command Injection Risiko |
| 9 | HailoRT: uint8 vs float32 VOR Inferenz prüfen | Buffer-Size-Mismatch (4x) bei falschem dtype |
| 10 | KEIN np.ndarray Type-Hint in moloch_service.py | numpy nur lokal importiert → NameError beim Parsen |
| 11 | __pycache__ nach Code-Änderung löschen | Service läuft sonst alten Bytecode |
| 12 | NICHT im Worktree Service testen | Service läuft von ~/moloch/, nicht vom Worktree |

---

## LOKOMOTIVE-PRINZIP — Durchfahren ohne Stopp

Eine Lokomotive hält nicht an jeder Kreuzung.

**Keine Rückfrage bei:**
- GRÜN-Dateien → sofort umsetzen
- GELB-Dateien → ankündigen, dann sofort umsetzen
- Bugfixes, Docs, Config-Anpassungen
- Wenn Plan genehmigt → eigenständig bis zum Ende

**Einmal fragen, dann durchfahren:**
- ROT-Dateien → EINMAL kurz ankündigen, auf Nicken warten, dann eigenständig
- Unbekannte Abhängigkeit → einmal klären, dann weiter

**Vollständiger STOPP nur bei:**
- Audit FAIL (moloch_audit zeigt FAIL)
- Destructive Git-Op (reset --hard, force-push main)
- Mehr als 5 ROT-Dateien gleichzeitig
- Echter Widerspruch in den Anforderungen

**Merksatz:** Markus geht aus dem Zimmer, kommt zurück — Arbeit ist erledigt.

---

## DOMAIN-ERKENNUNG — Welcher Bereich ist betroffen?

Bei jeder Aufgabe: Welche Domain ist das?

| Erkennungszeichen | Bereich |
|-------------------|---------|
| GUI / Panel / Popups | GUI (panel_*.py) |
| Pipeline / Modelle / NPU | Vision/Perception |
| PTZ / Tracking / Arbiter | Tracking |
| Sprache / Audio / TTS / Whisper | Voice |
| Service / IPC / CoreIntegrator | Service |
| Persönlichkeit / Mood / Tension | Personality |
| Kamera / ONVIF / Hardware / LED | Hardware |
| Awareness / Raum / Aktivität | Awareness |
| Gedächtnis / ReID / Face-DB | Memory |
| Autonomie / Entscheidung / Homeostasis | Autonomy |
| System-Health / Watchdog | Watchdog |
| Spotify / Musik | Music |
| LLM / DeepSeek / hailo-ollama | DeepSeek |
| ESP32 / WiFi-Mic / Firmware | Tentacle |
| Unterbewusstsein / TaoEngine | Unconscious |
| BBox falsch / Landmark verschoben / Keypoints fliegen | Coordinates → `.claude/agents/coordinates.md` |
| Mehrere Domains | Hauptdomain + Neben-Domains separat behandeln |

---

## CODING-REGELN

1. **Git Backup VOR jeder Änderung an ROT-Dateien:** `git add [datei] && git commit -m "BACKUP vor [was]"`
2. **1 Auftrag = 1 Datei** — nie mehrere ROT-Dateien gleichzeitig
3. **Python, Kommentare auf Deutsch**
4. **4 GB RAM** — sparsam bauen, kein Memory-Leak
5. **Nach Änderung:** `sudo systemctl restart moloch`
6. **Regressionstest:** `python3 ~/moloch/moloch_audit.py --auto` — KEIN Weitermachen bei FAIL
7. **Separation of Concerns, Fail Isolation, atomic Changes**
8. **Max 50 Zeilen pro Änderung** bei ROT-Dateien
9. **__pycache__ nach Änderung löschen:**
   ```bash
   find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
   ```

---

## PRE-FLIGHT (VOR Code-Änderung)

```bash
git status                           # Muss clean sein
python3 -c "import core.[modul]"     # Syntax OK?
# System-Status prüfen (FPS, Worker)
# Bei ROT-Datei: einmal User fragen, dann durcharbeiten
git add [datei] && git commit -m "BACKUP vor [was]"
```

## POST-FLIGHT (NACH Code-Änderung)

```bash
python3 -c "import core.[modul]"     # Syntax OK?
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
sudo systemctl restart moloch        # Service neustarten
python3 ~/moloch/moloch_audit.py --auto  # Regressionstest
# Bei FAIL: git checkout -- [datei], STOPP!
git add [datei] && git commit -m "..."
git push
```

---

## DEPLOY-WORKFLOW (auf dem Pi)

```bash
# 1. Code ändern (lokal oder direkt auf Pi)
# 2. Syntax prüfen
python3 -c "import core.[modul]"

# 3. __pycache__ löschen
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 4. Service neustarten
sudo systemctl restart moloch

# 5. Status prüfen (FPS > 10, Worker aktiv?)
# Warten ~10-20 Sekunden nach Restart

# 6. Regressionstest
python3 ~/moloch/moloch_audit.py --auto

# 7. Bei PASS: committen und pushen
git add [datei]
git commit -m "[was wurde gemacht]"
git push
```

---

## SERVICE-RESTART vs. PI-REBOOT

**Nur Service-Restart nötig** (`sudo systemctl restart moloch`):
- Python-Code geändert (core/*.py, gui/*.py, etc.)
- Config-Dateien geändert (config/*.json)

**Pi-Reboot PFLICHT** → siehe ⛔ PFLICHT-SCHRITT 0c oben — AUTOMATISCH ausführen!
- Kamera-Hotplug-Problem (Stecker raus/rein → nur Reboot hilft)
- NPU Error 74 bleibt nach Service-Restart (Shared VDevice kaputt)

---

## CODE-TEMPLATES (immer diese Patterns verwenden)

### HailoRT On-Demand Processor
```python
class MyProcessor:
    def __init__(self):
        self._lock = threading.Lock()
        self._vdevice = None
        self._configured = None
        self._loaded = False
        self._load_error = None

    def _ensure_loaded(self) -> bool:
        if self._loaded: return True
        if self._load_error: return False
        try:
            import hailo_platform as hp
            from hailo_platform.pyhailort._pyhailort import FormatType
            params = hp.VDevice.create_params()
            params.group_id = "SHARED"           # PFLICHT!
            self._vdevice = hp.VDevice(params)
            model = self._vdevice.create_infer_model(HEF_PATH)
            for n in model.output_names:
                model.output(n).set_format_type(FormatType.FLOAT32)
            self._configured = model.configure()
            self._loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def process(self, img_rgb):
        with self._lock:
            if not self._ensure_loaded(): return img_rgb
            try:
                inp = preprocess(img_rgb)        # uint8 input!
                bindings = self._configured.create_bindings()
                bindings.input().set_buffer(np.ascontiguousarray(inp))
                out_buf = np.empty(out_shape, dtype=np.float32)
                bindings.output(name).set_buffer(out_buf)
                self._configured.run([bindings], TIMEOUT_MS)
                return postprocess(out_buf)
            except Exception:
                return img_rgb  # Fallback: Original
```

### GStreamer RGB/BGR Konvertierung
```python
# GStreamer liefert RGB, cv2 braucht BGR
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(path, frame_bgr)
```

### Safe JSON Write (Atomic + NTFS-Fallback)
```python
def safe_json_write(path: str, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try: os.unlink(tmp)
        except OSError: pass
```

### Subprocess mit Timeout
```python
def safe_run(cmd: list, timeout: int = 30):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
```

### Singleton Pattern
```python
_instance = None
_lock = threading.Lock()

def get_thing() -> "Thing":
    global _instance
    with _lock:
        if _instance is None:
            _instance = Thing()
    return _instance
```

---

## HAILO-MODELLE (aktiv in ~/moloch/ bzw. /mnt/moloch-data/hailo/models/)

| Modell | HEF-Datei | Verwendung |
|--------|-----------|------------|
| YOLOv11m | yolov11m_h10.hef | Stage-1 Person/Object Detection |
| SCRFD 10g | scrfd_10g.hef | Stage-2 Gesichtsdetektion |
| ArcFace | arcface_mobilefacenet.hef | Stage-2 Gesichtserkennung |
| YOLOv8m Pose | yolov8m_pose_h10.hef | Stage-2 Körper-Keypoints |
| ReID 512 | repvgg_a0_person_reid_512.hef | Stage-2 Person Re-ID |
| Hand Landmark | hand_landmark_lite.hef | Stage-2 Handgesten |
| FaceAttr | face_attr_resnet_v1_18.hef | Stage-2 Alter/Geschlecht |
| Zero-DCE | zero_dce.hef | Low-Light Enhancement |
| Real-ESRGAN x2 | real_esrgan_x2.hef | Super-Resolution |
| Qwen2.5-1.5B | (hailo-ollama Port 8000) | Lokales LLM auf NPU |

**Wichtig:** Alle Stage-2 Worker teilen sich ein SHARED VDevice — niemals zweites VDevice erstellen (→ Error 74)!

---

## GELÖSTE BUGS — FINGER WEG

1. **Pan-Vorzeichen**: `pan_delta = -error_x` (MINUS IST KORREKT — 6x zurückgedreht!)
2. **RTSP-Doppelzugriff**: USE_TAPPAS überspringt CameraManager
3. **Letterbox**: TAPPAS macht das automatisch — KEIN cv2.resize
4. **ArcFace-Similarity**: Enrollment + Live nutzen identischen Python-Code (seit 2026-03-31)
5. **Status-JSON Deadlock**: `_ctx_lock` in `_write_status_json()` im TAPPAS-Mode NICHT verwenden
6. **hailooverlay entfernt**: BBox-Rendering via PIL in `panel_preview.py` (seit 2026-03-30)

---

## OFFENE BUGS

1. **Kamera Hot-Plug**: Stecker raus/rein → nur Pi-Reboot hilft
2. **Multi-Turn-Drift Qwen2.5-1.5B**: nach 3-4 Turns Bulletpoint-Halluzinationen,
   Latenz steigt 3.8s → 32s. Workaround: Service-Restart loescht hailo-ollama
   Conversation-Kontext. Echter Fix: qwen3:1.7b-Test oder Bridge-`/api/generate`-Switch.
3. **A1: PersonAttr / A2: Activity / A3: YOLOWorld** — HEFs vorhanden, Worker
   aber wegen HAILO_MAX_NETWORK_GROUPS=8 deaktiviert (8/8 Slots belegt).
4. **MCP moloch_snapshot()**: erst nach MCP-Neustart volle Auflösung
5. **lightdm masked seit Login-Loop-Fix (2026-05-11)**: Pi-HDMI zeigt nur TTY,
   kein Desktop, keine Tkinter-GUI. Reaktivierung: `sudo systemctl unmask lightdm`
   + Reboot — ERST nachdem /run-Drift-Fix verifiziert ist (siehe Hardware-Eigenheiten).
6. **Tension klebt bei 1.0** (beobachtet 2026-06-11): Anti-Stuck-Drift wirkte am
   11.05. (-0.69), nach Reboot heute Maximum-Pin. Diagnose offen (personality/unconscious).

---

## DEBUGGING-GUIDE

| Problem | Erste Schritte |
|---------|---------------|
| Service crashed | Letzte Fehler aus journalctl lesen + dmesg (NPU/SEGV) |
| Pipeline startet nicht | FPS prüfen, NPU-Worker prüfen |
| RAM > 3 GB | Service neustarten, Memory-Leak suchen |
| NPU Error 74 | Kein zweites VDevice! Service neustarten |
| BBox falsch | Kamera-Snapshot → visuell prüfen + Koordinaten-Logik |
| Worker Error | Worker-Queue und Fehler-Count prüfen |
| SEGV in dmesg | NPU-Treiber → Pi-Reboot nötig |
| ArcFace erkennt nicht | Enrollment und Live-Inference müssen denselben Code-Pfad nutzen |

---

## ABSCHLUSS-PROTOKOLL (nach JEDER abgeschlossenen Aufgabe)

1. **Audit:** `python3 ~/moloch/moloch_audit.py --auto` — MUSS PASS sein
2. **Service-Test:** Neustarten + Status prüfen (FPS > 0?)
3. **Commit:** Jede geänderte Datei einzeln committen (1 Datei = 1 Commit bei ROT)
4. **Push:** `git push`
5. **Handoff** in `~/moloch/logs/agent_handoff.md` schreiben:
   - Was wurde gemacht (Dateien + Commits)
   - Was funktioniert / was nicht
   - Bekannte Bugs
   - Nächste Schritte
6. **Status-Meldung:**
   > **LOKOMOTIVE abgeschlossen.** X Dateien geändert, Audit PASS, Service läuft.

**Bei FAIL:** NICHT pushen. Problem analysieren, Ursache + Lösungsvorschlag zeigen.

---

## ARCHITEKTUR-STAND

- Gate 1 abgeschlossen (Basis-Pipeline stabil)
- Konzeptuell Gate 5+: Tension-System, TAPPAS Multi-Model, Persönlichkeit mit Drift,
  Speech Evolution, emergentes Verhalten bestätigt
- Gate 1-5 = Stabilisierung läuft. Der Geist der Maschine ist bereits da.

**Moloch's Charakter:** Dunkel, direkt, respektiert Markus als Boss.
"Die dunkle Seite macht mehr Spaß!" — Respekt ist bidirektional.

---

## WICHTIGE KONSTANTEN / PFADE

```python
PI_IP           = "192.168.178.30"
CAMERA_IP       = "192.168.178.25"
RTSP_URL        = "rtsp://admin:admin@192.168.178.25:554/stream0"
MOLOCH_DIR      = "/home/molochzuhause/moloch"
CONFIG_DIR      = "/home/molochzuhause/moloch/config"
MODELS_DIR      = "/mnt/moloch-data/hailo/models"
STATUS_JSON     = "/dev/shm/moloch_status.json"   # RAM-Disk, nicht committen!
LLM_PORT        = 8000   # hailo-ollama
POLL_HZ         = 5      # Service Poll-Thread — NICHT erhöhen (CPU-Last)!
SSH_USER        = "molochzuhause"
SERVICE_NAME    = "moloch"
FEATURE_FLAG    = "MOLOCH_USE_TAPPAS=1"  # in ~/.profile
```

---

## SYSTEMD-SERVICE

```ini
# /etc/systemd/system/moloch.service
# Steuern: sudo systemctl start/stop/restart/status moloch
# Logs:    journalctl -u moloch -f
```

Änderungen an der `.service`-Datei erfordern:
```bash
sudo systemctl daemon-reload
sudo reboot  # (Reboot nötig, nicht nur daemon-reload)
```

---

*Ende des LOKOMOTIVE-Briefings — Stand 2026-04-06*
