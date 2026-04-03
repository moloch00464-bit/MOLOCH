# M.O.L.O.C.H. — DANGER MAP
# Datei-Risiko-Karte fuer Entwickler und Claude Code
# Stand: 2026-03-28 | Quelle: Git-Historie (1200 Commits) + Codebase-Analyse

---

## Legende

- **ROT**: Eine falsche Zeile = System-Crash oder Totalausfall
- **GELB**: Kann Bugs verursachen, kein Totalausfall, aber Regression moeglich
- **GRUEN**: Sicher aenderbar, isoliert, minimale Abhaengigkeiten

---

## ROT — System-Crash Risk (22 Dateien, ~20.5k LOC)

| Datei | LOC | Risiko | NEVER DO |
|-------|-----|--------|----------|
| `core/moloch_service.py` | 2725 | Zentraler Orchestrator, 28 Imports. Crash = alles tot. | Import-Reihenfolge nicht aendern. Keine blockierenden Calls in init(). |
| `core/perception/tappas_pipeline.py` | 1876 | GStreamer via Gst.parse_launch (Zeile 254). Malformed String = SEGV. | Pipeline-String nie ohne gst_lint.py aendern. SO-Pfade nur nach Existenz-Check. |
| `core/hardware/camera.py` | 1160 | Pan-Inversion (Zeile 732: `pan_delta = -error_x`). 6x falsch "gefixt". | MINUS IST KORREKT. Nie Vorzeichen aendern. Nie TRACKING_GAIN_PAN ohne Overshoot-Messung. |
| `core/hardware/hailo_manager.py` | 704 | VDevice-Singleton. Zwei VDevices = HAILO_OUT_OF_PHYSICAL_DEVICES(74). | Nie zweites VDevice erstellen. Nie hcl.Device() direkt aufrufen. |
| `core/core_integrator.py` | 868 | 2-Achsen State-Machine (Tension+Dominance), 8+ Consumer lesen. | Keine neuen Achsen. Tick-Rate bei 1Hz lassen. Nur State beeinflussen, nie direkte Actions. |
| `core/voice_pipeline.py` | 2072 | VDevice-Lifecycle + Threading + 6 subprocess Calls. | Kein neues VDevice. Subprocess immer mit timeout. |
| `core/mpo/autonomous_tracker.py` | 1964 | 30+ Case-Branches, schreibt Runtime-State in config/. | Keine States ohne FSM-Diagramm. State-Files gehoeren nach /dev/shm. |
| `core/gui/moloch_unified_panel.py` | 2504 | 30+ tk.after() Callbacks. Race mit Service-Thread. | Service NIE direkt aufrufen, immer IPC. |
| `core/speech/audio_pipeline.py` | 835 | subprocess.Popen (Zeile 158) ohne konsistenten timeout. | Immer timeout setzen. Kein zweiter RTSP-Zugriff (Single-Slot Kamera). |
| `core/inference_engine.py` | 1239 | Legacy-Pfad (nur wenn USE_TAPPAS=0). 860MB RAM. | Wird deprecated. Keine Investition. |
| `core/camera_manager.py` | 1055 | RTSP-Source-Routing. Wird bei USE_TAPPAS=1 uebersprungen. | Kein RTSP oeffnen wenn TAPPAS aktiv (Single-Slot). |
| `core/model_orchestrator.py` | 711 | HEF-Lifecycle Management. VDevice-Race bei configure() waehrend run(). | Modelle NUR ueber diesen Orchestrator laden. |
| `core/perception_engine.py` | 719 | Stage-Machine (IDLE/PERSON/FACE). Kann in FACE steckenbleiben. | Timeout fuer Stage-Transitions einhalten. |
| `core/ipc_router.py` | 163 | SHM Frame-Exchange (/dev/shm/moloch_frame). | Frame-Format nie ohne panel_preview.py Update aendern. |
| `core/hardware/thermal_manager.py` | 859 | Fan/Thermal-Control. subprocess.run() mit shell=True. | Thermal-Throttling nie deaktivieren. |
| `core/ptz_tracker.py` | 160 | Emotionaler Input fuer Core Integrator. Stale-Ref Crash. | CoreIntegrator-Referenz immer validieren. |
| `core/perception/model_scheduler.py` | 152 | Model-Aktivierung. Kein Heartbeat, kann deadlocken. | HailoRT Silent-Failure beachten. |
| `core/memory/episodic_memory.py` | 286 | Qdrant Vector-DB Client. Keine Retry-Logic. | Netzwerk-Timeout beachten, SSD2 Mount pruefen. |
| `core/memory/person_reid.py` | 340 | Person Re-ID via RepVGG. Threshold beeinflusst Tracking direkt. | Threshold nie ohne Regressionstest aendern. |
| `config/settings.json` | ~50 | Kein Schema-Validation. Malformed JSON = Service-Crash. | JSON immer validieren vor Schreiben. Atomic Write nutzen. |

---

## Dependency-Graph (ROT-Dateien)

```
moloch_service.py (28 imports)
  |
  +-- tappas_pipeline.py (wenn USE_TAPPAS=1)
  |     +-- perception_frame.py
  |     +-- model_scheduler.py
  |     +-- moloch_event_bus.py
  |
  +-- hailo_manager.py
  +-- camera_manager.py
  +-- model_orchestrator.py
  +-- ipc_router.py
  |
  +-- core_integrator.py (8 Consumer)
  |     +-- voice_pipeline.py
  |     +-- autonomous_tracker.py
  |     +-- spotify_controller.py
  |     +-- personality_engine.py
  |     +-- moloch_sprache.py
  |     +-- ptz_tracker.py
  |     +-- keyword_handler.py
  |     +-- moloch_service.py (zirkulaer, lazy import Zeile ~96)
  |
  +-- voice_pipeline.py
  +-- thermal_manager.py (indirekt, via Status)
```

**Kritisch**: core_integrator.py ist der meistgelesene ROT-File mit 8 Consumern.
Aenderungen hier kaskadieren ueberall hin. Zirkulaere Abhaengigkeit mit
moloch_service.py wird durch lazy import in Methoden-Body geloest.

---

## Bekannte Bugs mit Zeilennummern

| Bug | Datei | Zeile | Status | Anmerkung |
|-----|-------|-------|--------|-----------|
| Pan-Vorzeichen | camera.py | 732 | GELOEST | NICHT ANFASSEN. 6x falsch "gefixt" laut Git. |
| ArcFace Embedding-Inkompatibilitaet | tappas_pipeline.py | systemisch | OFFEN | GStreamer != HailoRT Embeddings. Threshold aendern ist nutzlos. |
| Runtime-State in Git | autonomous_tracker.py | 42-43 | OFFEN | last_face_position.json + learned_patrol_positions.json gehoeren nach /dev/shm |
| subprocess ohne timeout | audio_pipeline.py | 158 | OFFEN | Zombie-Prozesse moeglich |
| subprocess ohne timeout | music_visualizer.py | 228 | OFFEN | Buffer-Overrun bei FFT-Lag |
| shell=True Injection | moloch_console.py | 514, 518 | OFFEN | OWASP-Risiko |
| shell=True Injection | audio_manager.py | 552 | OFFEN | OWASP-Risiko |
| Non-atomic JSON writes | calibration_engine.py | 97, 851, 913 | OFFEN | Partial Write bei Crash |
| Non-atomic JSON writes | identity_manager.py | 72 | OFFEN | Partial Write bei Crash |
| Suchrichtung asymmetrisch | ptz_orchestrator.py | TBD | OFFEN | Links verschwinden → sucht nicht links |
| ArcFace Threshold zu niedrig | settings.json | TBD | OFFEN | Erkennt alles als Markus (0.45) |
| Kamera Hot-Plug | camera_manager.py | TBD | OFFEN | Stecker raus/rein → nur Reboot hilft |
| Tracking Gains zu hoch | settings.json | TBD | OFFEN | TRACKING_GAIN_PAN=0.7 → Ueberschwinger |

---

## GELB — Bug Risk (68 Dateien, ~35k LOC)

### Wichtigste GELB-Dateien

| Datei | LOC | Risiko |
|-------|-----|--------|
| `core/personality/personality_engine.py` | ~1014 | Speech-Generation, DriftRule, imports core_integrator |
| `core/personality/mood_engine.py` | ~276 | Mood→Musik-Tempo Mapping, Stale-Referenz moeglich |
| `core/personality/behavior_rules.py` | ~268 | Rule-Weighting ohne Bounds-Check |
| `core/gui/popups/popup_audio.py` | 1469 | Groesstes Popup, 6x subprocess, UI-Race |
| `core/gui/popups/popup_hardware.py` | 925 | /sys-Reads ohne Retry |
| `core/gui/popups/popup_gallery.py` | 307 | Kein Image-Cache-Eviction → Memory-Leak |
| `core/gui/popups/popup_dashboard.py` | 245 | Kein Datenpunkt-Limit → Memory-Leak |
| `core/gui/popups/popup_supervisor.py` | 318 | subprocess kill ohne SIGTERM |
| `core/gui/panel_preview.py` | 413 | cv2.resize Letterbox, Framerate hardcoded |
| `core/gui/panel_ptz.py` | 287 | Manual Override ohne Rate-Limiting |
| `core/gui/panel_talk_chat.py` | 348 | Claude API Token-Counting fragil |
| `core/gui/panel_avatar.py` | 1132 | Framerate CPU-abhaengig, kann stuttern |
| `core/spotify_controller.py` | 1342 | Token-Refresh fragil, Suche O(n) ueber 24k Tracks |
| `core/speech/hailo_whisper.py` | 242 | Shared VDevice, Embeddings inkompatibel mit Face-Training |
| `core/console/moloch_console.py` | 1432 | 10+ subprocess mit shell=True |
| `core/autonomy/decision_engine.py` | 407 | Konfligierende Inputs unbehandelt |
| `core/autonomy/atmosphere_controller.py` | 286 | subprocess haengt ohne timeout |
| `core/autonomy/night_cycle.py` | 318 | Kein NTP-Check, Clock-Drift Problem |
| `core/awareness/room_map.py` | 267 | Koordinaten ohne Bounds-Check |
| `core/awareness/motion_analyzer.py` | 289 | Threshold fragil bei Lichtwechsel |
| `core/awareness/activity_analyzer.py` | 301 | Kein Timeout fuer Stuck-States |
| `core/audio/wifi_mic.py` | 187 | TCP statt UDP (laut MEMORY.md TODO) |
| `core/audio/music_visualizer.py` | 246 | FFT-Buffer-Overrun bei Lag |
| `core/hardware/rgb_led_controller.py` | 203 | pigpio-Daemon Abhaengigkeit ohne Timeout |
| `core/hardware/camera_cloud_bridge.py` | 1209 | eWeLink API 30s Timeout, PTZ haengt |
| `core/hardware/ptz_calibration.py` | 288 | Stale ptz_limits.json bricht move_absolute() |
| `core/vision/face_database.py` | 403 | ArcFace Threshold 0.45 zu niedrig |
| `core/vision/identity_manager.py` | 262 | IDs flackern ohne Temporal-Smoothing |
| `core/vision/gst_hailo_detector.py` | 568 | Kein RTSP-Reconnect bei Stream-Loss |
| `core/vision/gst_hailo_pose_detector.py` | 1061 | pose_postprocess.so Validation fehlt |
| `core/longterm_memory.py` | 354 | 60s Persistence-Tick, kein Crash-Recovery |
| `core/deepseek_client.py` | 198 | Kein Exponential Backoff bei 429 |
| `core/music/music_memory.py` | 234 | Append-only JSON waechst unbegrenzt |
| `core/tts/tts_manager.py` | 290 | subprocess shell=True (Zeile 552) |
| `core/ptz_arbiter.py` | 315 | Race bei manual_override waehrend Timeout-Tick |
| `core/mpo/ptz_orchestrator.py` | 198 | Suchrichtungs-Bug (asymmetrisch links/rechts) |
| `core/status.py` | 735 | Stale Timestamps bei langsamer Update-Loop |
| `core/action_bridge.py` | 268 | Concurrent JSON-Write Race |
| `core/moloch_event_bus.py` | 187 | Subscriber-Crash kann Event-Queue korrumpieren |

### Weitere GELB-Dateien (ohne Detail)

Alle weiteren `core/gui/panel_*.py`, `core/gui/popups/popup_*.py`,
`core/autonomy/*.py`, `core/awareness/*.py`, `core/memory/*.py`,
`core/tts/selection/*.py`, `core/audio/audio_manager.py` (717 LOC),
`core/vision/unified_pipeline.py` (780 LOC, Legacy-Overlap mit TAPPAS),
`core/calibration_engine.py` (927 LOC),
`core/diagnostics.py`, `core/dashboard.py`, `core/environment_watcher.py`,
`core/capability_monitor.py`, `core/daily_learner.py`, `core/timeline.py`.

---

## GRUEN — Sicher editierbar (25+ Dateien, ~2.8k LOC)

| Datei | LOC | Anmerkung |
|-------|-----|-----------|
| `core/gui/panel_styles.py` | 198 | Reine Konstanten (Farben, Fonts). 0 Logik, 0 Imports von core. |
| `core/vision/emotion_detector.py` | 95 | Stub, unbenutzt in aktueller Pipeline |
| `core/vision/age_gender_detector.py` | 98 | Stub, unbenutzt |
| `core/vision/gesture_detector.py` | 406 | Hand-Gesten Stub, nicht in Autonomie integriert |
| `core/vision/hand_gesture_detector.py` | 184 | Hand-Landmarks Stub, unbenutzt |
| `core/net/internet_bridge.py` | 256 | Isolierter HTTP-Wrapper, keine Core-Abhaengigkeiten |
| `core/sensors/__init__.py` | 18 | Leeres Package, sicher erweiterbar |
| `core/agents/__init__.py` | 18 | Agent-System Placeholder, nicht integriert |
| `core/tts/config/voices.py` | 156 | Voice-Config Konstanten, 0 Logik |
| `core/eye_viewer.py` | 312 | Eye-Visualisierung Placeholder, unvollstaendig |
| `scripts/*` | varies | Alle Test/Diagnose-Utilities, nicht Teil des Service |
| `docs/*` | varies | Dokumentation, kein Runtime-Code |

---

## Historische Hotspots (Git-Analyse)

Die meistgeaenderten Dateien der letzten 30 Tage (= hoechstes Instabilitaets-Risiko):

1. **tappas_pipeline.py** — 30+ Commits in 30 Tagen (Valve, Cropper, Buffer, SO-Migration)
2. **autonomous_tracker.py** — 15+ Fixes in 3 Monaten (Pan-Sign, Coast-Bug, Orphan-Kill)
3. **config/settings.json** — 7+ Threshold-Aenderungen (ArcFace Ping-Pong)
4. **camera.py** — 6+ Pan-Vorzeichen-Korrekturen (war jedes Mal schon korrekt)
5. **moloch_service.py** — Regelmaessig bei Feature-Additions betroffen

---

## Wiederkehrende Fehler-Muster

| Muster | Haeufigkeit | Beschreibung |
|--------|-------------|--------------|
| Threshold Ping-Pong | 4+ Commits | Gleicher Wert wird hin-und-her geaendert statt Root Cause zu fixen |
| Custom/System SO Spirale | 3x hin-her | SO-Typ wechseln ohne zu verstehen warum es crashed |
| Shotgun Surgery | 5+ Commits | 3+ Subsysteme in einem Commit, Rollback unmoeglich |
| "Fix what ain't broken" | 6+ Commits | Pan-Vorzeichen war korrekt, wurde trotzdem "gefixt" |
| Config-State in Git | Jeder BACKUP | Runtime-Dateien verschmutzen Commit-Historie |
| Guard statt Fix | 10+ Commits | Workaround statt Root-Cause-Analyse |
