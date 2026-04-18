# Agent Handoff — 2026-04-18 (Session 18 — HailoRT 5.3.0 Upgrade, lokales LLM live)
# Letzter Commit: 277cf10 | Audit: 5/5 PASS | FPS: 20.0 | RAM: 35.9% | NPU-FW: 5.3.0

---

## SESSION 18 — LOKALES LLM LEBT (HailoRT 5.1.1 → 5.3.0)

**User-Wunsch:** Moloch soll lokal sprechen — hailo-ollama statt DeepSeek Cloud.

**Ausgangslage:** Vorige Session (17) hatte 5.3.0-Debs in `/home/molochzuhause/hailo-install/` +
komplettes 5.1.1-Rollback-Set in `~/Downloads/hailo_backup/` vorbereitet, aber nie
installiert. `logs/hailo_ollama_stability_2026-04-17.md` dokumentierte: R1 SEGV
deterministisch, Qwen-Instruct Error 74 (kein SHARED VDevice in 5.1.1).

**Durchbruch-Erkenntnis:** hailo-ollama 5.3.0 kennt die Env-Var
`HAILO_OLLAMA_VDEVICE_GROUP_ID=SHARED` (ab 5.3.0 eingefuehrt, nicht 5.1.1). Das
ist der fehlende Hebel aus der Vor-Analyse.

### Durchgefuehrte Schritte

1. **Paket-Swap (apt atomic):** `h10-hailort*` (RPi-Fork) raus, `hailort*` (Hailo Standard) rein
   - Entfernt: h10-hailort, h10-hailort-pcie-driver, python3-h10-hailort, hailo-h10-all, rpicam-apps-hailo-postprocess, hailo-tappas-core 5.1.0
   - Installiert: hailort 5.3.0, hailort-pcie-driver 5.3.0 (DKMS), hailo-tappas-core 5.3.0, hailo-gen-ai-model-zoo 5.3.0
   - Python-Wheel: `hailort-5.3.0-cp313-cp313-linux_aarch64.whl` via `pip --break-system-packages --no-deps`
2. **Env-Var:** `Environment=HAILO_OLLAMA_VDEVICE_GROUP_ID=SHARED` in
   `/etc/systemd/system/hailo-ollama.service` (Zeile 9, nach OLLAMA_KEEP_ALIVE).
   Backup: `hailo-ollama.service.bak_20260418_preupgrade`.
3. **Reboot** → NPU-Firmware auto-upgraded auf 5.3.0 (fitImage 10.9→11.4 MB).
   Kernel-Modul hailo1x_pci/5.3.0 via DKMS frisch gebaut.
4. **XDG-Fix:** hailo-ollama 5.3.0 schreibt nach `~/.local/share/hailo-ollama/models/`
   (nicht `/usr/share/...`). Pfade angelegt + Manifeste aus Paket kopiert,
   sonst crasht Pull am Ende mit `filesystem_error: cannot rename`.
5. **Pull qwen2.5:1.5b** (2.3 GB) — 6 Minuten ueber dev-public.hailo.ai.
6. **Bridge-Code:** `core/autonomy/local_llm_bridge.py` Default-Modell
   `deepseek_r1_distill_qwen:1.5b` → `qwen2.5:1.5b`, `LLM_MODE_DEFAULT`
   `cloud_only` → `local_first`. Timeouts auf Qwen-Realwerte.

### Verifikation (alles PASS)

| Test | Ergebnis |
|------|----------|
| Firmware-Version | `5.3.0 (release,app)` via `hailortcli fw-control identify` |
| Kernel-Modul | `hailo1x_pci/5.3.0` via DKMS |
| Python-API | `hailo_platform` + `hailo_platform.genai.Speech2Text` importierbar |
| hailo-ollama Log | `Using VDevice group_id: SHARED` (bestaetigt!) |
| Inference qwen2.5:1.5b | 3x PASS, NPU 2.85s / 6.83s / 4.03s, 0 SEGV, 0 ABRT |
| TAPPAS parallel | FPS 20.0 bleibt stabil, alle 8 Worker errors=0 |
| moloch_audit.py --auto | 5/5 PASS |
| Moloch Face-ID | markus erkannt (Sim 0.53) nach Reboot |

### Commits dieser Session

| Commit | Inhalt |
|--------|--------|
| `806eae0` | bridge: qwen2.5:1.5b default, local_first enabled |
| `277cf10` | config: llm_mode=local_first — hailo-ollama stabil |

### Bekannte Einschraenkungen (nicht-blocking)

- **Modell-Pull erfordert XDG-Pfade**: `~/.local/share/hailo-ollama/models/{blob,manifests}`
  muessen existieren BEVOR man pullt — sonst SIGABRT beim finalen Rename.
  Paket-postinst legt die nicht an. Fuer Nutzer: Script oder docs.
- **R1 (deepseek_r1:1.5b) ungetestet auf 5.3.0** — in 5.1.1 deterministischer
  SEGV, in 5.3.0 evtl. gefixt, aber nicht verifiziert. Qwen2.5 reicht vorerst.
- **`moloch.service` ExecStartPre macht `pkill -9 hailo-ollama`** — war fuer
  das alte VDevice-Konflikt-Problem. Mit SHARED nicht mehr noetig, kann spaeter
  entfernt werden (fuehrt aktuell zu 30s Verzoegerung bei Moloch-Restart bis
  hailo-ollama wieder erreichbar ist).
- **Modell-Store auf Root-Dateisystem** (`/` hat 384 GB frei, aber
  `/mnt/moloch-data` haette 426 GB). Bei mehreren Modellen evtl. migrieren.

### Rollback-Pfad (falls noetig)

Komplettes 5.1.1-Set in `/home/molochzuhause/Downloads/hailo_backup/`:
```bash
sudo systemctl stop moloch hailo-ollama
sudo apt install -y ~/Downloads/hailo_backup/{h10-hailort*,hailo-tappas-core_5.1.0*,python3-h10-hailort*,python3-hailo-tappas*,hailo_gen_ai_model_zoo_5.1.1*,hailo-models*}.deb
sudo reboot
```
Fuehrt zurueck auf 5.1.1, aber dann wieder Error 74 bei paralleler Nutzung.

---

## SESSION 17 — MOLOCH UNABHÄNGIG VON DISPLAY/GUI

**User-Wunsch:** Moloch soll immer aktiv bleiben, auch wenn Bildschirm aus oder
GUI-Fenster geschlossen. "Nur Service muss immer laufen."

**Befund der Analyse:** Backend war bereits entkoppelt. `moloch.service` ist
System-Service (`WantedBy=multi-user.target`), kein `graphical.target`-Bezug,
kein DISPLAY im Environment. `panel_main._on_close()` (Zeile 579-591) stoppt nur
Preview+Avatar, sendet keinen IPC-Befehl. SHM-Frame-Writer läuft unabhängig von
Reader. → Der wahrgenommene "Tracking pausiert"-Effekt war Wayland Screen-
Blanking + eingefrorene Preview-Anzeige, nicht echter Service-Stop.

### Änderungen dieser Session

| Datei | Zweck | Agent |
|-------|-------|-------|
| `scripts/verify_headless_runtime.py` (NEU) | 5-Min Status-Monitor (CSV) zum Beweisen dass Service durchläuft | watchdog |
| `~/.config/labwc/autostart` | `wlopm --on *` beim Desktop-Login, swayidle ausgeklammert | hardware |
| `~/.config/autostart/moloch-no-sleep.desktop` (NEU) | XDG-Autostart mit Wayland+X11 Fallback | hardware |
| `/boot/firmware/cmdline.txt` | `consoleblank=0` angehängt (TTY-Blanking aus) | hardware |
| `/etc/systemd/system/moloch.service` | `KillMode=process`, `RestartSec=10→5` (Restart=always war schon da) | service |

**Backup der moloch.service:** `/etc/systemd/system/moloch.service.bak_20260418_105901`
**Backup cmdline.txt:** `/tmp/cmdline.txt.bak` (vor Reboot sichern!)

### Regel für Markus
> **Screen aus ≠ Moloch aus.** Bei Zweifel: `mcp__moloch__moloch_status()` —
> nicht GUI öffnen. Die GUI ist nur Anzeige, nicht das Gehirn.

### Verifikation nach Reboot
1. `systemctl is-active moloch` → active
2. `systemctl show moloch -p Restart,RestartSec,KillMode` → always/5s/process
3. `mcp__moloch__moloch_audit()` → 62/62 PASS
4. `mcp__moloch__moloch_status()` → FPS ≥ 18
5. Realer Test: `python3 ~/moloch/scripts/verify_headless_runtime.py --duration 300`
   während Markus GUI öffnet/schließt und Bildschirm aus/an schaltet. CSV in
   `logs/verify_headless_<ts>.csv`. Erwartung: FPS-Stalls=0, status_age_ms<2000.

### Offen aus Vorsession (unverändert)
- B2 Moloch halluziniert Quellen → deepseek
- B4 News bei generischen Anfragen veraltet → autonomy
- Slider-Drift yolo_conf Max auf 0.7 → gui
- YOLOWorldWorker ~70 % integriert (Frequency/PFrame/IPC fehlen) → vision
- IPC Race poll_commands() löscht Datei vor Verarbeitung → service

---

## SESSION 16 — AUDIT & RACE-FIX (archiviert, Commit 0cb50d4 | 62/62 PASS)

System-Zustand: grün. FPS 20, RAM 39 %, 0 SEGV, 0 Worker-Errors, Face-ID
erkennt `markus` (Sim ~0.55-0.64). LLM/hailo-ollama antwortet, alle 8 NPU-Worker
aktiv (Activity/Depth/Face/Hand/PersonAttr/Pose/ReID/YOLOWorld).

### Gefundener Fehler
MCP-Audit zeigte **1 FAIL: "PerceptionMemory initialisiert — Kein Init-Log
gefunden"** bei ansonsten 61/62 PASS.

**Root Cause:** Race zwischen Service-Start und Audit-Test.
- Service-Start: 12:14:29 (Init-Log geschrieben)
- Vorheriger Audit-Lauf: 12:12:11 (2 min davor)
- Test `scripts/moloch_audit.py:797-809` greppt journalctl nach
  `PerceptionMemory.*Initialisiert` — Zeile existierte zu dem Zeitpunkt noch nicht.

### Fix (Commit 0cb50d4)
`scripts/moloch_audit.py` Test um Status-Fallback erweitert: wenn
journalctl nichts findet, prüft er `/dev/shm/moloch_status.json` auf
`face_id` / `person_present` / `active_models`. Wenn PerceptionMemory
Output liefert, läuft es per Definition. Bei totem Modul bleibt es FAIL.

Keine ROT-Dateien angefasst, kein Service-Restart nötig.

### Rollback-Obduktion (daf3e76, 2026-04-13)
"Kamera-Steuerung + Bild waren kaputt" nach Vision-Pause-Experiment
(`pause_for_llm` / `resume_after_llm` + disabled_workers). Nicht wiederholen
ohne neuen Plan. Danach `a701a38` → spontane Kommentare deaktiviert.

### Offen (aus voriger Session)
- B2 Moloch halluziniert Quellen → deepseek
- B4 News bei generischen Anfragen veraltet → autonomy
- Slider-Drift yolo_conf Max auf 0.7 → gui
- YOLOWorldWorker ~70 % integriert (Frequency/PFrame/IPC fehlen) → vision
- IPC Race poll_commands() löscht Datei vor Verarbeitung → service
- T2 Agent-Events nicht publiziert (rein architekturell, kein Runtime-Impact)

---

## SESSION-ERGEBNISSE (9 Commits)

| Fix | Commit | Datei |
|-----|--------|-------|
| HandWorker-Dispatch bei Nahaufnahme + pose_age Tolerance | 109ce71 | tappas_pipeline.py |
| PoseWorker conf_thresh 0.3->0.2, HandWorker wrist-vis 0.15->0.1 | 8955786 | pose_worker.py |
| panel_preview Pose-Keypoint Visibility 0.2->0.1 | a387ad0 | panel_preview.py |
| Night Cycle startet ab 23:00 + echte Daten in LLM-Reflexion | f41c05c | night_cycle.py |
| Shutdown-Cleanup loggt Fehler statt sie zu verschlucken | 3e884f9 | moloch_service.py |
| NPU-Extras VDevice/VLM Release Fehler-Logging | 2a6a553 | npu_extras.py |
| head_pitch/head_yaw in TAPPAS-PFrame + Worker-Health Logging | 8012b76 | tappas_pipeline.py |
| deepseek_client.py entfernt (230 Zeilen toter Code) | 5f69a62 | deepseek_client.py |
| ocr_texts in TAPPAS-PFrame setzen | 2c9c80d | tappas_pipeline.py |

### Wichtige Aenderungen:
- **Pose+Hand Landmarks**: 3 Root Causes behoben (HandWorker-Guard, pose_age, Thresholds)
- **Night Cycle**: Startet jetzt um 23:00 statt 00:00. LLM-Reflexion mit echten Tagesdaten.
- **Shutdown Logging**: 15x except:pass -> logger.warning/error (moloch_service + npu_extras)
- **PFrame komplett**: head_pitch/head_yaw + ocr_texts jetzt im TAPPAS-Modus verfuegbar
- **Systemscan**: 7 versteckte Bugs gefunden und behoben (Key-Mismatches, stille Fehler, toter Code)

---

## OFFENE BUGS

- **B2**: Moloch halluziniert Quellen ("Laut Suchergebnissen")
- **B4**: News bei generischen Anfragen veraltet (Google News RSS)
- **Slider-Drift**: yolo_conf Slider-Max auf 0.7 begrenzen (GUI)
- **YOLOWorldWorker**: Integration ~70% (keine Frequency, kein PFrame-Ort, kein IPC)
- **IPC Race**: poll_commands() loescht Datei VOR Verarbeitung
- **T2**: Agent-Events (build/test/review/chaos) nie publiziert — geplante Architektur, kein Runtime-Impact

---

## WAS BEREITS GEFIXT IST — NICHT NOCHMAL ANFASSEN

- Pose/Hand Landmarks (109ce71, 8955786, a387ad0)
- Night Cycle Date+LLM (f41c05c)
- Shutdown Logging (3e884f9, 2a6a553)
- head_pitch/head_yaw + ocr_texts (8012b76, 2c9c80d)
- Worker-Health Error-Dict (8012b76)
- deepseek_client.py entfernt (5f69a62)
- Pan-Vorzeichen, ArcFace, hailooverlay, Status-JSON Deadlock
- keywords.json Komma, PTZ Error-Handling
