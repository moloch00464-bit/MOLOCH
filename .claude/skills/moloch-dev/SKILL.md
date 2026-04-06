---
name: moloch-dev
description: Entwicklungs-Skill fuer M.O.L.O.C.H. — Pre/Post-Flight Checks, NEVER-Regeln, Code Templates, Debugging, Deploy-Workflow. Nutze bei jeder Code-Aenderung.
allowed-tools: Read Grep Glob Bash Edit Write
---

# M.O.L.O.C.H. Entwicklungs-Skill — Codename: LOKOMOTIVE

**LOKOMOTIVE aktiv.** — Dieser Skill gilt fuer JEDE Code-Aenderung.

---

## SESSION-START PROTOKOLL

1. `moloch_status()` — loest Session-Lock, zeigt System-Status
2. `moloch_npu_workers()` — Worker-Health pruefen
3. `git status` — bei dirty tree STOPPEN
4. `logs/agent_handoff.md` lesen — was war zuletzt?
5. Risiko-Stufe bestimmen (ROT/GELB/GRUEN)
6. **Agent spawnen:** `/moloch-agent` Skill → richtigen Agenten laden
7. `touch /tmp/moloch_agent_[name]` — Lock setzen (PFLICHT vor Edit!)

---

## NEVER-DO REGELN (alle 12 — durch Hooks erzwungen)

| # | Regel |
|---|-------|
| 1 | GStreamer-Pipeline-String NICHT blind aendern (SEGV bei Typo) |
| 2 | Pan-Vorzeichen NICHT aendern (`pan_delta = -error_x` ist KORREKT) |
| 3 | ArcFace-Threshold NICHT erhoehen (Root Cause = Embedding-Inkompatibilitaet) |
| 4 | NICHT mehrere ROT-Dateien in einem Commit |
| 5 | subprocess IMMER mit timeout=30 |
| 6 | JSON IMMER atomic schreiben (tempfile + os.replace) |
| 7 | Runtime-State NICHT committen (last_face_position.json etc.) |
| 8 | KEIN shell=True in subprocess |
| 9 | HailoRT: uint8 vs float32 VOR Inferenz pruefen |
| 10 | KEIN np.ndarray Type-Hint in moloch_service.py |
| 11 | __pycache__ nach Code-Aenderung loeschen |
| 12 | NICHT im Worktree Service testen (laeuft von ~/moloch/) |

---

## PRE-FLIGHT (VOR Code-Aenderung)

```bash
git status                           # Muss clean sein
python3 -c "import core.[modul]"     # Syntax OK?
# MCP: moloch_status() + moloch_npu_workers()
# Bei ROT-Datei: einmal User fragen, dann durcharbeiten
git add [datei] && git commit -m "BACKUP vor [was]"
touch /tmp/moloch_agent_[name]       # Agent-Lock setzen!
```

## POST-FLIGHT (NACH Code-Aenderung)

```bash
python3 -c "import core.[modul]"     # Syntax OK?
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
# MCP: moloch_service(action="restart")
# MCP: moloch_audit()  → bei FAIL: git checkout -- [datei], STOPP
rm /tmp/moloch_agent_[name]          # Agent-Lock freigeben
git add [datei] && git commit -m "..."
git push
```

---

## DEPLOY (NUR via MCP — kein SSH!)

```
moloch_service(action="restart")   # Service neustarten
moloch_status()                    # FPS + Status pruefen
moloch_audit()                     # 54 Tests — alle PASS?
```

**NIEMALS:** `ssh molochzuhause@...` — MCP ist der einzige Weg!

---

## REBOOT-ENTSCHEIDUNG — Service-Restart vs. Pi-Reboot

**Faustregel:** Service-Restart reicht fuer Python-Code. Pi-Reboot bei System-Level-Aenderungen.

### Nur Service-Restart noetig (`moloch_service(action="restart")`)
- Python-Code geaendert (core/*.py, gui/*.py, etc.)
- Config-Dateien geaendert (config/*.json)
- Agent/Skill-Dateien geaendert (.claude/*)
- Normalfall bei allen GRUEN/GELB/ROT-Dateien

### Pi-Reboot PFLICHT (`sudo reboot` via SSH)

| Was geaendert wurde | Warum Reboot |
|---------------------|-------------|
| `moloch.service` (systemd Unit) | systemd laedt Units nur beim Boot neu |
| `~/.profile` (Umgebungsvariablen wie `MOLOCH_USE_TAPPAS`) | Profile wird nur bei Login geladen |
| Hailo-Firmware / HailoRT-Update | NPU-Treiber nur per Reboot neu ladbar |
| GStreamer-Plugins installiert/aktualisiert | SO-Caches muessen geleert werden |
| Kernel-Module veraendert (`modprobe`) | Gilt sofort oder nach Reboot je nach Modul |
| Kamera-Hotplug-Problem (Stecker raus/rein → kein Feed) | Bekannter Bug: nur Reboot hilft |
| MCP-Server haengt / kein Snapshot moeglich | Reboot loest MCP-Init-Bug |
| hailo-ollama neu installiert | systemd-Service fehlt noch, Reboot registriert ihn |
| NPU Error 74 bleibt nach Service-Restart | Shared VDevice kaputt — Reboot pflicht |

### REBOOT-PROTOKOLL (VOR dem Reboot)

```
1. git add [alle geaenderten Dateien]
2. git commit -m "BACKUP vor Reboot: [was geaendert]"
3. git push
4. moloch_service(action="stop")   # Sauber beenden
5. SSH: sudo reboot
```

### VERIFIKATION (NACH dem Reboot, ~60 Sek warten)

```
1. moloch_status()           # Service wieder aktiv? FPS > 0?
2. moloch_npu_workers()      # Alle Worker geladen?
3. moloch_audit()            # Alle Tests PASS?
4. moloch_snapshot()         # Kamera-Feed OK?
5. git log --oneline -3      # Commits noch da?
```

**Bei Audit-FAIL nach Reboot:** `moloch_logs(n=50, filter_str="ERROR")` → Ursache finden, NICHT weitermachen.

---

## DATEI-AMPEL

**ROT** (einmal fragen, dann eigenstaendig — Git Backup vorher!):
`moloch_service.py`, `tappas_pipeline.py`, `camera.py`, `hailo_manager.py`,
`core_integrator.py`, `voice_pipeline.py`, `autonomous_tracker.py`,
`audio_pipeline.py`, `ipc_router.py`, `person_reid.py`, `settings.json`

**GELB** (Ankuendigung, kein Warten):
`personality/*.py`, `gui/panel_*.py`, `popups/*.py`, `audio/*.py`,
`ptz_arbiter.py`, `action_bridge.py`, `moloch_console.py`, `moloch_mcp_server.py`

**GRUEN** (sofort, kein Dialog):
`scripts/*`, `docs/*`, `config/*.json` (ausser settings.json), `.claude/hooks/*`

---

## CODE-TEMPLATES

Vollstaendige Templates in [templates.md](templates.md):
- HailoRT On-Demand Processor
- GStreamer RGB/BGR Konvertierung
- Singleton Pattern
- Safe JSON Write (atomic + NTFS-Fallback)
- Subprocess mit Timeout

---

## DEBUGGING

| Problem | Erste Schritte |
|---------|---------------|
| Service crashed | `moloch_logs(filter_str="ERROR")` + `moloch_dmesg()` |
| Pipeline startet nicht | `moloch_status()` + `moloch_npu_workers()` |
| RAM > 3 GB | `moloch_status()` → restart → Monitor |
| NPU Error 74 | Kein zweites VDevice! `moloch_service(action="restart")` |
| BBox falsch | `moloch_snapshot()` → visuell pruefen |
| Worker Error | `moloch_npu_workers()` → Fehlercount pruefen |
| Hook blockiert | Agent-Lock gesetzt? Domain korrekt? |

---

## HANDOFF (bei ~85% Kontext)

Datei: `~/moloch/logs/agent_handoff.md`

```markdown
# Agent Handoff — [Datum]
## Aktueller Task: [was]
## Erledigt: [Liste mit Commits]
## Offen: [priorisierte Liste]
## Geaenderte Dateien: [Liste]
## Service-Status: active | FPS | Audit X/54
## Blocker: [falls vorhanden]
```
