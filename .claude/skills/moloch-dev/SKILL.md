---
name: moloch-dev
description: Entwicklungs-Skill fuer M.O.L.O.C.H. — Pre/Post-Flight Checks, NEVER-Regeln, Code Templates, Debugging, Deploy-Workflow. Nutze bei jeder Code-Aenderung.
allowed-tools: Read Grep Glob Bash Edit Write
---

# M.O.L.O.C.H. Entwicklungs-Skill — Codename: LOKOMOTIVE

**LOKOMOTIVE aktiv.** — Dieser Skill gilt fuer JEDE Code-Aenderung.

---

## LOKOMOTIVE-PRINZIP — Durchfahren ohne Stopp

Eine Lokomotive haelt nicht an jeder Kreuzung. Claude faehrt durch.

**Keine Rueckfrage bei:**
- GRUEN-Dateien → sofort umsetzen
- GELB-Dateien → ankuendigen, dann sofort umsetzen
- Kleinen Korrekturen, Bugfixes, Docs, Config-Anpassungen
- Wenn der Plan genehmigt wurde → eigenstaendig bis zum Ende durcharbeiten
- Agent/Sub-Agent-Auswahl → Claude entscheidet selbst (dafuer gibt es die Agenten)

**Einmal fragen, dann durchfahren:**
- ROT-Dateien → EINMAL kurz ankuendigen, auf Nicken warten, dann eigenstaendig
- Unbekannte Abhaengigkeit → einmal klaeren, dann weiter

**Vollstaendiger STOPP nur bei:**
- Audit FAIL (moloch_audit zeigt FAIL)
- Destructive Git-Op (reset --hard, force-push main)
- Mehr als 5 ROT-Dateien gleichzeitig
- Echter Widerspruch in den Anforderungen

**Merksatz:** Markus geht aus dem Zimmer, kommt zurueck — Arbeit ist erledigt.
Kein "Darf ich?", kein "Soll ich?", kein "Bestaetigen Sie bitte."
Die Agenten und Sub-Agenten managen die Domain-Grenzen — Claude muss nicht staendig nachfragen.

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

### WARTEN AUF PI-BEREITSCHAFT (nach Reboot)

Der Pi braucht ~35 Sekunden bis SSH erreichbar ist.
TAPPAS-Pipeline + NPU-Worker brauchen weitere ~20-30 Sekunden.
**Gesamtwartezeit: 60 Sekunden** — dann aktiv pruefen, NICHT blind warten.

**Pruef-Logik (alle 15 Sek wiederholen bis OK oder Timeout 90 Sek):**
```
Versuch 1 (nach 30 Sek): moloch_status()
  → MCP-Timeout oder FPS = 0?  → 15 Sek warten, nochmal
  → FPS > 0 UND Service aktiv? → Naechster Schritt

Versuch 2: moloch_npu_workers()
  → Worker = 0 oder Errors?    → 15 Sek warten, nochmal
  → Alle Worker geladen?        → Naechster Schritt

Versuch 3: moloch_status() nochmal
  → FPS stabil (> 10)?         → Pi ist BEREIT
  → FPS = 0 nach 90 Sek total? → moloch_logs(filter_str="ERROR") → Fehler melden
```

**Indikatoren fuer NICHT-BEREIT (noch warten):**
- `moloch_status()` gibt MCP-Timeout zurueck
- FPS = 0.0 (Service laeuft, Pipeline noch nicht initialisiert)
- NPU-Worker-Count = 0 (HEF noch nicht geladen)
- Frame Age > 5.0s (Pipeline haengt)

**Indikatoren fuer BEREIT:**
- FPS > 10 (TAPPAS laeuft)
- Alle Worker geladen (arcface, scrfd, yolo mindestens)
- Frame Age < 1.0s
- Kein ERROR im Log der letzten 30 Sek

### VERIFIKATION (wenn BEREIT-Status erreicht)

```
1. moloch_audit()            # Alle Tests PASS?
2. moloch_snapshot()         # Kamera-Feed visuell OK?
3. git log --oneline -3      # Commits noch da?
```

**Bei Audit-FAIL nach Reboot:** `moloch_logs(n=50, filter_str="ERROR")` → Ursache finden, NICHT weitermachen.

---

## MASTER-AUDIT — Model-Agnostic Check via Aider/DeepSeek

**Zweck:** Pruefen ob LOKOMOTIVE-Briefing auch fuer andere Modelle (nicht nur Claude) verstaendlich ist.
Laeuft NICHT automatisch — nur on-demand, z.B. nach groesseren CLAUDE.md-Aenderungen oder neuen Agenten.

**Wann ausfuehren:**
- Nach groesseren Aenderungen an CLAUDE.md
- Nach Hinzufuegen neuer Agenten
- Als periodischer Sanity-Check (z.B. einmal pro Woche)
- Wenn Zweifel besteht ob das Briefing modell-agnostisch genug ist

**Ausfuehren (aus ~/moloch Verzeichnis):**
```bash
# Voraussetzung: DEEPSEEK_API_KEY gesetzt (einmalig via setx auf Windows)
# .aider.conf.yml laedt CLAUDE.md automatisch

# Schnell-Check (3 Kernfragen):
aider --message "1) Was ist LOKOMOTIVE? 2) Welcher Agent fuer BBox/Landmarks? 3) Was sind die ersten 3 Schritte vor Code-Aenderung?" --no-git

# Tiefer Agent-Routing-Test:
aider --message "Ich habe einen Bug: Pose-Keypoints erscheinen alle oben links. Welchen Agenten laedt du, was ist dein erster Schritt?" --no-git

# NEVER-Regeln-Test:
aider --message "Liste alle NEVER-Regeln auf und erklaere warum NEVER 2 (Pan-Vorzeichen) existiert." --no-git
```

**Erwartete Antwort bei PASS:**
- Beginnt mit "LOKOMOTIVE aktiv."
- Nennt coordinates-Agent fuer BBox/Keypoints
- Nennt moloch_status() + moloch_npu_workers() als erste Schritte
- NEVER-Regeln vollstaendig aufgelistet

**Bei FAIL:** CLAUDE.md ueberarbeiten — das betreffende Konzept ist nicht klar genug formuliert.

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
