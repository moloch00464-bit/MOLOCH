---
name: moloch-dev
description: Entwicklungs-Skill fuer M.O.L.O.C.H. — Pre/Post-Flight Checks, NEVER-Regeln, Code Templates, Debugging, Deploy-Workflow. Nutze bei jeder Code-Aenderung.
allowed-tools: Read Grep Glob Bash Edit Write
---

# M.O.L.O.C.H. Entwicklungs-Skill

---

## SESSION-START PROTOKOLL

1. `CLAUDE.md` lesen (Systemregeln)
2. `~/moloch/logs/agent_handoff.md` lesen (letzte Session)
3. Relevantes Agent-MD lesen (je nach Domain)
4. `git status` — bei dirty tree STOPPEN
5. Risiko-Stufe bestimmen (ROT/GELB/GRUEN)

---

## NEVER-DO REGELN (durch Hooks erzwungen)

Vollstaendige Liste in [never-rules.md](never-rules.md).
Die wichtigsten:
- **NEVER 1**: GStreamer-String nicht blind aendern (SEGV)
- **NEVER 2**: Pan-Vorzeichen nicht aendern (`pan_delta = -error_x` ist KORREKT)
- **NEVER 4**: Nicht mehrere ROT-Dateien in einem Commit
- **NEVER 6**: JSON immer atomic schreiben (tempfile + os.replace)
- **NEVER 8**: Kein shell=True in subprocess

---

## PRE-FLIGHT (VOR Code-Aenderung)

```bash
git status                    # Muss clean sein
python3 -c "import core.[modul]"  # Syntax OK?
systemctl is-active moloch    # Service laeuft?
# Bei ROT: User fragen
git add [datei] && git commit -m "BACKUP vor [was]"
```

## POST-FLIGHT (NACH Code-Aenderung)

```bash
python3 -c "import core.[modul]"      # Syntax OK?
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
sudo systemctl restart moloch && sleep 5
systemctl is-active moloch             # Laeuft noch?
python3 ~/moloch/moloch_audit.py --auto  # PASS?
# Bei FAIL: git checkout -- [datei], Root-Cause analysieren
```

---

## DEPLOY-WORKFLOW

```bash
git push origin [branch]
ssh molochzuhause@192.168.178.30 "cd ~/moloch && git pull && \
  find core -name __pycache__ -exec rm -rf {} + 2>/dev/null && \
  sudo systemctl restart moloch"
sleep 5
ssh molochzuhause@192.168.178.30 "systemctl is-active moloch"
```

Oder via MCP: `moloch_service(action="restart")` + `moloch_status()`

---

## CODE-TEMPLATES

Fuer vollstaendige Templates siehe [templates.md](templates.md):
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

---

## HANDOFF (bei ~85% Kontext)

Datei: `~/moloch/logs/agent_handoff.md`

```markdown
# Agent Handoff — [Datum]
## Aktueller Task: [was]
## Erledigt: [Liste]
## Offen: [Liste]
## Geaenderte Dateien: [Liste]
## Service-Status: active/inactive
## Blocker: [falls vorhanden]
```
