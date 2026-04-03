# HOOKWIRE — Hooks, Skills, MCP und Agenten verbinden
# Codewort: HOOKWIRE
# Version: 1.0 | Stand: 2026-04-02
# Ziel: Alle Claude Code Features als EIN System verbinden

> Wenn Du dieses Dokument liest, wurde Dir gesagt: "HOOKWIRE".
> Das bedeutet: Richte ALLES hier beschriebene ein. Keine Diskussion.

---

## WAS IST HOOKWIRE?

HOOKWIRE verbindet die bisher isolierten Teile von Claude Code zu einem System:

```
VORHER (isoliert):                    NACHHER (HOOKWIRE):
                                      
Skills ←×→ Hooks                      Skills ←→ Hooks
  ↕×↕        ↕×↕                        ↕         ↕
MCP   ←×→ Agent-MDs                  MCP   ←→ Agent-MDs
                                           ↕
                                      CLAUDE.md (zentrale Quelle)
```

---

## SCHRITT 1: Hooks installieren (aus diesem Repo)

Die Hook-Scripts liegen bereits im Repo unter `.claude/hooks/`.
Die `settings.json` liegt unter `.claude/settings.json`.

Nach `git pull` pruefen:

```bash
# Hook-Scripts muessen ausfuehrbar sein
chmod +x ~/moloch/.claude/hooks/*.sh

# Pruefen ob settings.json die Hooks referenziert
cat ~/moloch/.claude/settings.json | python3 -m json.tool | grep -A2 "hooks"
```

Falls `.claude/` in `.gitignore` steht und die Dateien NICHT ankommen:
```bash
git checkout origin/main -- .claude/settings.json .claude/hooks/ .claude/skills/
```

---

## SCHRITT 2: Hooks DYNAMISCH machen (ROT-Liste aus moloch-dev.md lesen)

Aktuell sind die ROT-Dateien in `pre-edit-check.sh` hart kodiert.
Das muss geaendert werden: Die Liste soll aus `moloch-dev.md` gelesen werden.

**Aenderung in `.claude/hooks/pre-edit-check.sh`:**

Ersetze den statischen `ROT_FILES=(...)` Block durch:

```bash
# ROT-Dateien DYNAMISCH aus moloch-dev.md lesen
ROT_FILES=()
if [ -f "$CLAUDE_PROJECT_DIR/.claude/skills/moloch-dev.md" ]; then
    while IFS= read -r line; do
        # Zeilen wie "core/moloch_service.py" extrahieren
        FILE=$(echo "$line" | grep -oP '^\s*\S+\.py' | xargs basename 2>/dev/null)
        [ -n "$FILE" ] && ROT_FILES+=("$FILE")
    done < <(sed -n '/^### ROT/,/^### GELB/p' "$CLAUDE_PROJECT_DIR/.claude/skills/moloch-dev.md")
fi

# Fallback falls Datei nicht lesbar
if [ ${#ROT_FILES[@]} -eq 0 ]; then
    ROT_FILES=("moloch_service.py" "tappas_pipeline.py" "camera.py" "settings.json")
fi
```

---

## SCHRITT 3: MCP-Server in Hooks einbinden

Der MCP-Server (`mcp/moloch_mcp_server.py`) laeuft auf dem Pi.
Die Hooks koennen ihn nutzen — aber NUR auf dem Pi, nicht in der Cloud.

**SessionStart Hook erweitern** — Status vom Pi automatisch laden:

In `.claude/hooks/session-start.sh` ergaenzen:

```bash
# MCP Status-JSON lesen (nur auf Pi verfuegbar)
STATUS="/dev/shm/moloch_status.json"
if [ -f "$STATUS" ]; then
    echo ""
    echo "--- MOLOCH LIVE STATUS ---"
    python3 -c "
import json
with open('$STATUS') as f:
    s = json.load(f)
print(f\"  Service: {'AKTIV' if s.get('running') else 'GESTOPPT'}\")
print(f\"  FPS: {s.get('fps', 'N/A')}\")
print(f\"  RAM: {s.get('ram_mb', 'N/A')} MB\")
print(f\"  Tracking: {s.get('tracking_state', 'N/A')}\")
print(f\"  NPU: {s.get('npu_scenario', 'N/A')}\")
" 2>/dev/null
else
    echo ""
    echo "--- KEIN LIVE STATUS (nicht auf Pi oder Service gestoppt) ---"
fi
```

---

## SCHRITT 4: Skills mit Hooks verknuepfen

Skills koennen Hooks triggern und umgekehrt. Konkret:

### Skill `moloch-dev` → Pre-Flight wird zum Hook

Die Pre-Flight Checks aus `moloch-dev.md` (Abschnitt 4) laufen jetzt
AUTOMATISCH ueber Hooks. Der Skill muss das nicht mehr beschreiben.

**In `.claude/skills/moloch-dev.md` ersetze Abschnitt 4 durch:**

```markdown
## 4. PRE-FLIGHT CHECKS

Pre-Flight Checks laufen AUTOMATISCH ueber Hooks:
- SessionStart: Git-Status, Handoff, Bugs
- PreToolUse: ROT-Dateien, NEVER-Regeln
- PostToolUse: Syntax-Check

Manuell nur noch noetig:
- Baseline erfassen (FPS, RAM, CPU Temp)
- Service-Status pruefen bei Zweifel
```

### Skill `moloch-audit` → Post-Flight wird zum Hook

Fuer automatischen Audit NACH groesseren Aenderungen, ergaenze in
`.claude/settings.json` einen Stop-Hook:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Wurden in dieser Session ROT-Dateien geaendert? Falls ja, erinnere den User an: python3 ~/moloch/moloch_audit.py --auto"
          }
        ]
      }
    ]
  }
}
```

---

## SCHRITT 5: Agent-MDs als echte Subagenten nutzen

Die Agent-MDs unter `agents/AGENT_*.md` sind bisher nur Kontext-Dokumente.
Claude Code kann sie als echte Subagenten starten.

**Pattern fuer den User:**

```
Lies ~/moloch/CLAUDE.md und ~/moloch/agents/AGENT_VISION.md.
Pruefe ob die TAPPAS Pipeline korrekt laeuft.
```

**Pattern fuer automatische Nutzung (in CLAUDE.md ergaenzen):**

```markdown
## SUBAGENTEN-NUTZUNG

Bei komplexen Auftraegen: Starte Subagenten parallel.

Beispiel — nach grosser Aenderung an tappas_pipeline.py:
1. Subagent Vision: "Pruefe Pipeline-Status und FPS"
2. Subagent Service: "Pruefe journalctl auf Fehler"
3. Subagent Hardware: "Pruefe Temperatur und RAM"

Claude Code kann diese parallel starten (Agent-Tool mit subagent_type).
```

---

## SCHRITT 6: Permissions erweitern

In `.claude/settings.json` sind jetzt erweiterte Permissions:

```json
"permissions": {
  "allow": [
    "Bash(python3:*)",
    "Bash(git:*)",
    "Bash(journalctl:*)",
    "Bash(systemctl:*)",
    "Bash(cat /dev/shm/*)",
    "Bash(ls:*)",
    "Bash(mkdir:*)",
    "Bash(chmod:*)",
    "Bash(find:*)"
  ]
}
```

Falls auf dem Pi weitere Befehle noetig sind (z.B. `hailortcli`, `ffprobe`):

```json
"Bash(hailortcli:*)",
"Bash(ffprobe:*)",
"Bash(vcgencmd:*)"
```

---

## ZUSAMMENFASSUNG

Nach HOOKWIRE-Einrichtung passiert AUTOMATISCH:

| Ereignis | Was passiert | Quelle |
|----------|-------------|--------|
| Session startet | Git-Status, Handoff, Bugs, Live-Status | Hook + MCP |
| Vor jedem Edit | ROT-Warnung (dynamisch aus moloch-dev.md) | Hook + Skill |
| Vor jedem Edit | NEVER-Regeln geprueft und erzwungen | Hook |
| Nach jedem Edit | Syntax-Check automatisch | Hook |
| Vor Bash-Befehl | Cache-Clear Erinnerung, Worktree-Warnung | Hook |
| Session endet | Uncommitted Changes, Audit-Erinnerung | Hook + Skill |
| Komplexer Task | Subagenten parallel gestartet | Agent-MDs |

**Nichts davon muss der User manuell ausloesen.**

---

## CODEWORT

**HOOKWIRE** — sage in einer neuen Claude Code Session:

> "Lies HOOKWIRE.md aus dem Repo und richte alles ein."

Claude Code wird dieses Dokument finden und alle Schritte ausfuehren.
