---
name: pc-cowork-startup
description: Session-Start-Routine fuer Cowork PC-Side Claude-Code Session. Fuehrt LOKOMOTIVE-Schritt-0 vollstaendig durch (MCP-session-init, working-dir-check, Mailbox, Audits, Service-Health). Nutze ZUERST in jeder neuen Session bevor du andere Aufgaben angehst.
user-invocable: true
---

# PC-Cowork-Startup — Lokomotive-Startprotokoll

Vollstaendiger LOKOMOTIVE-Schritt-0 fuer PC-Side Claude-Code-Sessions. Nicht ueberspringen ausser bei trivialen Single-File-Reads.

## Schnell-Variante: ein Befehl

```bash
bash pc/lokomotive_check.sh
```

Strukturierter Output mit ✓/⚠/✗. Macht alle Schritte automatisiert.

## Manuelle Schritte (PC-Cowork-Variante 2026-05-02)

### 0. Identitaet ankuendigen

In der ersten Antwort auf einen Code-Auftrag:
```
LOKOMOTIVE aktiv.
Pre-Flight: Domain / Datei-Ampel / Reboot
```

### 1. MCP-Session-Init (Pi-Health)

```
mcp__moloch__moloch_session_init()
```

→ FPS, RAM, last commit, ERROR/CRITICAL-Logs, agent_handoff. Bei FAIL: STOPP.

### 2. Working-Dir-Check

```bash
pwd  # sollte C:\Users\49179\moloch_repo sein
```

Wenn Working-Dir woanders (z.B. Cowork-Mirror Desktop\Kleine Moloch\...): warnen, weil dann werden `.claude/skills/` und `.claude/agents/` NICHT geladen. Empfehlung: `cd C:\Users\49179\moloch_repo` oder neu starten.

### 3. Agent-File laden

```
Read .claude/agents/pc.md
```

Antwort: `Agent pc geladen.`

### 4. Repo-Sync (PC-Side)

```bash
cd C:/Users/49179/moloch_repo
git fetch
git status
git log --oneline -5
```

Bei behind origin: `git -c user.email=cowork@moloch.local -c user.name="Cowork PC-Side" pull --rebase`. Untracked files notieren — NICHT spontan committen.

### 5. Pi-Status (MCP)

```
mcp__moloch__moloch_status()
mcp__moloch__moloch_npu_workers()
mcp__moloch__moloch_git_log(n=15)
```

Ergibt: FPS, Worker-Health, was Pi-Opus zuletzt gepusht hat.

### 6. Mailbox-Top (HTTP-API)

```bash
curl -s http://192.168.178.30:9100/mailbox/PI_TO_PC | head -100
curl -s http://192.168.178.30:9100/mailbox/PC_TO_PI | head -100
```

Newest-on-top. Suche nach `topic=reply_*` Pi-Antworten + offene `task_*` von mir.

### 7. PC-Service-Health

```bash
for url in ":11600/health" ":11650/health" ":11650/stats" ":11434/api/tags" ":9000"; do
  curl -sS -o /dev/null -w "$url HTTP %{http_code}\n" --max-time 3 "http://localhost$url"
done
```

Erwartet: 200 fuer alle. :9000 ist SSH-Tunnel, kann 200 oder timeout sein.

### 8. Audits

```bash
python pc/moloch_health_check.py     # 8-Layer Self-Test (~5s)
python pc/web_pipeline_auditor.py --once  # 4-Layer Web-Pipeline
```

Ergebnis: PASS-WARN-FAIL Counts. Bei FAIL → erst beheben dann arbeiten.

### 9. Auto-Pipeline-Detection

```bash
git log --since="1 day ago" --author="Cowork" --oneline
git log --since="1 day ago" --author="cowork-monitor\|cowork-claude-auto" --oneline
```

Was hat Daemon ohne dich gemacht (Federation-Replies, Auto-Trigger)?

### 10. Status-Summary

In 1-2 Saetzen an Markus:
- HEAD: <sha> "<msg>"
- Pi-Opus letzte Aktion: <commit-msg-erste-Zeile>
- Mailbox: PC=<open-count>, PI=<open-count>, neueste reply: <topic>
- Audits: PC=<P/W/F>, Web-Pipeline=<P/W/F>
- Was tun?

## Autonom-Modus (seit 2026-05-02)

Markus' Direktive 14:10: **bei mehrteiligen Aufgaben durcharbeiten ohne Frag-pro-Punkt.** Wenn der User "alles fertig", "alle 27 Punkte", "macht alles nacheinander" sagt, ist das implizite Genehmigung fuer die ganze Aufgaben-Liste.

### Was das heisst
- KEIN "soll ich Punkt X bauen?" → einfach machen
- KEIN "welche Variante moechtest du?" → entscheide selbst (sinnvolle Reihenfolge: klein-zu-gross, parallel-zu-Pi)
- KEIN "fertig — was als naechstes?" → einfach den naechsten Punkt der Liste angehen
- Pro Code-Turn: Lokomotive-Header + Pre-Flight bleibt PFLICHT (Selbst-Disziplin, nicht Markus-Permission)

### Wann trotzdem stoppen
- **Audit-FAIL** (moloch_audit zeigt FAIL)
- **Destruktive git-Op** (reset --hard, force-push main)
- **Mehr als 5 ROT-Files** in einem Commit
- **Echter Widerspruch** in den Anforderungen
- **Block durch externe Abhaengigkeit** (z.B. fehlender API-Key, Pi-Service down) → kurzes Update + naechster Punkt
- **Markus widerspricht** explizit per Chat

### Mailbox-Sync mit Pi-Opus
- Bei Punkten die Pi betreffen: **Mailbox-Update** mit Status statt zu warten
- Wenn Pi-Reply ankommt: einarbeiten + weiter
- Cross-Audit alle paar Punkte (jede Schicht synchronisiert)

### Analogie
"Markus geht aus dem Zimmer, kommt zurueck — Arbeit ist erledigt." (aus moloch-dev Skill, gilt analog fuer PC-Cowork)

## Anti-Pattern

- LOKOMOTIVE-Schritt-0 ueberspringen weil "ich weiss schon was Sache ist" → falsch, Pi-Opus pusht parallel
- Working-Dir nicht pruefen → Skills/Agents werden nicht geladen, Tabelle inkonsistent
- Mailbox nicht lesen → Pi-Reply auf vorherige Anfrage verpasst
- Audit skippen → Bugs schleichen sich ein
- Status-Summary auslassen → Markus weiss nicht was du weisst
- **Pro-Punkt-Frage stellen** wenn Markus die Liste schon genehmigt hat → Autonom-Modus brechen, frustriert Markus
