---
name: pc-cowork-startup
description: Session-Start-Routine fuer Cowork PC-Side Claude-Code Session. Fuehrt LOKOMOTIVE-Schritt-0 vollstaendig durch (git fetch, smoke, mailbox-tail, status-summary). Nutze ZUERST in jeder neuen Session bevor du andere Aufgaben angehst.
user-invocable: true
---

# PC-Cowork-Startup — Session-Start-Routine

Vollstaendiger LOKOMOTIVE-Schritt-0 fuer PC-Side Claude-Code-Sessions. Nicht ueberspringen ausser bei trivialen Single-File-Reads.

## Schritte (sequenziell)

### 1. Identitaet ankuendigen
Sag in der Antwort: `LOKOMOTIVE aktiv.`

### 2. Agent-File laden
```
Read .claude/agents/pc.md
```
Antwort: `Agent pc geladen.`

### 3. Master-Briefing laden
```
Read docs/LOKOMOTIVE_PC_COWORK.md
```
Antwort: `Cowork-Briefing geladen.`

### 4. Repo-Sync
```bash
cd C:\Users\49179\moloch_repo && \
  git fetch && \
  git status && \
  git log --oneline -5
```
Wenn behind origin: `git pull --rebase`. Bei untracked files: notieren, NICHT spontan committen.

### 5. Smoke-Test
```bash
pc\smoke.cmd
```
Muss `[smoke] OK` ausgeben. Bei FAIL: nicht weiter, debug zuerst.

### 6. Mailbox-Top lesen
```bash
echo "=== PI_TO_PC top ===" && \
grep -nE "^## \[" docs/PI_TO_PC.md | head -5 && \
echo "" && echo "=== PC_TO_PI top ===" && \
grep -nE "^## \[" docs/PC_TO_PI.md | head -5
```

### 7. Service-Health (optional, ~5s)
```bash
for url in ":11600/health" ":11700/api/state" ":11800/api/state" ":9000/feedback_stats" ":11434/api/tags"; do
  curl -sS -o /dev/null -w "$url HTTP %{http_code}\n" --max-time 3 "http://localhost$url"
done
```

### 8. Status-Summary an Markus
1-2 Saetze: was ist Stand? Was ist letzter Pi-Topic? Was sind offene PC-Topics? Letzter Commit. Was laeuft / ist down.

Format-Beispiel:
```
Bereit. Last commit: <sha> "<msg>". Pi-Topic offen: <topic>. PC-Topics
offen: <count>. Services: alle UP / X DOWN. Was tun?
```

## Anti-Pattern

- LOKOMOTIVE-Schritt-0 ueberspringen weil "ich weiss schon was Sache ist" → falsch, repo-state aendert sich oft (Pi pusht parallel)
- Mailbox nicht lesen → Pi-Reply auf vorherige Anfrage verpasst
- Smoke skippen → Bugs schleichen sich ein
- Status-Summary auslassen → Markus weiss nicht was du weisst
