---
name: pc-cowork-startup
description: Session-Start-Routine fuer Cowork PC-Side Claude-Code Session. Fuehrt LOKOMOTIVE-Schritt-0 vollstaendig durch (git fetch, smoke, mailbox-tail, status-summary). Nutze ZUERST in jeder neuen Session bevor du andere Aufgaben angehst.
user-invocable: true
---

# PC-Cowork-Startup — Session-Start-Routine

Vollstaendiger LOKOMOTIVE-Schritt-0 fuer PC-Side Claude-Code-Sessions. Nicht ueberspringen ausser bei trivialen Single-File-Reads.

## Schnell-Variante: ein Befehl

```bash
bash pc/lokomotive_check.sh
```

Fuehrt alle 10 Schritte automatisiert durch (10 = 8 Standard + Auto-Pipeline-Detection + Federation-Status). Strukturierter Output mit ✓/⚠/✗. Ende: Summary mit HEAD, Mailbox-Tops, Open-Counts, Auto-Pipeline-Aktivitaet 24h.

## Manuelle Schritte (sequenziell, falls Skript nicht laufbar)

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

### 8. Auto-Pipeline-Detection (NEU 2026-04-27 — Heute-Lehre)
```bash
git log --since="1 day ago" --author="cowork-monitor\|cowork-claude-auto" --oneline
grep -c '"kind": "federation_reply"' ~/moloch_logs/cross_session.jsonl
```
Was hat der Daemon ohne dich gemacht? Auto-Trigger v_next_ready_to_train? Federation-Replies (sollte 0 sein da fed_kill aktiv).

### 9. Pool + Adapter
```bash
curl -sS http://localhost:9000/feedback_stats   # Pool: total/approved/pending/rejected
curl -sS http://localhost:11600/list            # active adapter + alle versions
```

### 10. Status-Summary an Markus
1-2 Saetze: was ist Stand? Was ist letzter Pi-Topic? Was sind offene PC-Topics? Letzter Commit. Was laeuft / ist down. **Auch: was hat Daemon autonom gemacht (heute war v2 live ohne dass ich's wusste).**

Format-Beispiel:
```
Bereit. HEAD: <sha> "<msg>". Auto-Pipeline 24h: <N> commits (<Beispiel>).
Pi-Topic offen: <topic>. Open: PC=X PI=Y. Services: alle UP / X DOWN.
Pool: <approved>/<total>. Adapter: <active>. Was tun?
```

## Anti-Pattern

- LOKOMOTIVE-Schritt-0 ueberspringen weil "ich weiss schon was Sache ist" → falsch, repo-state aendert sich oft (Pi pusht parallel)
- Mailbox nicht lesen → Pi-Reply auf vorherige Anfrage verpasst
- Smoke skippen → Bugs schleichen sich ein
- Status-Summary auslassen → Markus weiss nicht was du weisst
