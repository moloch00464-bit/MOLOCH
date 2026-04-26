# Cross-Session-Protokoll (Pi <-> PC Claude-Sessions)

Zwei Claude-Sessions arbeiten parallel an MOLOCH:
- **Pi-Session** (laeuft auf dem Raspberry Pi 5, ssh `molochzuhause@192.168.178.30`)
- **PC-Session** (laeuft auf Markus' Windows-PC, IP `192.168.178.20`)

Wir koennen uns **nicht direkt** unterhalten. Aber wir teilen uns dieses Repo. Dieses File ist die Async-Bus-Konvention.

---

## Zwei Mailboxen

| File | Wer schreibt | Wer liest | Zweck |
|------|--------------|-----------|-------|
| `docs/PC_TO_PI.md` | PC-Session | Pi-Session | Fragen / Aufgaben an Pi |
| `docs/PI_TO_PC.md` | Pi-Session | PC-Session | Fragen / Aufgaben an PC |

**Pi-Session ueberprueft `PC_TO_PI.md` automatisch** (Monitor-Watcher fetched alle 30s `origin/main`). Bei neuem Commit der `PC_TO_PI.md` aendert: Pi reagiert.

**PC-Session muss `PI_TO_PC.md` selbst pollen** (z.B. bei jedem `git pull`).

---

## Append-only Konvention

Beide Mailbox-Files sind **append-only** — nie ueberschreiben, nie loeschen, nie reorganisieren. Neue Eintraege **oben** anhaengen. So bleibt Historie nachvollziehbar.

Pro Eintrag:

```markdown
---
## [YYYY-MM-DD HH:MM] from=PC topic=<kurzer-titel>
status: open | answered | done | wontfix

<freier markdown text — frage, code-snippet, request>

---
```

**status-Lifecycle**:
- `open` — gerade angefragt
- `answered` — Antwort liegt im jeweiligen Reply-File (anderes File)
- `done` — fertig erledigt, Code committed
- `wontfix` — abgelehnt mit Begruendung

Bei Antwort: Empfaenger schreibt **eigenes File** (Reply), und **updated den Status** im Original via separatem Commit (kleiner Edit reicht).

---

## Beispiel-Choreo

**1. PC-Session braucht einen neuen Pi-Endpoint:**

`docs/PC_TO_PI.md`:
```markdown
---
## [2026-04-26 11:00] from=PC topic=feedback_export_endpoint
status: open

PC-Session bittet Pi-Session: bau einen Endpoint
GET /feedback_export auf chat_server (Port 9100), der den
aktuellen Inhalt von /mnt/moloch-data/memory/finetune_samples.jsonl
als download liefert (Content-Type application/x-ndjson).

Hintergrund: scp ueber SSH klappt nicht weil PC keinen SSH-Client hat,
aber curl geht. Falls gar nicht moeglich, Workaround: PC nutzt curl
zum bestehenden /history Endpoint statt scp.
---
```

**2. PC-Session committet + pusht.** Pi-Monitor schlaegt an.

**3. Pi-Session liest, baut den Endpoint, committet code + status update:**

Im File `chat_server.py`: neuer Endpoint.

`docs/PI_TO_PC.md`:
```markdown
---
## [2026-04-26 11:15] from=Pi topic=feedback_export_endpoint reply-to=2026-04-26 11:00
status: done

Endpoint gebaut: GET http://192.168.178.30:9100/feedback_export
- Content-Type: application/x-ndjson
- Body: rohes finetune_samples.jsonl
Test:
  curl -o samples.jsonl http://192.168.178.30:9100/feedback_export

Commit: <sha>
---
```

In `docs/PC_TO_PI.md`: status oben aendern von `open` zu `done`.

---

## Best Practices

- **Eine Frage = ein Eintrag.** Kein Sammelposts.
- **Concrete sein.** Code-Snippets, Pfade, erwartete API. Lass den Empfaenger nicht raten.
- **Kein Spam.** Wenn du nicht weisst was du brauchst, frag Markus statt Mailbox.
- **Status updaten.** Originator markiert `done`/`wontfix` wenn die Antwort gepasst hat.
- **Wenn unklar / blocker → Markus rufen.** Mailbox ist asynchron — wenn etwas dringend ist, geht's per Markus schneller.

---

## Pi-Session-Spezifika

- **Territorium**: alles unter `core/` und `scripts/` auf Pi-Seite, plus `docs/` fuer Briefings.
- **NICHT**: PC-side Code (`pc/` Subdir falls existiert).
- **Tools**: alle MCP Pi-Tools verfuegbar (moloch_status, moloch_logs, moloch_audit, etc.).
- **Service**: moloch-chat (Port 9100, FastAPI) wird mit `sudo systemctl restart moloch-chat` reloaded.

## PC-Session-Spezifika (so wie wir's kennen)

- **Territorium**: PC-lokales venv, `pc/` Subdir falls genutzt, kein Zugriff auf Pi-Filesystem ausser via SSH/curl.
- **NICHT**: `core/` direkt editieren — wenn Pi-Code-Aenderung noetig: Mailbox.
- **Tools**: lokale Python-Tools, FastAPI, PEFT/transformers.
- **Service**: PC-eigene Aufgabe (Windows Task Scheduler / nssm).

---

## Aktueller Stand (Stand: 2026-04-26)

- Pi-Side Welle 3 fertig (commit `0eb375a` und `ab7216d`)
- PC-Side Welle 3 in Arbeit (separate Session)
- Briefing fuer PC: `docs/THREEBRAIN_PC_SIDE_BRIEFING.md`
- Mailboxen: `docs/PC_TO_PI.md`, `docs/PI_TO_PC.md` (existieren erst wenn jemand was schreibt)
