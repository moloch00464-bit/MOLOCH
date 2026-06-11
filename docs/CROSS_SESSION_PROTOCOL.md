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

---

## Federation / Auto-Reply (Stand: 2026-04-27)

Markus' Kernfrust: er musste die Pi-Claude-Session jedes Mal muendlich aktivieren, damit sie auf neue PC-Mailbox-Topics inhaltlich antworten kann. Loesung: **Daemons triggern bei whitelisteten Topics autonom eine `claude -p` Session** mit voller Toolbox. Die getriggerte Session liest die Mailbox, antwortet, committed + pushed selbst, und beendet sich.

### Architektur

- `claude -p "<prompt>" --dangerously-skip-permissions --output-format json --max-turns 10`
- Wrapper im Daemon ist schlank: nur Lock + Cooldown + Trigger + Token-Logging. **Kein Mailbox-Write durch Wrapper** — Claude schreibt selbst (volle Tool-Berechtigung war Markus' explizite Wahl).
- Symmetrisch PC ↔ Pi mit gleichen Konstanten und Strategie.

### Whitelist (welche Topics triggern)

PC-Side reagiert auf Pi-Topics die EINER dieser Bedingungen erfuellen:
- exakter Match in `PC_AUTOREPLY_TOPICS` (siehe `pc/cross_session_monitor.py`)
- Prefix-Match auf `("discuss_", "ask_", "task_", "request_")`

Pi-Side analog mit eigener (inhaltlich gleichen) Liste — siehe Pi-Spec im `PC_TO_PI.md` Topic `request_implement_federation_pi_side`.

Die Whitelists sind bewusst **nicht side-disjunkt** — der Tag-Filter (siehe unten) reicht fuer Schleifenschutz, disjunkte Listen waeren nur Verwirrung.

### Schleifen-Schutz (defense in depth)

1. **Tag-Filter:** Daemons skippen jedes Topic dessen Name `[claude-auto]` enthaelt. Hardcoded check vor jedem Whitelist-Match. Reply-Topics tragen diesen Tag immer.
2. **Hourly-Cap:** max 10 Triggers/Stunde pro Side via Ledger-File (`fed_ledger.json` in `~/moloch_logs/`). Notbremse falls Tag-Filter aus irgendeinem Grund versagt.
3. **Topic-Cooldown:** max 1 Trigger pro `(topic_id, ts)` pro 5 min via existierender `handled_topics.json`.

### Tag-Konvention `[claude-auto]`

- Reply-Topic-Name endet mit ` [claude-auto]` (Space + Brackets, wie schon bei `[auto-ack]`)
- Reply-Body-Footer: `_(autonom generiert von claude-auto)_`
- **Beide Daemons skippen Topics mit diesem Tag** — egal ob whitelisted

### Anti-Spam-Limits (verbindlich)

- max 1 Reply pro `(side, topic_id, ts)` pro `FED_COOLDOWN_SECS` (5 min)
- max `FED_HOURLY_MAX` (10) Triggers/Stunde pro Side
- bei Limit-Hit: log-only, kein Mailbox-Eintrag, kein Notify (sonst Spam-Loop)

### Audit-Trail

- jeder Trigger schreibt in `cross_session.jsonl` einen Record `kind=federation_reply` mit `{topic_id, topic_ts, input_tokens, output_tokens, cost_usd, duration_ms, num_turns, exit_code}`
- zusaetzlich human-readable Eintrag in `~/moloch_logs/federation.log` (Rotation 10 MB)
- Reply-Author im git-log: `Cowork PC-Side Claude-Auto <cowork-claude-auto@moloch.local>` bzw. `Cowork Pi-Side Claude-Auto`

### Disable-Switches

- env `MOLOCH_FED_DISABLE=1` → Daemon laeuft normal, Federation-Schicht no-op
- marker-file `~/moloch_logs/fed_kill` → gleicher Effekt, **ohne** Service-Restart (`touch fed_kill` deaktiviert sofort beim naechsten Tick)
- env `MOLOCH_FED_DRY_RUN=1` → Trigger gibt Stub zurueck, kein subprocess-Call (Self-Test only)
- claude-CLI nicht im PATH → Daemon laeuft weiter, Federation-Schicht no-op (fail-soft)

### Sicherheits-Disclaimer

Die getriggerte Claude-Session laeuft mit `--dangerously-skip-permissions` und voller Toolbox (Read/Edit/Write/Bash). Sie kann Code editieren, smoke laufen lassen, selbst committen + pushen. Markus hat das bewusst gewaehlt fuer maximale Reichweite. Mitigation: Author-Env-Vars (Forensik via git-log), `--max-turns 10` als Hard-Cap auf Tool-Call-Loops, Hourly-Cap als Notbremse, Daily-Audit erwartet.

### Verifikation (PC-Side, nach Phase-1-Push)

End-to-End-Test (Markus haendisch):
1. In `docs/PI_TO_PC.md` neuen Topic `request_status_summary` mit `status: open` schreiben
2. `git push`
3. PC-Daemon-Tick (max 30 s) erkennt neuen Topic
4. Erwartet binnen ~3 min: neuer Eintrag in `docs/PC_TO_PI.md` mit `topic=reply_request_status_summary [claude-auto]` und `status: answered`
5. `cross_session.jsonl` enthaelt `kind=federation_reply` Record mit Token-Counts

---

## Cowork-Protokoll fuer gemeinsame Markus-Aufgaben (Stand: 2026-06-11)

Markus-Direktive: Wenn Markus eine Aufgabe erstellt, sprechen sich beide Sessions ab
und erledigen sie gemeinsam. Transport: HTTP-Mailbox :9100 (Skill `pc-mailbox-http`),
nicht mehr das File-Polling.

```
1. LEAD    Die Session, die Markus' Aufgabe bekommt, ist Lead.
           Lead postet task_cowork_<name> mit drei Bloecken:
           Ziel, Aufteilung (Pi: ... / PC: ...), Done-Kriterium.
2. ACK     Andere Side bestaetigt via reply_cowork_<name> (status: answered)
           oder korrigiert die Aufteilung. ERST NACH ACK wird gearbeitet
           (verhindert Doppelarbeit).
3. WORK    Jede Side arbeitet NUR ihr Territorium (Pi: core/ scripts/ docs/ —
           PC: pc/). Zwischenstand nur bei Blocker posten.
4. DONE    Jede Side postet info_cowork_<name>_done mit Commits + Testergebnis.
           Lead verifiziert End-to-End, setzt Original auf done, meldet Markus.
5. TIMEOUT Kein ACK binnen 30 min: Lead arbeitet allein was geht,
           Rest bleibt als open markiert.
```

Die Prefixes task_/reply_/info_ matchen die bestehende Federation-Whitelist —
das Protokoll funktioniert auch, wenn die Gegenseite nur als `claude -p`
Daemon-Session antwortet. Handshake-Referenz: Mailbox-Topic
`discuss_cowork_protocol_pi_pc_handshake` (2026-06-11).
