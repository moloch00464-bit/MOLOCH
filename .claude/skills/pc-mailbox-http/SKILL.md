---
name: pc-mailbox-http
description: HTTP-Mailbox-API Konvention auf :9100. POST/GET-Templates, Body-Format, Topic-Prefixes, auto_push-Verhalten. Nutze bei jeder Mailbox-Aktion zwischen PC-Cowork und Pi-Opus.
user-invocable: true
---

# HTTP-Mailbox auf :9100

Pi-chat_server bietet eine HTTP-API als Mailbox-Backend. Loest die alte `docs/PC_TO_PI.md` / `docs/PI_TO_PC.md`-File-basierte Mailbox ab.

## Endpoints

```
GET  http://192.168.178.30:9100/mailbox/PC_TO_PI    -> Markdown-Stream
GET  http://192.168.178.30:9100/mailbox/PI_TO_PC    -> Markdown-Stream
POST http://192.168.178.30:9100/mailbox/PC_TO_PI    Body JSON, sender muss "PC" sein
POST http://192.168.178.30:9100/mailbox/PI_TO_PC    Body JSON, sender muss "Pi" sein
```

## JSON-Body

```json
{
  "sender": "PC|Pi",
  "topic": "<prefix>_<beschreibender_name_unter_strich_separiert>",
  "status": "open|done|info|answered|wontfix",
  "body": "Markdown-String mit Forward-Slashes, KEINE Backslashes, einfache Quotes"
}
```

### Wichtig

- **KEINE Backslashes/Pfade im body** — JSON-Parser stirbt. Forward-Slash + simple Quotes.
- **Sender-Match**: PC_TO_PI nimmt nur `sender: PC`, PI_TO_PC nur `sender: Pi`. Sonst HTTP 4xx.
- **auto_push: true** im Background — Eintrag wird sofort committed + pusht zu github.com/moloch00464-bit/MOLOCH.

## Topic-Prefixes

| Prefix | Wann | Beispiel |
|---|---|---|
| `discuss_` | Diskussion / Brainstorming | `discuss_pi_pc_uebergang_abstimmung` |
| `task_` | konkrete Aufgabe an die andere Side | `task_welle19_web_pipeline_fix` |
| `reply_` | Antwort auf vorherigen Topic | `reply_welle20a_url_fetch_pi_integration` |
| `info_` | Status/Info, kein Action erwartet | `info_welle21_phase2_pc_skeleton_ready` |
| `plan_` | langfristiger Plan-Marker | `plan_welle22_echter_browser_playwright_mit_vision` |
| `request_` | Anfrage / Service-Wunsch | `request_implement_federation_pi_side` |

## Status-Lifecycle

```
open ────┬─> answered ─────────> done
         │
         └─> wontfix
```

Status auf `done` setzen = neuen Eintrag mit gleichem topic + status=done schreiben (Mailbox ist append-only — der jüngste Eintrag pro Topic gilt).

## curl-Templates

### Lesen
```bash
curl -sS http://192.168.178.30:9100/mailbox/PI_TO_PC | head -100
```

### Schreiben (kleine Body)
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"sender":"PC","topic":"info_test","status":"info","body":"hello"}' \
  http://192.168.178.30:9100/mailbox/PC_TO_PI
```

### Schreiben (groesserer Body via Datei — empfohlen)

Wegen Bash-Quoting-Problemen bei langen Markdown-Bodies: in Datei schreiben + `--data @file.json` POSTen.

```bash
# 1. JSON-Datei via Write-Tool erstellen
# Inhalt: {"sender":"PC","topic":"...","status":"...","body":"...\n..."}
# Bei \\n im body: schreib echtes \n (nicht escaped)

# 2. POST
curl -X POST -H "Content-Type: application/json" \
  --data @/tmp/mailbox_post.json \
  http://192.168.178.30:9100/mailbox/PC_TO_PI
```

## Anti-Pattern

- POST mit Bash-Quoting bei langem Body -> meist Body-Parser-Error. Nutze @file.
- Backslashes im body -> HTTP 4xx. Nutze Forward-Slash.
- Em-Dashes (`—`) und sonstige Unicode -> kann body-parser stressen. Nutze ASCII (`-`, `--`).
- Apostrophe `'` im body sind OK aber werden manchmal escaped — achte auf doppeltes Escaping.

## Hygiene

- Mein 12:52-Task auf `done` setzen sobald Pi reply geschickt hat
- Alte open-Topics nach 24h pruefen (mailbox_auditor schreibt das in `audit_state.layers.mailbox`)
- Bei Mailbox-Backlog (>10 open): explizit `done` POSTen oder `wontfix`

## Lokomotive-Hinweis fuer Pi-Tasks

Wenn ein `task_*`-Topic an Pi-Opus geschickt wird, MUSS der erste Body-Block der Lokomotive-10-Punkte-Block sein (siehe Memory `feedback_briefing_lokomotive_step0.md`).
