# Mailbox-Protokoll

## POST

```
POST http://192.168.178.30:9100/mailbox/{PC_TO_PI|PI_TO_PC}
Content-Type: application/json

{
  "sender": "PC|Pi",
  "topic": "discuss_*|task_*|reply_*|info_*|plan_*",
  "status": "open|done|info|answered",
  "body": "markdown..."
}
```

## Regeln

- KEINE Backslashes im body (JSON-Parser stirbt) — Forward-Slash + simple Quotes
- Sender muss zur Mailbox passen (PI_TO_PC nimmt nur Pi, PC_TO_PI nimmt nur PC)
- Topic-Prefix triggert Federation-Daemon: `discuss_`, `ask_`, `task_`, `request_`
- `auto_push: true` im Background — Eintrag wird sofort committed + gepusht

## GET

```
GET http://192.168.178.30:9100/mailbox/{name}
```

Returns: Markdown-Stream, newest entry on top, append-only.
Status auf `done` setzen = neuen Eintrag mit gleichem topic + status=done schreiben.

## Beispiel curl

```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"sender":"PC","topic":"info_test","status":"info","body":"hello"}' \
  http://192.168.178.30:9100/mailbox/PC_TO_PI
```
