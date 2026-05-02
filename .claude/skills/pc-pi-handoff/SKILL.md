---
name: pc-pi-handoff
description: HTTP-Protokoll fuer Pi -> PC und PC -> Pi Handoffs. Welle 5+19+20a+21 komplett. Welche prompt_types wohin routen, welcher Endpoint welches Format erwartet, Tool-Dispatch-Protokoll, Beispiel-curls.
user-invocable: true
---

# PC <-> Pi Handoff-Protokoll

Detaillierter Pfad-Plan fuer jeden Cross-System-Aufruf. Stand 2026-05-02.

## Pi -> PC (Specialist-Routing, W5+W19+W20a)

### prompt_type=code (Welle 5)

Pi `chat_server.py:_route_specialist` -> POST Ollama:

```http
POST http://192.168.178.20:11434/api/generate
Content-Type: application/json

{
  "model": "moloch-coder",
  "prompt": "<augmented user_query mit Kontext>",
  "stream": false,
  "options": {"temperature": 0.2, "num_ctx": 8192}
}
```

Response: `{response: "<code-text>", total_duration: <ns>, eval_count: <n>, ...}`

### prompt_type=web (Welle 19)

Pi 2-Step:

1. **Augmentation**: POST Search-Proxy
```http
POST http://192.168.178.20:11650/search
{"query": "<refined-query mit site:-Filter>", "max_results": 5}
```
-> `{query, results: [{title, url, snippet}], duration_ms, cached}`

2. **LLM-Call mit augmented prompt**: POST DeepSeek-Cloud (web_model = api_deepseek)
```http
POST https://api.deepseek.com/v1/chat/completions
Authorization: Bearer <key>

{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "<persona>"},
    {"role": "user", "content": "WEB-RESULTS: <results>\n\nFRAGE: <user_query>"}
  ]
}
```

### prompt_type=web_fetch (Welle 20a)

Pi extrahiert URL aus user_query, ruft Search-Proxy /fetch:

```http
POST http://192.168.178.20:11650/fetch
{"url": "<extracted-url>", "max_chars": 8000}
```
-> `{url, final_url, title, text, chars, truncated, duration_ms, cached}`

Dann LLM-Call analog web mit `URL: <url>\nTITEL: <title>\nINHALT: <text>\nFRAGE: <rest>`.

### prompt_type=complex

```http
POST http://192.168.178.20:11434/api/generate
{"model": "dolphin-llama3:8b", "prompt": "<query>", "stream": false}
```

### prompt_type=simple_smalltalk / hardware_action

Bypass — bleibt Pi-lokal. KEIN PC-Aufruf.

## PC -> Pi (Tool-Dispatch, W21)

### Tool-Catalog laden

```http
GET http://192.168.178.30:9100/api/agent/tools
```
-> `{tools: [<function-calling-Schema>]}`

Schema-Beispiel:
```json
{
  "type": "function",
  "function": {
    "name": "spotify_top_artists",
    "description": "Markus' Top-Artists aus Spotify-Stats.",
    "parameters": {
      "type": "object",
      "properties": {"n": {"type": "integer", "default": 20}},
      "required": []
    }
  }
}
```

### Tool dispatchen

```http
POST http://192.168.178.30:9100/api/agent/dispatch
{"tool_name": "spotify_top_artists", "params": {"n": 5}}
```
-> `{result: [{name, plays}, ...], error: null}`

oder
-> `{result: null, error: "<message>"}`

### Beispiel-Loop (PC-Orchestrator)

```python
# pc/agent/orchestrator.py
messages = [{"role": "user", "content": user_query}]
tools = bridge.get_catalog()  # GET /api/agent/tools
for iter in range(MAX_ITER):
    resp = deepseek.complete(messages, tools=tools)
    msg = resp.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        return msg.content  # finale Antwort
    for tc in msg.tool_calls:
        result = bridge.dispatch(tc.function.name, json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
```

## Mailbox-Handoff (Cross-Session-Sync)

### Pi schreibt an PC
```http
POST http://192.168.178.30:9100/mailbox/PI_TO_PC
{"sender": "Pi", "topic": "reply_welle21_phase1_done", "status": "done",
 "body": "5 Tools live, Tool-Dispatcher antwortet GET /tools 200..."}
```

### PC schreibt an Pi
```http
POST http://192.168.178.30:9100/mailbox/PC_TO_PI
{"sender": "PC", "topic": "task_welle21_phase3_extra_tools", "status": "open",
 "body": "## Lokomotive...\n\n## Aufgabe...\n"}
```

`auto_push: true` -> Eintrag wird sofort committed + pusht zu github.com/moloch00464-bit/MOLOCH.

## Routing-Tabelle (komplett)

| User-Query-Form | prompt_type | Wer haendelt | Endpoint |
|---|---|---|---|
| `wie geht's?` | simple_smalltalk | Pi-NPU | qwen2.5:1.5b lokal |
| `naechster Song` | hardware_action | Pi-IPC | direkt (kein LLM) |
| `schreib mir einen Auditor` | code | PC-Ollama | moloch-coder |
| `erklaer mir Sortier-Algos` | complex | PC-Ollama | dolphin-llama3:8b |
| `wieviel Bands aufm WGT?` | web | DeepSeek-Cloud | search + LLM |
| `https://...bands.php was steht da?` | web_fetch | DeepSeek-Cloud | fetch + LLM |
| `welche P-Bands aufm WGT die mich interessieren?` | recommendation (W21+) | PC-Orchestrator | function-calling-loop |

## Latency-Budget

| Operation | Erwartet | Worst-Case |
|---|---|---|
| LAN-Ping Pi -> PC | <5ms | 50ms |
| Ollama-Inferenz CPU (small) | 5-10s | 30s |
| Ollama-Inferenz CPU (8B) | 10-20s | 60s |
| DeepSeek-Cloud | 3-10s | 30s |
| Search-Proxy /search | 0.5-2s | 15s |
| Search-Proxy /fetch | 0.3-3s | 25s |
| Tool-Dispatch | <1s | 30s (NEVER 5) |
| Orchestrator-Loop (3 iter) | 15-45s | 120s |

Bei >5s: Pi sollte TTS-Filler einwerfen (`moment, ich denk nach...`).

## Authentication

KEINE im LAN. Firewall-Scope STRIKT 192.168.178.0/24. Niemals ins Internet exponieren ohne Reverse-Proxy + Auth.

DeepSeek-Cloud: API-Key in `config/api_keys.json:deepseek` (gitignored).
