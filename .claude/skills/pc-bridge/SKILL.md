---
name: pc-bridge
description: Cross-Platform Pi <-> Markus-PC Bruecken-Skill. Aktueller Stand 2026-05-02 (W3+W5+W19+W20a+W21). LLM-Tentakel Multi-Modell, Adapter-Proxy, Search-Proxy mit /search + /fetch, Halluzination-Detector, Pi-Tool-Dispatcher, PC-Orchestrator. Nutze bei Bridge-Aufgaben oder Erreichbarkeits-Problemen.
user-invocable: true
---

# PC-Bridge — Pi <-> Markus-PC (Stand 2026-05-02)

## Topologie aktuell

```
Pi (192.168.178.30, Brain)
  | Klassifikator (chat_server._classify_prompt_type)
  | Specialist-Router
  ↓ HTTP
PC (192.168.178.20, Co-Worker)
  ├── Ollama          :11434  Multi-Modell (W5)
  │   ├── moloch-coder      (code, Layer ueber deepseek-coder:6.7b)
  │   ├── dolphin-llama3:8b (complex)
  │   └── dolphin-mistral:7b (web, falls nicht api_deepseek)
  ├── Adapter-Proxy   :11600  Qwen2.5-1.5B + LoRA (W3)
  ├── Search-Proxy    :11650  /search + /fetch + /stats (W19+W20a)
  ├── Agent-Orch.     (W21 PC-Side Skeleton in pc/agent/)
  ├── Avatar          :11800
  └── Dashboard       :11700

Pi-Mailbox-API     :9100/mailbox/{PC_TO_PI,PI_TO_PC} (HTTP, auto_push)
Pi-Tool-Dispatch   :9100/api/agent/{tools,dispatch} (W21 Phase 1)
Pi-Cockpit         :9100/ HTTP + :9443/ HTTPS
PC SSH-Tunnel      :9000 -> Pi-Cockpit

DeepSeek-Cloud     api.deepseek.com (api_keys.json:deepseek)
                   = web_model in tentacle_llm.web_model (W19)
                   = Orchestrator-LLM mit function-calling (W21)
```

## Welle-Lifecycle der Bridges

| Welle | Was | Stand |
|---|---|---|
| W2 | Critic-Service (Ollama dolphin-mistral) | LIVE |
| W3 | Adapter-Inference-Proxy :11600 (LoRA) | LIVE |
| W3 | chat_server :9100/:9443 (HTTP/HTTPS Cockpit) | LIVE |
| W3 | Sample-Sync via /feedback_export | LIVE |
| W5 | Multi-Modell-Routing (3 Ollama-Modelle pro prompt_type) | LIVE |
| W12 | PC-Side-Audit-Layer (5 Auditoren POSTen via /mailbox/audit/*) | LIVE |
| W19 | Search-Proxy `/search` + Augmentation-Schritt vor LLM | LIVE |
| W19 | web_model -> api_deepseek (User-facing-Recherche zu Cloud) | LIVE |
| W20a | Search-Proxy `/fetch` (URL -> plain-text) | LIVE |
| W20a | Halluzination-Detector mit Band-Mentions vs Corpus | LIVE |
| W20a-A3 | `moloch_service(restart)` restartet alle 3 Units | LIVE |
| W21 P1 | Pi-Tool-Dispatcher + 5 Tools + Catalog | Pi-Opus baut |
| W21 P2 | PC-Orchestrator-Loop (DeepSeek function-calling) | LIVE Skeleton, MockBridge -> HttpBridge auto-fallback |
| W22 | Echter Browser (Playwright) | geplant, noch nicht gebaut |

## Pi-Klassifikator -> PC-Routing-Tabelle

| prompt_type | Pi entscheidet | PC erhaelt | Modell/Tool |
|---|---|---|---|
| `simple_smalltalk` | NPU-Bypass | — (bleibt Pi) | qwen2.5:1.5b lokal |
| `hardware_action` | IPC-Bypass | — (bleibt Pi) | direkt IPC, kein LLM |
| `code` | Welle 5 -> PC | Ollama `moloch-coder` | code-Specialist |
| `complex` | Welle 5 -> PC | Ollama `dolphin-llama3:8b` | reasoning |
| `web` (W19) | search_proxy + Cloud | search:11650 -> api_deepseek | recherche |
| `web_fetch` (W20a) | fetch_proxy + Cloud | fetch:11650 -> api_deepseek | URL-Inhalt |
| `recommendation` (W21+) | Orchestrator | DeepSeek function-calling Loop | multi-tool |

## HTTP-Endpoints Cheat-Sheet

### Pi-seitig (von PC aufgerufen)

```
GET  http://192.168.178.30:9100/health
GET  http://192.168.178.30:9100/status
GET  http://192.168.178.30:9100/mailbox/PI_TO_PC
GET  http://192.168.178.30:9100/mailbox/PC_TO_PI
POST http://192.168.178.30:9100/mailbox/PC_TO_PI         (Body JSON {sender, topic, status, body})
POST http://192.168.178.30:9100/mailbox/audit/<component> (Audit-Receiver, W12)

# W21 Phase 1 (Pi-Opus baut grad)
GET  http://192.168.178.30:9100/api/agent/tools          -> {tools: [function-calling-Schema]}
POST http://192.168.178.30:9100/api/agent/dispatch       (Body {tool_name, params})
```

### PC-seitig (von Pi aufgerufen)

```
GET  http://192.168.178.20:11434/api/tags                # Ollama-Modelle
POST http://192.168.178.20:11434/api/generate            # LLM-Inferenz
POST http://192.168.178.20:11434/api/chat                # function-calling-Format
GET  http://192.168.178.20:11600/health                  # Adapter-Proxy
POST http://192.168.178.20:11600/infer                   # LoRA-Inferenz
POST http://192.168.178.20:11650/search                  # DDG-Suche
POST http://192.168.178.20:11650/fetch                   # URL-Fetch
GET  http://192.168.178.20:11650/stats                   # Audit-Indikator
```

## Halluzination-Detection (W19.7 + W20a.4)

Pi-Side closed_loop/web_search_verify.py prueft:
- Genannte Band-Namen in LLM-Antwort vs Corpus aus search_results + fetch_text
- ungrounded_count >= 2 AND no_url AND no_research_marker -> FAIL
- W19.7 grouped_count vs ungrounded_count

Bei FAIL: re-try mit Cloud (Claude oder DeepSeek mit hoeherer temperature) — geplant W21+.

## NEVER-Regeln Bridge

- HTTP-Calls IMMER Timeout: Discovery 5s, Inferenz 30s, Streaming 60s, Cloud 90s
- Circuit-Breaker bei wiederholten Fehlern (3 fails -> 300s Backoff)
- Firewall-Scope STRIKT 192.168.178.0/24, KEINE LAN-Auth
- Failover-Kette IMMER bis Stille (kein Crash)
- Tokens loggen NIE (only counts)
- API-Keys NIE in Logs
- KEIN shell=True

## Debug-Befehle

```bash
# PC-Erreichbarkeit von Pi
ping -c 3 192.168.178.20

# Multi-Service-Health
for url in :11434/api/tags :11600/health :11650/health :11650/stats :11800/api/state; do
  curl -sS -o /dev/null -w "$url HTTP %{http_code}\n" --max-time 3 http://192.168.178.20$url
done

# Search-Proxy Activity-Indikator (zeigt ob Pi-Routing /search nutzt)
curl -sS http://192.168.178.20:11650/stats | python -m json.tool

# Bridge-Logs
journalctl -u moloch -u moloch-chat -u moloch-chat-https -n 100 | grep -iE 'BRIDGE|tentacle|search|fetch|kaskade'
```

## Watchdog

`core/system_watchdog.py` probed Bridges alle 30 Min. Status in `/dev/shm/audit_state.json:layers.bridge` (W14).
