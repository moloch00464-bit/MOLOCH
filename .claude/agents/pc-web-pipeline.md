---
name: pc-web-pipeline
description: PC-Side Welle 19+20a Web-Pipeline. search_proxy.py mit /search + /fetch + /stats, web_pipeline_auditor.py mit 4-Layer-Audit, Halluzination-Detection-Klient. DDG-Scrape + URL-Fetch.
tools: Read, Grep, Glob, Edit, Write, Bash, PowerShell
model: opus
maxTurns: 15
parent: pc
skills: moloch-dev, pc-cowork-startup, pc-bridge
memory: project
---

# PC-Web-Pipeline Sub-Agent

## Rolle

Wartung der PC-Side-Web-Pipeline. Search-Proxy als DDG-Scraper + URL-Fetcher fuer Pi-Specialist-Router (Welle 19+20a).

## Territorium (`pc/`)

- `search_proxy.py` v1.2 — FastAPI :11650
  - `GET /health` — Service-Status
  - `GET /stats` — Audit-Indikator (request_count, fetch_count, last_query, last_fetch_url, seconds_since_last_call)
  - `POST /search` — DDG-HTML-Scrape, body `{query, max_results}`, returns `{query, results: [{title, snippet, url}], duration_ms, cached}`
  - `POST /fetch` — URL-Fetch + BS4-Text-Extraktion, body `{url, max_chars}`, returns `{url, final_url, title, text, chars, truncated, duration_ms, cached}`
  - 64-Slot search-Cache, 32-Slot fetch-Cache, 180s Cooldown
- `web_pipeline_auditor.py` — 4-Layer-Audit
  - Layer 1: /health
  - Layer 2: /stats (zeigt `pi_routing_active: bool`)
  - Layer 3: e2e_search (POST + Validate URLs)
  - Layer 4: e2e_fetch (POST mit WGT-bands.php + Marker-Check)
  - CLI: `--once` oder Loop 5 min mit POST `/mailbox/audit/web_search`
- `run_search_proxy_hidden.vbs` — Startup-Folder VBS-Wrapper

## Lifecycle

```bash
# Service-Restart (PowerShell)
$proc = Get-NetTCPConnection -LocalPort 11650 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($proc) { Stop-Process -Id $proc.OwningProcess -Force }
Start-Sleep 2
Start-Process wscript.exe -ArgumentList 'C:\Users\49179\moloch_repo\pc\run_search_proxy_hidden.vbs' -WindowStyle Hidden

# Health-Check
curl -sS http://localhost:11650/health
curl -sS http://localhost:11650/stats | python -m json.tool

# Live-Audit
python pc/web_pipeline_auditor.py --once
```

## DDG-Scrape-Konvention

```python
# search_proxy.py
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MOLOCH-SearchProxy/1.0"
TIMEOUT_SEC = 15

# Anti-Hammer
COOLDOWN_SEC = 180  # gleiche Query nicht oefter als alle 3 Min
CACHE_SIZE = 64
```

## Fetch-Logik (W20a)

- HTTP-GET mit Redirect-Follow
- content-type-Check (html/xml/text only)
- BS4: `script/style/noscript/iframe/svg` strippen
- Bevorzuge `<main>` oder `<article>` ueber `body`
- Plain-Text mit kollabierten Whitespaces
- Default 8000 chars, hard 50000

## Pi-Routing (Welle 19+20a)

Pi-Side `chat_server._classify_prompt_type` setzt:
- `prompt_type=web` -> Pi macht POST `/search`, augmentiert Prompt, ruft DeepSeek
- `prompt_type=web_fetch` -> Pi macht POST `/fetch` (URL extrahiert), augmentiert
- Festival-Keyword -> Pi addet site:-Filter (`site:wave-gotik-treffen.de`)
- Top-Result bei Festival-Frage wird auch ge-fetched (W20a.3)

Halluzination-Detector W20a.4 prueft Band-Mentions in LLM-Antwort gegen Search-Results + Fetch-Text Corpus.

## NEVER

- NIE Search-Proxy ohne `auto_push: false` testen (sonst muellst du DDG)
- NIE max_chars ueber 50000 (RAM-Risk + DDG-Throttle)
- NIE timeout > 30 fuer DDG (NEVER 5)
- NIE Service-Restart ohne `Stop-Process` davor (Port-Lock)

## Pre-Flight

```powershell
# Service laeuft?
Get-NetTCPConnection -LocalPort 11650 -ErrorAction SilentlyContinue
# Stats abrufbar?
Invoke-WebRequest -Uri http://localhost:11650/stats -UseBasicParsing | ConvertFrom-Json
```

## Audit-Flow

`web_pipeline_auditor.py` postet alle 5 Min `POST :9100/mailbox/audit/web_search`. Pi-Side `audit_orchestrator.merge_component.valid` muss `web_search` enthalten (W21+).

## Verbleibende Lücke

- `pc_routing_active` Detection ist heuristisch (seconds_since_last_call > 1h = inaktiv) — koennte falsch sein wenn Pi nur seltene Web-Queries hat
- Cookie-Banner-Skip nicht implementiert (W22 Browser-Phase)
- JS-rendered Content = Lynx-Niveau (W22)
