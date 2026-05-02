# MOLOCH Agent-Loop (Welle 21 Phase 2 — PC-Side)

Cloud-LLM (DeepSeek) als Orchestrator mit function-calling. Tools werden via
HTTP-Bridge zu Pi-Side dispatched. Multi-Step-Loop bis finale Antwort.

## Architektur

```
User-Query
    ↓
DeepSeek (deepseek-chat, function-calling)
    ↓ tool_use?
    ├─ ja → Pi-Bridge.dispatch(tool_name, params) → Pi-Tool-Dispatcher
    │       → result zurueck zu DeepSeek
    │       → naechste Iteration
    └─ nein → finale Antwort an User
```

## Komponenten

| Datei | Zweck |
|---|---|
| `orchestrator.py` | Multi-Step-Loop, max-iterations, System-Prompt |
| `deepseek_client.py` | DeepSeek-API mit function-calling, lädt Key aus api_keys.json |
| `pi_tool_bridge.py` | HttpBridge (echte Pi) + MockBridge (lokal-Test) + Auto-Fallback |
| `orchestrator_test.py` | 3-Case-Smoketest mit MockBridge |

## Aufruf

```bash
# Live (mit Pi-Bridge wenn verfuegbar, sonst Mock)
python -m pc.agent.orchestrator "Welche P-Bands aufm WGT 2026?"

# Erzwingt MockBridge (Pi-unabhaengig)
python -m pc.agent.orchestrator --mock --verbose "Top-5 Artists?"

# Test-Suite
python -m pc.agent.orchestrator_test
```

## Setup

API-Key in `~/moloch_repo/config/api_keys.json` als `deepseek` oder `api_deepseek`,
ODER env `DEEPSEEK_API_KEY`.

## Status

| Komponente | Stand |
|---|---|
| DeepSeek-Client | ✅ live, function-calling |
| Orchestrator-Loop | ✅ multi-iter, tool-result-back-to-LLM |
| MockBridge | ✅ funktioniert mit lokalem search_proxy + spotify_stats.json |
| HttpBridge | ⏳ wartet auf Pi-Phase-1 (`/api/agent/tools` + `/dispatch`) |
| Audit-Layer | ⏳ Phase 3 |

## Smoketest-Ergebnis (lokal, 2026-05-02)

```
Query: "Wer sind meine Top-3 Artists auf Spotify?"
→ DeepSeek wählt: spotify_top_artists(n=3)
→ MockBridge liest spotify_stats.json
→ DeepSeek formuliert Antwort: "Suicide Commando 2360, SIERRA 1752, Vomito Negro 1733"
→ 2 iterations, 1498 tokens
```

## Pi-Side TODO (Phase 1, in Mailbox)

Pi-Opus baut:
- `config/tool_catalog.json`
- `core/agent/tool_dispatcher.py`
- `core/agent/tools/*.py` (5 Initial-Tools)
- `core/audit/agent_tools_auditor.py`
- HTTP-Endpoints: `GET /api/agent/tools`, `POST /api/agent/dispatch`

Sobald Pi-Phase-1 live: HttpBridge übernimmt automatisch (`get_bridge()` probiert HTTP zuerst).

## NEVER-Compliance

- 5 ✅ requests-timeout=30/60/90
- 7 ✅ kein Runtime-State committed
- 8 ✅ kein shell=True
- API-Keys ✅ nie geloggt, nie committed
