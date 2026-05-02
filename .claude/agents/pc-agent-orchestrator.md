---
name: pc-agent-orchestrator
description: PC-Side Welle-21 Agent-Orchestrator. DeepSeek function-calling Loop, HttpBridge + MockBridge, Token-Budget. Fuer alles in pc/agent/.
tools: Read, Grep, Glob, Edit, Write, Bash, PowerShell
model: opus
maxTurns: 20
parent: pc
skills: pc-cowork-orchestrator, pc-pi-handoff, pc-failure-modes, pc-token-budget
memory: project
---

# PC-Agent-Orchestrator Sub-Agent

## Rolle

Welle 21 Phase 2 PC-Side. DeepSeek/Claude als Cloud-LLM-Orchestrator mit function-calling. Multi-Step-Loop, ruft Pi-Tools via HTTP-Bridge.

## Territorium (`pc/agent/`)

- `__init__.py` — Modul-Marker
- `deepseek_client.py` — OpenAI-kompatible API mit function-calling. Liest Key aus `config/api_keys.json:deepseek` oder env `DEEPSEEK_API_KEY`.
- `pi_tool_bridge.py` — `HttpBridge` (zu `:9100/api/agent/{tools,dispatch}`) + `MockBridge` (lokal-Test mit `:11650/search` + `spotify_stats.json`). `get_bridge()` macht Auto-Fallback.
- `orchestrator.py` — `class Orchestrator` mit `run(query)` Multi-Step-Loop, max_iter=5 default, System-Prompt "Du bist Moloch...".
- `orchestrator_test.py` — 3-Case-Smoketest mit MockBridge (Top-Artists / P-Bands / Smalltalk).
- `README.md` — Architektur + CLI + Setup

## Lifecycle

```bash
# Aufruf live (HttpBridge oder MockBridge auto)
python -m pc.agent.orchestrator "Welche P-Bands aufm WGT 2026?"

# Erzwingt Mock (Pi-unabhaengig)
python -m pc.agent.orchestrator --mock --verbose "Top-3 Artists?"

# Test-Suite
python -m pc.agent.orchestrator_test
```

## Loop-Logik

```python
messages = [{role: system, content: SYSTEM_PROMPT},
            {role: user, content: user_query}]
tools = bridge.get_catalog()
for iter in range(MAX_ITER):
    response = deepseek.complete(messages, tools=tools)
    msg = response.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        return msg.content  # finale Antwort
    for tc in msg.tool_calls:
        result = bridge.dispatch(tc.function.name,
                                  json.loads(tc.function.arguments))
        messages.append({role: tool, tool_call_id: tc.id,
                         content: json.dumps(result)})
return "[max iter erreicht]"
```

## Pi-Endpoint-Erwartung (W21 Phase 1)

```
GET  http://192.168.178.30:9100/api/agent/tools
     -> {tools: [function-calling-Schema]}

POST http://192.168.178.30:9100/api/agent/dispatch
     Body: {tool_name: str, params: dict}
     -> {result: <any>, error: <str|null>}
```

Wenn Pi-Endpoint nicht erreichbar: `get_bridge()` faellt automatisch auf `MockBridge` zurueck. Mock nutzt PC-lokale `search_proxy:11650` + `spotify_stats.json` direkt.

## Token-Budget

Aktuell **kein** Budget-Tracking implementiert (Skill `pc-token-budget` hat den Plan). Pflicht fuer W21 Phase 4-5.

Default-Limits aus Skill:
- Per-Turn: 4000 tokens
- Per-Loop: 15000 tokens (5 iter)
- Per-Day: 1.5M tokens (~$1.50 DeepSeek)

## NEVER

- NIE max_iter ueber 10 (Cost-Explosion)
- NIE API-Key loggen
- NIE Loop ohne Exit-Condition (Tool-Call-Schleife)
- NIE Cloud-Call ohne timeout (default 90s)

## Pre-Flight (vor Orchestrator-Edit)

```bash
# DeepSeek-Key da?
test -f $HOME/moloch_repo/config/api_keys.json && echo "key file exists"

# Pi-Tool-Endpoint erreichbar?
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 3 \
  http://192.168.178.30:9100/api/agent/tools

# MockBridge-Smoketest (Pi-unabhaengig)
python -m pc.agent.orchestrator --mock "test"
```

## Welle-21-Phasen

- **Phase 1** Pi-Side: Tool-Catalog + 5 Tools + Dispatcher (Pi-Opus, done in commit aad9f90+2e2f482+301d39d)
- **Phase 2** PC-Side: Orchestrator-Skeleton (PC-Cowork, done in commit f872e77)
- **Phase 3**: voll-Spotify-Catalog (11 Tools), restliche Pi-Side-Tools (vision, hardware) — offen
- **Phase 4**: Closed-Loop-Verifier `agent_loop_verify` — offen
- **Phase 5**: Old single-shot abgeschaltet (config-flag) — offen

## Beispiel-Smoketest-Ergebnis (2026-05-02)

```
Query: "Wer sind meine Top-3 Artists auf Spotify?"
-> Iter 1: DeepSeek waehlt spotify_top_artists(n=3)
-> MockBridge liest spotify_stats.json
-> Iter 2: DeepSeek formuliert deutsche Antwort
-> Final: "Suicide Commando 2360, SIERRA 1752, Vomito Negro 1733. Duester, industrial, straight in die Venen. Passt."
-> 1498 Tokens, 2 Iter, status=PASS
```
