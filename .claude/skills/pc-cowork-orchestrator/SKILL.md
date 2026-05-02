---
name: pc-cowork-orchestrator
description: Welle 21 Phase 2 PC-Side. DeepSeek/Claude Cloud-LLM mit function-calling als Orchestrator. Multi-Step-Loop mit Pi-Tool-Bridge. Build/Debug/Test des Stacks in pc/agent/.
user-invocable: true
---

# PC-Cowork-Orchestrator Skill (Welle 21 Phase 2)

Cloud-LLM als Agent mit function-calling. Ruft Pi-Tools via HTTP-Bridge. Multi-Step bis finale Antwort.

## Architektur

```
User-Query
    ↓
PC-Orchestrator (pc/agent/orchestrator.py)
    ↓
DeepSeek/Claude function-calling
    ├─ tool_use? -> dispatch via Bridge
    │   ├─ HttpBridge -> Pi :9100/api/agent/dispatch
    │   └─ MockBridge -> lokal (search_proxy 11650 + spotify_stats.json)
    │
    └─ finale Antwort -> Markus
```

## Setup

```bash
# DeepSeek-Key (api_keys.json oder env)
# Pflicht: config/api_keys.json:deepseek="<sk-...>" ODER env DEEPSEEK_API_KEY

# Smoketest mit MockBridge (Pi-unabhaengig)
cd C:/Users/49179/moloch_repo
python -m pc.agent.orchestrator --mock --verbose "Wer sind meine Top-3 Artists?"

# Live-Test (HttpBridge, wenn Pi-Tool-Dispatcher erreichbar)
python -m pc.agent.orchestrator "Welche P-Bands aufm WGT 2026?"

# Test-Suite (3 Cases)
python -m pc.agent.orchestrator_test
```

## Loop-Pattern

```python
def run(self, user_query: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    tools = self.bridge.get_catalog()
    total_tokens = 0
    tool_calls_log = []

    for iteration in range(self.max_iter):
        response = deepseek_client.complete(messages, tools=tools)
        usage = response.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)
        msg = response["choices"][0]["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            return {"answer": msg["content"], "iterations": iteration+1,
                    "tool_calls": tool_calls_log, "total_tokens": total_tokens}

        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            result = self.bridge.dispatch(name, args)
            tool_calls_log.append({"name": name, "params": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result)})

    return {"answer": "[max iterations reached]", ...}
```

## Tool-Bridge-Pattern

```python
# pc/agent/pi_tool_bridge.py
class HttpBridge:
    def get_catalog(self) -> list:
        r = requests.get(f"{PI_BASE}/api/agent/tools", timeout=10)
        return r.json().get("tools", [])

    def dispatch(self, tool_name: str, params: dict) -> dict:
        r = requests.post(f"{PI_BASE}/api/agent/dispatch",
                          json={"tool_name": tool_name, "params": params},
                          timeout=30)
        return r.json()

class MockBridge:
    # Lokal-Test, nutzt search_proxy:11650 + spotify_stats.json direkt
    ...

def get_bridge() -> ToolBridge:
    # Auto-Fallback: HttpBridge wenn Pi erreichbar, sonst MockBridge
    try:
        r = requests.get(f"{PI_BASE}/api/agent/tools", timeout=3)
        if r.status_code == 200:
            return HttpBridge()
    except:
        pass
    return MockBridge()
```

## DeepSeek-Client (OpenAI-kompatibel)

```python
# pc/agent/deepseek_client.py
def complete(messages, tools=None, model="deepseek-chat",
             max_tokens=2000, temperature=0.3, timeout=90):
    api_key = _load_api_key()  # nie loggen
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = requests.post(API_URL,
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()
```

## System-Prompt

```
Du bist Moloch — anatomisches AI-System auf Markus' Pi+PC.
Du hast Zugriff auf Tools fuer Web-Suche, URL-Fetch, Spotify, Mood/Zone und
Hardware-Aktoren. Nutze Tools wenn noetig — recherchiere selbst statt zu raten.
Antworte deutsch, knapp, direkt. Kein "natuerlich gerne". Wenn ein Tool fehlt,
sag das klar.
```

## Akzeptanztests

| Query | Erwartetes Tool | Erwartete Antwort |
|---|---|---|
| `Top-3 Artists?` | spotify_top_artists | Suicide Commando, SIERRA, Vomito Negro mit Plays |
| `Welche P-Bands aufm WGT?` | web_search + web_fetch + spotify_top_artists | Portion Control + Perturbator |
| `Wie geht's?` | none (smalltalk) | direkt-Antwort, kein Tool |
| URL-Paste | web_fetch | Page-Inhalt |

## NEVER

- NIE max_iter > 10 (Cost-Explosion)
- NIE API-Key loggen
- NIE Loop ohne max_iter
- NIE Cloud-Call ohne timeout
- NIE Tool-Result direkt in TTS (kann Hallu sein -> Halluzination-Detector pflichten)

## Phasen-Status

- **Phase 2 ready** (commit f872e77): orchestrator.py + deepseek_client.py + pi_tool_bridge.py + test
- Phase 3 offen: voll-Spotify-Catalog (11 Tools), restliche Pi-Tools
- Phase 4 offen: Closed-Loop-Verifier `agent_loop_verify`
- Phase 5 offen: Old single-shot abgeschaltet hinter config-flag
