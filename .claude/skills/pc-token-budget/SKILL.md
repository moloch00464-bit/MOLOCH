---
name: pc-token-budget
description: Cloud-LLM-Cost-Tracking fuer DeepSeek + Claude. Per-Turn-Budget, Daily-Cap, Cost-Estimate, Logging. Verhindert Cost-Explosion bei Agent-Loop oder Pipeline-Bug.
user-invocable: true
---

# Token-Budget — Cloud-LLM Cost-Tracking

Welle 19 (web_model = api_deepseek) und Welle 21 (Orchestrator-Loop) generieren Cloud-Calls. Ohne Budget-Tracking kann ein Bug in der Klassifikation oder ein Loop-without-exit Markus' API-Budget binnen Stunden killen.

## Pricing-Estimate (Stand 2026-05-02)

| Modell | $/1M input tokens | $/1M output tokens | Anmerkung |
|---|---|---|---|
| deepseek-chat | $0.14 | $0.28 | Cache hit: $0.014 |
| deepseek-reasoner | $0.55 | $2.19 | reasoning |
| claude-haiku-4.5 | $1 | $5 | Anthropic |
| claude-sonnet-4.5 | $3 | $15 | Anthropic |

DeepSeek ist 7-10x guenstiger als Claude. Default-Cloud = DeepSeek.

## Budget-Limits

| Scope | Default | Hard-Cap |
|---|---|---|
| Per-Turn (single user_query) | 4000 tokens | 10000 tokens |
| Per-Orchestrator-Loop | 15000 tokens | 30000 tokens (5 iter * 6000) |
| Per-Hour | 100000 tokens | 300000 tokens (~$0.10 DeepSeek) |
| Per-Day | 1.5M tokens | 5M tokens (~$1.50 DeepSeek) |

Bei Hard-Cap: Cloud-Calls blocken, fallback auf lokal (qwen NPU oder dolphin-mistral) mit Hinweis an Markus.

## Tracking-State (PC-Side)

In `/dev/shm/moloch_token_budget.json` (atomic-write, NEVER 6):

```json
{
  "started_at": "2026-05-02T08:00:00Z",
  "totals": {
    "deepseek_input_tokens": 12345,
    "deepseek_output_tokens": 3456,
    "claude_input_tokens": 0,
    "claude_output_tokens": 0
  },
  "hourly_buckets": [<60min-Buckets>],
  "daily_total_usd": 0.0123,
  "alerts_today": []
}
```

Schreibt: `pc/agent/orchestrator.py` nach jedem Cloud-Call. Liest: `pc/agent_tools_auditor.py` Auditor-Layer.

## Implementation in pc/agent/deepseek_client.py

```python
def complete(messages, tools=None, ...):
    # ... existing ...
    response = requests.post(...).json()
    usage = response.get("usage", {})
    track_tokens(
        provider="deepseek",
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )
    return response
```

## Per-Turn-Budget im Orchestrator

```python
class Orchestrator:
    def run(self, user_query):
        budget_remaining = TURN_BUDGET
        for iter in range(MAX_ITER):
            # ... LLM-call ...
            usage = extract_usage(response)
            budget_remaining -= usage.get("total_tokens", 0)
            if budget_remaining <= 0:
                return {"answer": "[token budget exceeded — kuerz die Frage]",
                        "iterations": iter, "total_tokens": ...}
            # ... continue ...
```

## Daily-Cap-Logic

```python
def is_over_daily_cap():
    state = read_budget_state()
    today_total = state["totals"]["deepseek_input_tokens"] \
                + state["totals"]["deepseek_output_tokens"]
    return today_total > DAILY_HARD_CAP

# In orchestrator.py vor Cloud-Call:
if is_over_daily_cap():
    # Fallback auf lokal
    return call_local_llm(...)
```

## User-Visible Cost-Report

CLI:
```bash
python pc/token_budget_report.py
```

Output:
```
=== TOKEN-BUDGET 2026-05-02 ===
DeepSeek:  12345 in / 3456 out  = $0.0027
Claude:        0 in /    0 out  = $0.0000
Total:                              $0.0027

Hourly: ##........  (12% of cap)
Daily:  #.........  (3% of cap)
```

In Cockpit als Sub-Tab "Cost" (W21+ wenn relevant).

## Audit-Layer fuer Tokens

`pc/audit/token_budget_auditor.py`:
- collect() returnt {score, max, status, detail: budget_state}
- WARN bei >50% daily cap
- FAIL bei >90% daily cap

POST `/mailbox/audit/token_budget` (Pi-Whitelist erweitern noetig).

## NEVER

- NIE Cloud-Call ohne Budget-Check
- NIE Token-Counts loggen ohne Aggregation (Privacy: kein Prompt-Content)
- NIE Loop-without-exit (max-iterations Pflicht)
- NIE Cost-Schaetzung als Fakt — nur Estimate (echte Pricing siehe Provider-Dashboard)

## Aktueller Stand

Welle 19+21 noch ohne Budget-Tracking implementiert. Welle 21 Phase 4 oder Phase 5 sollte das einbauen — sonst Risiko bei Agent-Loop-Bugs (z.B. Tool-Call-Schleife wenn Tool fehlschlaegt).
