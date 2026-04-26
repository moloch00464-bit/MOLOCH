---
name: finetune-loop
description: Steuert den ThreeBrain Critic-Actor-LoRA-Cycle End-to-End. Schritte Orchestrator-Run, Markus-Review-Hint, Mailbox an PC-Session, Adapter-Test. Nutze fuer "alles fertig" Direktive oder regelmaessigen Trainings-Run.
---

# Finetune-Loop — ThreeBrain-Trainings-Cycle End-to-End

Dieser Skill steuert den vollstaendigen Critic-Actor-LoRA-Cycle vom Pi aus.
Die PC-Session laeuft parallel und macht den eigentlichen Training-Step.

## Vorbedingungen pruefen

```bash
# 1. PC-Adapter-Proxy lebt? (Welle 3 PC-Side fertig)
curl -s --max-time 3 http://192.168.178.20:11600/health

# 2. Critic-Service lebt? (PC-Ollama mit dolphin-mistral:7b)
curl -s --max-time 3 http://192.168.178.20:11434/api/tags | grep -i dolphin

# 3. Pool-Stand pruefen
python3 -c "from core.memory.feedback_store import get_feedback_store; print(get_feedback_store().get_state())"
```

Wenn PC down: stoppen + Markus melden. Mailbox `docs/PC_TO_PI.md` checken auf offene Anfragen.

## Schritt 1 — Sample-Generation (Orchestrator)

```bash
python3 -m core.autonomy.finetune_orchestrator --max 30 > /tmp/orch_$(date +%H%M).log 2>&1 &
```

~28-40s pro Sample, abhaengig von verfuegbaren drift.recency_weighted_top Seeds.
Bei vorhandenen 10 drift-Top-Events kommen ~10 Samples raus, nicht 30.

Pollen alle 60-120s ob fertig (`pgrep -af finetune_orchestrator`). 
Wenn fertig: Pool-Stand zeigen.

## Schritt 2 — Markus-Review-Hint

```
Pool: X total, Y critic, Z thumbs_up, W thumbs_down, V pending, A approved.

Bitte reviewe pending Critic-Samples:
  python3 scripts/review_pending_rules.py --samples
```

Sortiert nach niedrigstem Score zuerst. Markus geht durch [a]/[r]/[s]/[q].

## Schritt 3 — Trigger PC zum Training

Wenn approved-Pool gewachsen ist und Markus "Go" gibt: Eintrag in Mailbox.

Bridge-Lock setzen, dann `docs/PI_TO_PC.md` Eintrag oben anhaengen:

```markdown
---
## [<UTC-jetzt>] from=Pi topic=v_next_ready_to_train
status: open

Approved-Pool ist auf <N> gestiegen (vorher: <M>). Bitte pull, train v<N+1>.
Mailbox-Status auf `done` setzen wenn /health gruen `adapter: v<N+1>` zeigt.

Cycle: <X> Samples seit letztem Training, davon <Y> critic-approved + <Z> thumbs_up.

Ich teste End-to-End sobald dein Monitor /health adapter-Wechsel sieht.
---
```

Push:
```bash
git add docs/PI_TO_PC.md
git commit -m "PI_TO_PC: v<N+1> ready to train"
git push
rm /tmp/moloch_agent_bridge
```

## Schritt 4 — Warten auf v_next live

Monitor (Bash-Hintergrund) sieht `GET /health` Adapter-Version-Wechsel.
Wenn nicht aktiv: manuell pollen alle 5min:

```bash
curl -s --max-time 5 http://192.168.178.20:11600/health
```

PC-Session schreibt typischerweise selbst eine Mailbox `topic=vX_trained` zurueck — abwarten.

## Schritt 5 — End-to-End-Test v_alt vs v_neu

Sobald Adapter-Wechsel detected:

```bash
python3 -c "
from core.bridge.adapter_inference_client import get_adapter_client
c = get_adapter_client()
print('Aktiv:', c.list_adapters())
out = c.infer(prompt='Wer bist du?', system='Du bist Moloch.', max_tokens=80)
print('Antwort v_neu:', out)
"
```

Vergleich gegen vorige Antwort (logs/finetune_compare/<date>.md falls vorhanden).
Bewertung: enthaelt v_neu echten Moloch-Charakter? Oder noch Halluzinationen?

## Schritt 6 — Markus entscheidet

- v_neu **besser**: weiter zur naechsten Iteration (zurueck zu Schritt 1, mehr Samples sammeln) ODER Welle 4 freischalten
- v_neu **schlechter**: Rollback per `curl -X POST http://192.168.178.20:11600/load -d '{"version":"v<N>"}'`
  ODER Mailbox-Request an PC: "Bitte v<N+1> verwerfen, zurueck auf v<N>"
- v_neu **gleich**: mehr Samples sammeln, neue Iteration

## Out of Scope dieses Skills

- Welle-4-Routing (Cascade + Session-Mode-Override) — eigener Plan
- HEF-Recompile (Pi-LLM autonom) — Phase Z
- Wochenend-Auto-Trigger via Cron — kommt mit session_modes
