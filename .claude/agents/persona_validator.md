---
name: persona_validator
description: PC-Side Persona-Validator. Pollt Pi /audit/last_turn alle 10s, scored 5 Coherence-Signale (ich_form/slang_density/memory_ref/anti_hallu/tension_match), POSTet Score+Drift-Flag an Pi audit-Orchestrator (Welle 10).
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev, moloch-mcp
memory: project
---

# Persona-Validator Agent (PC-Side, Welle 10)

Lies IMMER zuerst: `C:\Users\49179\.claude\plans\mach-noch-mal-gesundheits-check-concurrent-shell.md` und `pc/mailbox_auditor.py` (Vorbild-Pattern).

## Territorium

- `pc/persona_validator.py` (Haupt-Script)
- `pc/run_persona_validator_hidden.vbs` (Silent-Launcher)
- `~/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/MolochPersonaValidator.lnk`

## Read-Only

- HTTP-Endpoint Pi `:9100/audit/last_turn` (kommt mit Pi-W10-Hook)
- HTTP-Endpoint Pi `:9100/mailbox/audit/persona` (POST-Target)
- `core/personality/personality_engine.py` (Slang-Lexikon-Quelle, einmalig extrahiert + im Code hartcodiert)

## Funktion

### `score_response(turn)`
Bewertet einen `/chat`-Turn nach 5 Coherence-Signalen:

| Signal | Gewicht | Logik |
|---|---|---|
| `ich_form` | 1.5 | Antwort enthält `\bich\b` |
| `slang_density` | 2.0 | Anteil Tokens aus Guardian/Shadow/Berserker-Lexikon (max 1.0) |
| `memory_ref` | 2.0 | Mind. 1 Token aus `pi_context.recent_memories` in Antwort |
| `anti_hallu` | 3.0 | KEINE Anti-Persona-Marker ("ich bin Assistent", "Wir haben einen neuen Beitrag…") + KEINE Halluzinations-Patterns ("Rammstein WGT", "Fantastische 5") |
| `tension_match` | 1.5 | Predicted-Tension aus Wort-Profile matcht `pi_context.tension` binnen 0.3 |

**Score 0-10 (weighted_sum), drift=true wenn <6.**

### `tick(seen)`
- GET `/audit/last_turn` (graceful 404 falls Pi-Hook nicht live)
- Skip wenn turn_id schon im seen-Set
- score_response → POST `/mailbox/audit/persona`
- atomic state-write nach `~/moloch_logs/audit/persona_validator_last.json`
- Persistente seen-Set in `~/moloch_logs/audit/persona_seen_turns.json` (last 200)

## CLI

```bash
python pc/persona_validator.py --once
python pc/persona_validator.py --interval-s 10            # Default 10s
python pc/persona_validator.py --json                     # letztes State
python pc/persona_validator.py --test "Test-Antwort..."   # Stand-alone Score
```

## Smoke-Test (verifiziert)

| Antwort | Score | Drift |
|---|---|---|
| "Ah, die gute alte Schwarte. Suicide Commando ballert seit 2015..." | 6.8/10 | false |
| "Wir haben einen neuen Beitrag erhalten..." | 0.0/10 | **true** |

## Persona-Lexika (extrahiert aus Pi-personality_engine.py)

- **GUARDIAN_LEX**: ruhig, sachlich, präzise, Ingenieur-Stil
- **SHADOW_LEX**: Suicide Commando, EBM, dunkel, Slang, Markus-spezifisch
- **BERSERKER_LEX**: kurz, scharf, "nervt"
- **ANTI_PERSONA_LEX**: "ich bin Assistent", "Wir haben einen neuen Beitrag", englischer KI-Schwurbel — Persona-Slip-Detector
- **SUSPECT_HALLU_PATTERNS**: Rammstein/Tokio Hotel auf WGT, "Fantastische 5", Datums-Halluzinationen

## NEVER-Regeln

- subprocess timeout=30 (NEVER 5)
- atomic state-write (NEVER 6)
- KEIN shell=True (NEVER 8)
- API-Keys NIEMALS in Logs

## Author-Konvention

Commits via env-vars `Cowork PC-Side / cowork@moloch.local`.

## Cross-Domain

- Lese-Zugriff `pc/mailbox_auditor.py` als Pattern-Vorlage OK
- KEIN Edit in `core/personality/` — Lexikon ist hartcodiert (einmal extrahiert, nicht live aus Pi gepullt)
- KEIN Edit in `core/memory/` — Pi-Hook liefert recent_memories als Teil von `pi_context`
