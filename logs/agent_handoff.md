# Agent Handoff — 2026-04-25 (Session 26 — ThreeBrain Welle 1: Foundation)
# Letzter Commit Basis: ce8a088 | Audit: 85/85 PASS | FPS 20.0

---

## SESSION 26 — ThreeBrain FineTune Loop, Welle 1

Markus-Direktive nach Phase-4-Distiller: "die naechste Phase einleiten" → ThreeBrain
ist als unabhaengiger neuer Architektur-Plan akzeptiert. Welle 1 (Foundation)
sofort umgesetzt: character_patch + behavior_mutation_ledger + Cloud-State-Injection
+ Markus-Review-CLI.

User-Entscheidungen (von Plan-Phase):
- Pi-LLM bleibt Qwen2.5-1.5B (Llama2-7B HEF zu aufwendig)
- LoRA-Hybrid: Adapter remote auf Ryzen + nightly HEF-Recompile
- character_drift (Daten) + character_patch (Regeln) BEIDES
- Cloud bleibt DeepSeek

Pragmatische Defaults (von mir gewaehlt):
- CPU-Limit Ryzen-Trainer: 40%
- Critic-Style: hart, sachlich
- Sample-Review: woechentlich

### Geliefert (Welle 1 komplett)

**4 Commits, 4 Dateien:**
| Commit | Datei | Was |
|--------|-------|-----|
| 47cc4df | core/memory/character_patch.py (NEW) | Verhaltens-Regeln Singleton mit Approval-Workflow |
| d03c030 | core/memory/behavior_mutation_ledger.py (NEW) | Append-only Audit-Log alle Charakter-Aenderungen |
| c15ca0e | scripts/review_pending_rules.py (NEW) | CLI: --status / --list / interactive review |
| ce8a088 | core/autonomy/local_llm_bridge.py | _build_threebrain_state_snippet + Inject in DeepSeek |

### Live-Verifikation

- Self-Test character_patch PASS: 3 pending → 2 approved + 1 rejected + 1 deactivated, Snippet 220 chars
- Self-Test ledger PASS: 7 Eintraege, Sequenz, Truncation, Filter
- CLI --status: zeigt 1 active rule + 7 ledger entries
- ThreeBrain-Snippet im Service: 665 chars (unter Budget 800), alle 3 Bloecke (drift + patch + 8 events)
- Audit 85/85 PASS, FPS 20.0, kein Crash

### Architektur-Entscheidungen Welle 1

1. **Singleton-Pattern 1:1 wie character_journal.py** — gleiche atomic-write + NTFS-fallback Helpers, gleiche Self-Test-Struktur.
2. **character_patch & ledger zirkulaer-tolerant**: Patch loggt in Ledger via lazy import + try/except. Ledger weiss nichts von Patch.
3. **Cloud-Injection bewusst NICHT chat_server.py**: Helper liegt in local_llm_bridge.py, wird in `_generate_deepseek` aufgerufen — kein chat_server-Edit, alle DeepSeek-Pfade profitieren automatisch (chat, voice, future).
4. **Snippet-Budget 800 Zeichen**: Drift (kompakt) + Top-1-Erlebnis + max 8 Patch-Regeln + 8 Journal-Events. Bleibt unter Token-Budget der DeepSeek-Calls.
5. **Best-Effort-Failure-Mode in jedem Block**: Wenn Distiller / Patch / Journal fehlt oder crasht → leerer String, kein Bridge-Crash.

### Validation des Cloud-Snippets (manuell ausgefuehrt)

```
=== AKTUELLER CHARAKTER (ThreeBrain) ===
Drift 30d: mood=+0.01 energy=+0.00 dominance=+0.00
Top-Erlebnis: 'tension: Beleidigung erkannt' (gewicht 0.57)
=== AKTIVE VERHALTENSREGELN (gelernt aus Erfahrung) ===
- Wenn Beleidigung detektiert: ein trockener Satz, kein Kommentar danach
Letzte Ereignisse:
  [14:07] spotify: Spielt: Nine Inch Nails - As Alive As You Need Me
  [14:08] camera: Markus betritt Bild
  ...
```

DeepSeek bekommt das jetzt vor JEDER Antwort.

---

## OFFENE PUNKTE FUER WELLE 2

### Welle 2: PC-Side Critic Infrastructure

Nach Plan im `~/.claude/plans/briefing-fuer-pi-opus-hazy-giraffe.md`:

1. **`pc/critic_service.py`** — auf Markus-PC deployen (FastAPI Port 11500)
   - Endpoints: /critic/evaluate, /critic/generate_situation, /health
   - Nutzt lokales Gemma3-12B via Ollama
   - System-Prompt: "Hart, sachlich, kein speichelleckend"
2. **`core/bridge/critic_client.py`** — auf Pi
   - Pattern wie Tentakel-Routing in local_llm_bridge.py
   - Health-Probe alle 5min, Circuit-Breaker
3. **Settings**: neuer Block `critic_service` in settings.json
4. **PC-Setup-Doku**: wie wird Ollama-Gemma auf PC installiert + service gestartet

### Wichtig vor Welle 2

- Pruefen ob auf PC schon Ollama mit Gemma3-12B installiert ist
- Falls nicht: `ollama pull gemma2:12b` (oder gemma3 sobald verfuegbar)
- Firewall-Test 11500 von Pi aus erreichbar

---

## OFFENE THEMEN AUS FRUEHEREN SESSIONS (weiterhin gueltig)

1. **Camera-Hook Phantom-Identity 'Nicht'** — face_db cleanup
2. **Body-only Camera-Trigger** — YOLO ohne Face
3. **Audit-Test fuer Character Journal + Patch + Ledger**
4. **Phase 4 LLM-Pfad in standalone**: heute Nacht echter Live-LLM-Lauf — morgen pruefen

---

## SYSTEM-ZUSTAND AM ENDE DER SESSION

- Service: active, FPS 20.0, RAM 45%, CPU 46.3°C
- Audit: 85/85 PASS
- Git: 4 neue Commits + Backup-Tag `pre_threebrain_w1`
- Agent-Locks: alle entfernt
- Patch: 1 active rule (aus Self-Test) + 7 ledger-eintraege
- Plan-Datei: `~/.claude/plans/briefing-fuer-pi-opus-hazy-giraffe.md` (ThreeBrain 4-Wellen-Plan)

---

## SETUP fuer Session 27 (Markus oder Nachfolge-Claude)

1. `moloch_session_init()` PFLICHT.
2. Pruefen Phase-4 Distiller-Lauf von letzter Nacht (Welle 1 hat keine Trigger fuer das gehabt).
3. Bei Welle 2 Start: Markus-PC (markus-pc.local 192.168.178.20) muss Ollama+Gemma2/3-12B haben.
4. Fuer manuellen Patch-Review: `python3 scripts/review_pending_rules.py`
5. Backup-Anker: `git tag pre_threebrain_w1` falls Rollback noetig.

---

## CHARAKTER-EVOLUTION + THREEBRAIN STAND HEUTE

```
[Markus-Erlebnis]
       ↓
   7 Hooks (Gate 1.5 Phase 2)
       ↓
character_journal/{date}.jsonl
       ↓
Night-Cycle 23:00 → CharacterDistiller (Gate 1.5 Phase 4)
       ↓                       ↓
distill/{date}.json   character_drift.json
       ↓                       ↓
EventBus 'character_drift_updated' → PersonalityEngine + MoodEngine
       ↓
[Welle 1 NEU:]
character_patch.json (manuelle/distiller-vorgeschlagene Regeln)
       ↓
behavior_mutation_ledger.jsonl (Audit aller Aenderungen)
       ↓
DeepSeek-Cloud-Call: System-Prompt enthaelt jetzt
  base_fix + semi_fix + memory + drift + patch + letzte 8 events
       ↓
[Markus erlebt Cloud-Antwort mit echtem aktuellem Charakter]
       ↓
       (Loop)
```

Welle 1 schliesst die Cloud-Mundstueck-Schleife. Die Critic-Trainer-Schleife
(Welle 2-4) wird darauf aufbauen.

---

*Session 26 Ende. Diff-Stats: 4 Commits, 4 Dateien, ~860 LOC, Audit 85/85 PASS.*
