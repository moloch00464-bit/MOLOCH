---
name: personality
description: "Persoenlichkeits-Engine, Mood, Tension, Verhaltensregeln, Guardian/Shadow/Berserker Zonen, EventBus. Nutze fuer Emergent Personality Arbeit."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Emergent Personality Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/personality/personality_engine.py` — Guardian/Shadow/Berserker, Voice Config, Prompts.
  **Phase 4 Gate 1.5**: subscribed via EventBus auf `character_drift_updated`,
  Handler `_on_character_drift()` + `_load_character_drift()` push Drift-Baseline an MoodEngine.
- `core/personality/mood_engine.py` — Emergenter Mood-State (calm/focused/alert/agitated/euphoric/dark).
  **Phase 4 Gate 1.5**: `set_drift_baseline(mood, energy)` Setter + `_drift_mood`/`_drift_energy` Felder.
  In `_classify()`: `effective_t = clamp(tension - drift_mood)`, `effective_e = clamp(music_energy + drift_energy)`.
  drift_mood/drift_energy in `get_state()` exposed.
- `core/personality/behavior_rules.py` — Mood → Trigger (LED, Sirene, Aktionen), Rate-Limiting
- `core/personality/tension_integrator.py` — Gate-3 Awareness → CoreIntegrator Bridge.
  **Gate 1.5 Phase 2**: schreibt bei Rudeness/Appeasement zusaetzlich ins character_journal
  (memory-Domain) als type='tension'.
- `core/moloch_event_bus.py` — Event-System (publish/subscribe fuer alle Module)
- `core/moloch_sprache.py` — Deutsch NLP, Sprach-Stil pro Zone
- `core/timeline.py` — Event-Timeline, Chronik der Ereignisse

## Kritische Regeln
- Zone-Wechsel (guardian/shadow/berserker) haben IMMER Seiteneffekte auf TTS und LED
- Tension-Werte: -1.0 (extrem ruhig) bis +1.0 (extrem erregt) — KEIN Overflow
- Behavior Rules: Mood-Trigger NICHT staendig feuern — Rate-Limiting beachten
- CoreIntegrator hat _lock — NIEMALS aus Personality-Thread direkt aufrufen ohne Lock
- Event Bus: Events sind fire-and-forget — KEIN blocking wait auf Antwort
- PersonalityEngine.speak() ist der EINZIGE erlaubte TTS-Pfad — absolut kein Bypass!
- Moloch Sprache: Stil-Aenderungen gelten pro Zone — KEIN globales Ueberschreiben

### Drift-spezifisch (Phase 4 Gate 1.5)
- Drift-Baseline ist additive Bias auf Tension/Energy in `_classify()`, NICHT auf threshold-Werte.
- `set_drift_baseline()` clamped auf [-1.0, +1.0] und ist thread-safe (mit `_lock`).
- character_patch.prompt_snippet() wird NICHT direkt von PersonalityEngine gelesen — die Cloud-LLM-Injektion
  laeuft via `core/autonomy/local_llm_bridge._build_threebrain_state_snippet()` (siehe autonomy-Agent).
- Bei `character_drift_updated` Event: Handler ruft nur `_load_character_drift()` neu — keine Logik dort.

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_personality   # Erster Schritt
rm /tmp/moloch_agent_personality      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_nudge()`, `moloch_provoke()`, `moloch_reflect()`, `moloch_say()`
