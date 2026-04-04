---
name: personality
description: "Persoenlichkeits-Engine, Mood, Tension, Verhaltensregeln, Guardian/Shadow/Berserker Zonen. Nutze fuer Gate-4 Emergent Personality Arbeit."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Emergent Personality Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/personality/personality_engine.py` — Guardian/Shadow/Berserker Dual-Persoenlichkeit, Voice Config, Prompts
- `core/personality/mood_engine.py` — Emergenter Mood-State (calm, focused, alert, agitated, euphoric, dark)
- `core/personality/behavior_rules.py` — Mood → konkrete Trigger (LED, Sirene, Aktionen)
- `core/personality/tension_integrator.py` — Gate-3 Awareness → CoreIntegrator Bridge
- `core/moloch_event_bus.py` — Event-System (publish/subscribe fuer alle Module)
- `core/moloch_sprache.py` — Deutsch NLP, Sprach-Stil pro Zone
- `core/timeline.py` — Event-Timeline, Chronik der Ereignisse

## Regeln
- Zone-Wechsel (guardian/shadow/berserker) haben IMMER Seiteneffekte auf TTS und LED
- Tension-Werte: -1.0 (extrem ruhig) bis +1.0 (extrem erregt) — KEIN Overflow erlaubt
- Behavior Rules: Mood-Trigger NICHT staendig feuern — Rate-Limiting beachten
- CoreIntegrator hat _lock — NIEMALS aus Personality-Thread direkt aufrufen ohne Lock
- Event Bus: Events sind fire-and-forget — KEIN blocking wait auf Antwort
- PersonalityEngine.speak() ist der EINZIGE erlaubte TTS-Pfad (kein Bypass!)
- Moloch Sprache: Stil-Aenderungen gelten pro Zone — KEIN globales Ueberschreiben

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_personality
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_personality
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_nudge()`, `moloch_provoke()`, `moloch_reflect()`, `moloch_say()`
