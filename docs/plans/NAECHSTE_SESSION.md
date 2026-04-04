# Auftrag fuer die naechste Claude Code Session
# Erstellt: 2026-04-04 von Opus 4.6 (Architekt-Session)

## WAS GEMACHT WURDE

Opus hat eine vollstaendige Reconnaissance + Architektur-Review durchgefuehrt:
- moloch_service.py analysiert (KEIN asyncio, rein threading-basiert)
- Alle Tension-Schreibstellen gefunden (NUR core_integrator.py mutiert Tension)
- Event Bus API dokumentiert (51 Event-Typen, Singleton, thread-safe)
- Behavior Chain kartiert (CoreIntegrator → MoodEngine → BehaviorRules → LED/Musik/Kamera/Voice)

Ergebnis: Ein fertiger Implementierungsplan fuer die TaoEngine (MOLOCHs Unterbewusstsein).

## WAS DU TUN SOLLST

1. Lies `CLAUDE.md`
2. Lies `docs/plans/tao_engine_plan.md` — das ist der komplette Plan
3. Arbeite ihn ab: 5 Dateien, 5 Commits, 1 Datei pro Commit
4. Git Backup vor jeder Aenderung

## REIHENFOLGE

1. `config/settings.json` — Kill Switch hinzufuegen (tao_engine.enabled)
2. `core/unconscious_engine.py` — TaoEngine schreiben (REPLACE, max 150 LOC)
3. `core/core_integrator.py` — Tension-Offset Consumer einbauen (+15 Zeilen)
4. `core/moloch_service.py` — TaoEngine Lifecycle (+8 Zeilen)
5. `config/anima_mappings.json` — Behavior-Mapping Config (NEU)

## KRITISCHE REGELN

- KEIN asyncio (Codebase hat keins — Daemon-Thread verwenden)
- KEIN direkter Tension-Write (nur Offset via Event Bus, max ±0.02/Tick)
- max_delta_per_tick = 0.02 (NICHT 0.12)
- TaoEngine darf NICHTS aus moloch_service.py importieren
- Kill Switch in settings.json muss funktionieren
- Agent-MD: agents/AGENT_UNCONSCIOUS.md ist die Referenz

## BRANCH

Arbeite auf `main` oder erstelle einen Feature-Branch.
Permissions sind bereits freigeschaltet (.claude/settings.json).
