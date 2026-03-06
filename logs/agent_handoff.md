# AGENT HANDOFF — Gate 2/3/4 Session
# Geschrieben: 2026-03-06 ~16:00
# Naechste Instanz: Lies CLAUDE.md, dann diese Datei

## AKTUELLER STAND

Gate 2 + 3 + 4 in einer Session implementiert. Alles committed, NICHT deployed.
Letzter Commit: c2a06ee "Gate 4: Emergent Personality in moloch_service.py verdrahtet"
Service: NICHT neugestartet seit Aenderungen!

## COMMITS DIESER SESSION (chronologisch)

### Gate 2: Identity
| Commit | Beschreibung |
|--------|--------------|
| 5f19c2b | Episodisches Gedaechtnis — Qdrant embedded auf SSD2 |
| 72e5923 | Episodic Memory in _tappas_perception_loop() verdrahtet |
| 562f20f | Music Memory — Track+Person+Mood Assoziationen auf SSD2 |
| 9fb08a7 | Music Memory in moloch_service.py verdrahtet |

### Gate 3: Situational Awareness
| Commit | Beschreibung |
|--------|--------------|
| 48413e1 | RoomMap — PTZ-Winkel zu 5 Raumzonen |
| 3d8e701 | MotionAnalyzer — BBox-Deltas → Bewegungszustand |
| 9bd8909 | ActivityAnalyzer — Kombinierte Signale → Aktivitaet |
| b16618f | ContextEvaluator — 4-Achsen Situationsbewertung Score 0-1 |
| 77391e1 | Awareness Module in moloch_service.py verdrahtet |

### Gate 4: Emergent Personality
| Commit | Beschreibung |
|--------|--------------|
| 1111a67 | TensionIntegrator — Awareness→CoreIntegrator Bridge |
| cc2faeb | MoodEngine — Emergenter Mood (6 States) |
| eb6cf24 | BehaviorRules — Mood→Verhalten Regelwerk |
| c2a06ee | Emergent Personality in moloch_service.py verdrahtet |

## NEUE DATEIEN (10 Module)

```
core/memory/episodic_memory.py        — Qdrant Vector-DB, store_episode()/recall()
core/music/music_memory.py            — Track+Person+Mood JSON auf SSD2
core/awareness/__init__.py            — Package Init
core/awareness/room_map.py            — PTZ-Pan → Zone (Tuer/Schreibtisch/Mitte/Sofa/Fenster)
core/awareness/motion_analyzer.py     — BBox-Deltas → stationary/walking/approaching/leaving
core/awareness/activity_analyzer.py   — → alone/working/conversation/party/away
core/awareness/context_evaluator.py   — → Score 0-1 (familiarity/comfort/alertness/engagement)
core/personality/tension_integrator.py — Awareness Events → CoreIntegrator update_input()
core/personality/mood_engine.py        — → calm/focused/alert/agitated/euphoric/dark
core/personality/behavior_rules.py     — Mood → LED/Sirene/Zone Triggers
```

## GEAENDERT: core/moloch_service.py (~250 neue Zeilen)
- 7 neue Imports (episodic, music, awareness x4, personality x3)
- Init: Awareness + Personality Module (Bloecke 9 + 10)
- _tappas_perception_loop(): Awareness-Chain + MoodEngine nach DailyLearner
- start(): 3 neue Event-Bus Subscriber-Bloecke (Awareness, Personality, Behavior)

## ARCHITEKTUR-DATENFLUSS

```
PFrame (5 Hz Perception Loop)
  ├→ RoomMap(PTZ-Pan) → zone_entered Event
  ├→ MotionAnalyzer(BBox) → motion_state_changed Event
  ├→ ActivityAnalyzer(Signale) → activity_changed Event
  ├→ ContextEvaluator(alles) → context_update Event (Score 0-1)
  ├→ EpisodicMemory(face_id+embedding) → Qdrant
  └→ MoodEngine(Tension+Dominance+Musik+Activity) → mood_changed Event

Event Bus Kette:
  context_update ─────→ TensionIntegrator → CoreIntegrator.update_input()
  activity_changed ───→ TensionIntegrator → CoreIntegrator.update_input()
  motion_state_changed → TensionIntegrator → CoreIntegrator.update_input()
  mood_changed ────────→ BehaviorRules.evaluate() → behavior_trigger Event
  behavior_trigger ────→ LED on/off/blink + CoreIntegrator.set_impulse_flag()
  music_track_started ─→ MusicMemory.store_association() (wenn Person erkannt)
  music_features ──────→ music_energy Cache fuer ActivityAnalyzer
```

## BEKANNTE PROBLEME

### Kritisch — Vor Deploy pruefen
1. **NICHT getestet!** Nur syntax-gecheckt. Service muss neu gestartet werden.
2. **feed_event() existiert NICHT** in CoreIntegrator — wird in moloch_service.py
   aufgerufen (Zeilen 350/352/729/731/795/797), aber die Methode fehlt.
   Alle Aufrufe sind in try/except → silent fail. Neue Module nutzen korrekt update_input().
3. **Qdrant Import**: qdrant-client muss installiert sein (`pip3 install qdrant-client`)
4. **RoomMap Pan-Winkel**: Default-Werte GESCHAETZT, muessen kalibriert werden

### Preview-Stream Bug (analysiert, NICHT gefixt)
- Moegliche doppelte BGR↔RGB Konvertierung in tappas_pipeline.py:1033
- SHM immer 640x360, Preview skaliert hoch → Unschaerfe bei HD+
- Markus informiert, kein Fix beauftragt

### Aus frueherer Session
- Face-ID Threshold evtl. zu niedrig (0.30)
- G1-T11 Labelme uebersprungen (keine Spec)

## EMPFEHLUNG NAECHSTE SCHRITTE

1. `sudo reboot` → `journalctl -u moloch -f` auf Import-Fehler pruefen
2. Falls Qdrant fehlt: `pip3 install qdrant-client`
3. Bei Crash: Imports einzeln testen (python3 -c "from core.awareness.room_map import ...")
4. RoomMap kalibrieren (PTZ an Zonengrenzen fahren, Winkel notieren)
5. Preview-Bug fixen wenn beauftragt
6. feed_event() in CoreIntegrator implementieren ODER bestehende Aufrufe auf update_input() migrieren

## SERVICE-STATUS
- MOLOCH_USE_TAPPAS=1 AKTIV (in ~/.profile)
- Service: MUSS NEU GESTARTET WERDEN nach Reboot
- Letzter stabiler Stand VOR dieser Session: Commit 0fe6eec
