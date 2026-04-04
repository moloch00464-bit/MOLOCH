# Agenten-Briefing — Opus 4.6 Analyse vom 2026-04-04
# Abgleich mit Pi-Session empfohlen

## IST: 8 Agenten vorhanden (.claude/agents/)

1. **vision** — TAPPAS, GStreamer, NPU, Perception
2. **hardware** — ONVIF, RTSP, PTZ, eWeLink, LED, Thermal, Fan
3. **gui** — Tkinter Panel, Popups
4. **tracking** — PTZ-Tracker, Such-FSM, Arbiter
5. **voice** — Whisper, TTS, Personality Engine, Spotify
6. **service** — moloch_service, IPC, CoreIntegrator, Memory, systemd
7. **unconscious** — TaoEngine, Unterbewusstsein, Self-Tune (NEU, heute erstellt)
8. **stresstest** — Chaos Engineering, Stabilitaetstests

## ZUSAETZLICH in agents/ (Original-Format, kein .claude/ Pendant)

- **AGENT_DEEPSEEK** — DeepSeek R1 / lokales LLM auf NPU
- **AGENT_TENTACLE** — Tentacle Bridge / externe Anbindungen

## FEHLEND: 5 Agenten fuer volle Abdeckung

### 9. personality
- **Domain:** Persoenlichkeit, Mood, Behavior, Tension-Kette
- **Dateien:** personality_engine.py, mood_engine.py, behavior_rules.py,
  tension_integrator.py, arbitration.py, moloch_sprache.py
- **Warum:** Die gesamte Persoenlichkeits-Kette (Guardian/Shadow/Berserker,
  Mood-Klassifikation, Behavior-Trigger) hat keinen eigenen Agenten.
  Aktuell teilen sich voice und service das — unsauber.

### 10. autonomy
- **Domain:** Entscheidungen, Atmosphaere, Nacht-Zyklus, Selbststaendigkeit
- **Dateien:** decision_engine.py, atmosphere_controller.py, homeostasis.py,
  night_cycle.py, autonomous_search.py, capability_monitor.py, introspection.py
- **Warum:** MOLOCH trifft eigenstaendig Entscheidungen (Musik, Atmosphaere,
  Suche). Das ist ein eigener Bereich, nicht Service.

### 11. memory
- **Domain:** Gedaechtnis, Lernen, Kontext-Speicher
- **Dateien:** longterm_memory.py, episodic_memory.py, persistent_memory.py,
  vector_memory.py, music_memory.py, daily_learner.py, preference_learner.py,
  temporal_memory.py, motor_learner.py, face_database.py
- **Warum:** 10 Dateien zum Thema Gedaechtnis — braucht eigenen Spezialisten.
  Service-Agent ist damit ueberfordert.

### 12. music
- **Domain:** Spotify, Musik-Reaktion, Beat-Erkennung, Visualisierung
- **Dateien:** spotify_bridge.py, spotify_controller.py, music_listener.py,
  music_memory.py, music_visualizer.py
- **Warum:** Musik ist eigenstaendig genug. Voice-Agent deckt Sprache ab,
  nicht Spotify-Integration und Beat-Reaktion.

### 13. awareness
- **Domain:** Kontext, Aktivitaet, Raum, Welt-Modell
- **Dateien:** context_evaluator.py, activity_analyzer.py, world_state.py,
  room_map.py, spatial_learning.py, near_field_handler.py, environment_watcher.py
- **Warum:** MOLOCH versteht seinen Raum und Kontext. Das ist weder Vision
  (die sieht nur Pixel) noch Service (der routet nur Daten).

## ZUSAMMENFASSUNG

| Status | Anzahl | Agenten |
|--------|--------|---------|
| Vorhanden | 8 | vision, hardware, gui, tracking, voice, service, unconscious, stresstest |
| Nur agents/ | 2 | deepseek, tentacle |
| Fehlt komplett | 5 | personality, autonomy, memory, music, awareness |
| **Gesamt Ziel** | **15** | **Volle Abdeckung aller core/ Dateien** |

## AKTION FUER PI-SESSION

Die Pi-Session kann die 5 fehlenden Agenten erstellen:
- Je 1 Datei in `.claude/agents/` (Claude Code Format, ~30 Zeilen)
- Je 1 Datei in `agents/` (Original AGENT_*.md Format, ~60 Zeilen)
- CLAUDE.md Agenten-Tabelle aktualisieren (8 → 15)
- deepseek + tentacle ebenfalls nach .claude/agents/ portieren
