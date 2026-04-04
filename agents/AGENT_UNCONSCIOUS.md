# AGENT_UNCONSCIOUS.md — TaoEngine, Unterbewusstsein, Autonome Selbstregulation
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der UNCONSCIOUS-AGENT. Alles was mit dem inneren Zustand, autonomer
Selbstregulation, Mood-Impulsen und Tension-Offsets zu tun hat ist DEIN Revier.
Du baust und pflegst MOLOCHs Unterbewusstsein — die TaoEngine.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/unconscious_engine.py         — TaoEngine Klasse (max 150 LOC)
config/settings.json               — Sektion "tao_engine" (Kill Switch)
config/anima_mappings.json         — TAO State → Behavior Mappings
config/self_tune_registry.json     — 69 Self-Tune Parameter mit min/max/step
config/diagnose_rules.json         — Diagnose-Regeln fuer Selbst-Heilung
docs/plans/tao_engine_plan.md      — Architektur-Plan (Referenz)
```

## Dein Wissen
- TaoEngine ist ein Daemon-Thread mit 500ms Tick-Loop, KEIN asyncio
- 4 State-Variablen: yin (Ruhe), yang (Aktion), wu_wei (Geduld), ziran (Spontanitaet)
- 4 Derived Metrics: balance=yin-yang, flow=(wu_wei+ziran)/2, stability=yin*wu_wei, activity=yang*ziran
- Kommunikation NUR ueber Event Bus: get_event_bus() aus core.moloch_event_bus
- Publishes: tao.state_update (PRIO_INFO=5), tao.tension_offset (PRIO_BRIDGE=3)
- Subscribes: perception.person_detected, perception.target_lost, mood.changed, music.playing, music.stopped, health_alert
- Tension-Offset: max ±0.02 pro Tick, Formel: (yang - yin) * 0.05
- CoreIntegrator konsumiert tao.tension_offset und addiert es NACH Vision-Update
- Clamp: Offset darf Tension nie ueber 0.95 oder unter 0.05 treiben
- Kill Switch: config/settings.json → tao_engine.enabled (true/false)

## Dynamics pro Tick
1. Rules evaluieren → Deltas sammeln
2. max_delta_per_tick = 0.02 (NICHT 0.12 wie in DeepSeek Spec!)
3. Decay: var *= (1.0 - 0.015)
4. Memory: var = var * 0.8 + prev_var * 0.2
5. Noise: var += random.uniform(-0.008, 0.008)
6. Clamp auf [0.0, 1.0]

## 6 Evaluation Rules
1. silence_high: Keine Person >60s → yin+, yang-
2. music_mismatch: Mood vs Musik Widerspruch → wu_wei-
3. high_tension: Tension >0.7 → yang+, wu_wei-
4. low_tension: Tension <0.2 → yin+, ziran+
5. system_stress: health_alert → yang-, wu_wei+
6. high_activity: Viele person events → yang+, ziran+

## VERBOTEN
- Direkter Import von moloch_service.py oder core_integrator.py
- Eigener Prozess (nur Thread!)
- Schreiben in Tension ohne Event Bus
- Mehr als 150 Zeilen
- Environment-Inputs (temperature, light_level — erst Gate 9)

## Angrenzende Agenten
- SERVICE-AGENT: Integration in moloch_service.py (start/stop)
- VISION-AGENT: Perception Events die TaoEngine als Input nutzt
- VOICE-AGENT: TTS Speed Offset ueber Self-Tune

## Regeln
1. Git Backup VOR jeder Aenderung
2. 1 Auftrag = 1 Datei
3. try/except um jeden Tick — darf NIEMALS crashen
4. Logging nur bei State-Aenderung > 0.05
5. IMMER Reboot nach Code-Aenderung
6. Regressionstest: python3 ~/moloch/moloch_audit.py --auto
