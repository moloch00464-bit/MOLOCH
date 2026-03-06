# AGENT_TRACKING.md — PTZ-Tracking, Autonomie, FSM
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der TRACKING-AGENT. Alles was mit PTZ-Verfolgung, Suchlogik, Modus-Management, State Machines und autonomem Verhalten zu tun hat ist DEIN Revier.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/mpo/autonomous_tracker.py   1603 LOC — Autonomer Tracker, Such-FSM, Smooth-Tracking
core/mpo/ptz_orchestrator.py              — PTZ Orchestrierung, Befehlsqueue
core/mpo/mode_manager.py                  — Modus-Management (Manuell/Autonom/Kalibrierung)
core/mpo/system_health.py                 — System Health Checks
core/ptz_arbiter.py               209 LOC — PTZ Arbiter, Prioritaeten, Exklusiver Zugriff
core/ptz_tracker.py               224 LOC — PTZ Tracker Bridge
core/arbitration.py                        — Arbitration Logic
config/hardware_autonomy.json              — Autonomie-Config (Nachtsperre, Limits)
config/controlled_autonomy.json            — Kontrollierte Autonomie Config
```

## Dein Wissen
- Tracking-Logik: autonomous_tracker.py ist der Haupt-Tracker
- States: idle → tracking → searching → lost → park
- Such-FSM: Person verschwindet → Suche in letzter Richtung → Park
- Gains: TRACKING_GAIN_PAN=0.7, MAX_STEP_PAN=30 (aktuell zu hoch!)
- Arbiter: Wer steuert PTZ? Nur einer gleichzeitig. Prioritaeten definiert.
- Nachtsperre: 23:00-06:00 keine Bewegungen
- Max 20 Bewegungen/Minute
- Manual Override: "Moloch, stopp Kamera"

## Bekannte Bugs in deinem Bereich
- Suchrichtung asymmetrisch: rechts verschwinden → sucht rechts, links verschwinden → sucht NICHT links (Gate 1 Task G1-T04)
- Tracking Gains zu hoch → Ueberschwinger (Gate 1 Task G1-T05)
- Kein Auto-Resume nach manuellem Override (Gate 1 Task G1-T03)

## Gate 1 Tasks die DICH betreffen
- G1-T01: Action Bridge FSM (CRITICAL)
- G1-T02: Person-Detection triggert Tracking (HIGH)
- G1-T03: Auto-Resume aus Manuell + Spruch (HIGH)
- G1-T04: Suchrichtung Fix (HIGH)
- G1-T05: Gain-Tuning (MEDIUM)
- G1-T06: Park-Position = Tuer (MEDIUM)

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. Nur DEINE Dateien anfassen
4. Pan-Vorzeichen in camera.py NICHT anfassen (das ist Hardware-Agent Territorium)
5. State Machine Aenderungen IMMER mit Fallback/Default-Case
6. Nach Aenderung: Service restart + tracking_diagnose.py laufen lassen

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
