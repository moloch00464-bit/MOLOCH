# AGENT HANDOFF — Gate 1
# Geschrieben: 2026-03-06 ~08:30 UTC
# Naechste Instanz: Lies CLAUDE.md, dann diese Datei

## AKTUELLER STAND

Gate 1 | Tasks T02-T06 DONE, T03 teilweise, T01+T07-T11 OFFEN | NICHT DEPLOYED

## WAS DIESE SESSION ERLEDIGT HAT

### Agententeam aufgesetzt (COMMITTED)
- 7 Domain-Agenten unter ~/moloch/agents/
- CLAUDE.md auf Gate 1.0 aktualisiert
- Commit: 14d641a

### G1-T04: Suchrichtung Fix — DONE (NICHT COMMITTED)
- Datei: core/mpo/autonomous_tracker.py ~Zeile 1337
- Patrol startet bei Position naechst zur letzten Kamera-Pan
- Vorher: Immer bei Index 0 (Home), egal wohin Person verschwand

### G1-T05: Gain-Tuning — DONE (NICHT COMMITTED)
- Datei: core/mpo/autonomous_tracker.py ~Zeile 93
- pan_gain 0.25→0.20, tilt_gain 0.20→0.15, max_step 5→4/3→2.5
- tracking_speed 0.7→0.6, cooldown 400→500ms

### G1-T06: Park-Position = Tuer — DONE (NICHT COMMITTED)
- Datei: core/mpo/autonomous_tracker.py
- park_pan=-120.0, park_tilt=0.0 (Tuer ist links)
- Park-Aufrufe nutzen config.park_pan statt hardcoded (0,0)

### G1-T02: Person-Detection triggert Tracking — FUNKTIONIERT BEREITS
- Kein Code noetig, Flow existiert

### G1-T03: Auto-Resume + Spruch — TEILWEISE (NICHT COMMITTED)
- ptz_arbiter.py: on_auto_resume Callback
- moloch_service.py: Callback → speak_event(TRACKING_RESUMED)
- personality_engine.py: TRACKING_RESUMED Event + 3 Sprueche

## GEAENDERTE DATEIEN (UNCOMMITTED!)
- core/mpo/autonomous_tracker.py (T04+T05+T06)
- core/ptz_arbiter.py (T03)
- core/moloch_service.py (T03)
- core/personality/personality_engine.py (T03)

## NAECHSTE SCHRITTE FUER NEUE INSTANZ
1. git add -A && git commit -m "Gate 1: T04+T05+T06 done, T03 Auto-Resume"
2. sudo systemctl restart moloch && sleep 10
3. Verify: systemctl is-active moloch + journalctl check
4. Testen: Person verschwinden lassen, Suchrichtung pruefen
5. G1-T01 (Action Bridge FSM) als naechstes

## OFFENE TASKS
| ID | Prio | Status |
|----|------|--------|
| G1-T01 | CRITICAL | OFFEN — Action Bridge FSM |
| G1-T07 | MEDIUM | OFFEN — Silence-Level Sensor |
| G1-T08 | MEDIUM | OFFEN — Auto-Enrollment via Chat |
| G1-T09 | MEDIUM | OFFEN — NPU-Dashboard im Panel |
| G1-T10 | LOW | OFFEN — Tension-Popup Farben |
| G1-T11 | LOW | OFFEN — Labelme Kalibrierung |

## SERVICE-STATUS
- War AKTIV bei Session-Start (TAPPAS 20 FPS)
- Aenderungen NICHT deployed
- Face-ID noch kaputt (sim=0.200, BLOCKER aus Gate 0.5)
