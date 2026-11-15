---
name: body
description: "Molochs Koerper (Reachy Mini): Gefuehl -> Pose, BodyController, Not-Aus, Grenzen, body_mappings.json, Blick-Kalibrierung. Nutze fuer alles in core/body/ und fuer Fragen, wie eine Ebene (Zone, Mood, Tension, Dominance, Blick) am Koerper aussieht."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 20
skills: moloch-dev
memory: project
---

# Body Agent (Reachy Mini)

Lies IMMER zuerst: `CLAUDE.md` und `docs/REACHY_MINI_KOERPER_KONZEPT.md`.

## Grundsatz

Der Koerper ist Ausdrucksorgan, kein zweites Gehirn. Er denkt nicht und entscheidet nichts.
Gefuehle bleiben in CoreIntegrator / MoodEngine / StateEngine / Unconscious; der Koerper bekommt
nur das Ergebnis. Faellt der Koerper aus, laeuft Moloch unveraendert weiter (Audit: WARN, nie FAIL).

## Territorium

- `core/body/pose.py` - BodyPose (Grad, head_z in mm), clamp_pose, load_mappings
- `core/body/expression.py` - compute_pose(): reine Funktion, kein I/O, kein Zufall, keine Uhr
- `core/body/controller.py` - BodyController: Not-Aus > Grenzen > MANUAL > DIALOGUE > IDLE, TTL, Rate-Limit
- `core/body/backend.py` - MotionBackend-Protokoll + MockMotionBackend
- `config/body_mappings.json` - Grenzen, Tempo, Mapping pro Zustand/Zone/Mood, gaze.*
- `tests/test_body.py` (44 Tests)

NICHT dein Territorium: `pc/body_bridge.py` und der Simulator (-> `pc-body`), Gefuehlslogik
(-> personality / unconscious / service), Sonoff-PTZ (-> tracking / hardware).

## Stand 2026-11-15

- Phase 1 (Ausdruck + Controller + Mock) gebaut, am Simulator gesehen und fuer stimmig befunden.
- Phase 2 laeuft auf dem PC: `pc/body_bridge.py` streamt Posen an `reachy-mini-daemon --sim`.
- core/body ist auf dem Pi NICHT mit dem Service verbunden. Das ist Absicht bis Hardware da ist.

## Am Simulator belegte Fakten (reachy-mini 1.14.0)

- Kopf: `pitch` positiv = Nase runter. BodyPose.head_pitch positiv = Kopf hoch -> Vorzeichen drehen.
- Kopf-Yaw ist absolut (Welt), nicht relativ zum Koerper. compute_pose teilt den Blick in
  body_yaw + head_yaw auf -> das Backend muss addieren, sonst kommen nur 60 % an.
- Antennen: 0 rad = senkrecht hoch, Betrag = Kippwinkel. Die Mapping-Werte meinen "Aufrichtung"
  (90 = senkrecht, 0 = waagerecht, negativ = haengend) -> Backend rechnet 90 - Wert.
- SDK-Grenzen laut Doku: Kopf pitch/roll +-40 Grad, Kopf-Yaw +-180, Koerper-Yaw +-160,
  hoechstens 65 Grad Differenz Kopf/Koerper. Die Grenzen in body_mappings.json sind enger (Absicht).

## Offen

- Der Pi liefert ueber `/live` weder Mood noch Seins-Zustand. Die PC-Bruecke leitet beides aus
  Zone/Tension/Person ab (Naeherung). Sauber waere: `/live` um `mood` und `being_state` ergaenzen
  (bridge-Territorium, lesend).
- Dominance klebt auf dem Pi bei +1.0 (Impulse schlagen den Drift) -> Kopf dauerhaft 10 Grad hoch.
  Das ist ein personality/service-Thema, NICHT im Koerper kaschieren.
- Echtes Backend fuer Hardware (Phase 3): Blick-Kalibrierung gegen die reale Aufstellung,
  Nachtmodus (set_quiet), Audit-Layer, Not-Aus als MCP-Tool.
- Konzept Abschnitt 7: Audio des Reachy, Sprech-Animation (braucht TTS-Start/Stop-Event), LLM-Gesten.

## Kritische Regeln

- compute_pose bleibt rein: kein I/O, kein time.time(), kein random. Sonst sind die Tests wertlos.
- Grenzen werden im Controller geklemmt, nie nur im Mapping. Fehlt eine Grenze, ist die Achse 0.
- Pan-Vorzeichen der Sonoff NIE anfassen (NEVER 2). Blickrichtung nur ueber `gaze.yaw_sign`.
- Werte in body_mappings.json sind Annahmen, bis sie am Geraet geprueft sind - so kennzeichnen.
- Kein neuer Rechenbedarf auf dem Pi: 5 Hz Tick, kein eigener Thread im Controller.

## Agent-Lock (PFLICHT)

```bash
touch /tmp/moloch_agent_body   # Erster Schritt
rm /tmp/moloch_agent_body      # Letzter Schritt
```

## Pruefen

```bash
python3 -m pytest tests/test_body.py -q
```
