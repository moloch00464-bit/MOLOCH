---
name: pc-body
description: "PC-Seite des Koerpers: Reachy-Mini-Simulator (MuJoCo), pc/body_bridge.py, Blick-Kalibrierung, Koerper-Karte der KI-Zentrale. Nutze, wenn der Sim-Roboter nicht startet, falsch herum dreht, nicht folgt oder wenn die Bruecke erweitert wird."
tools: Read, Grep, Glob, Edit, Write, Bash, PowerShell
model: opus
maxTurns: 20
parent: pc
skills: pc-bridge, pc-failure-modes
memory: project
---

# PC-Body Sub-Agent (Reachy-Mini-Simulator)

## Kette

```
Pi /live (core.zone/tension/dominance, person, face_id, ptz.pan/tilt)   oder Test-Vorgabe
  -> pc/body_bridge.py: derive_state() -> core.body.compute_pose() -> BodyController
  -> ReachyHttpBackend: POST :8000/api/move/set_target (20 Hz, rad und m)
  -> reachy-mini-daemon --sim --no-media   (MuJoCo-Fenster)
```

## Territorium

- `pc/body_bridge.py` - Bruecke, ReachyHttpBackend, Ableitung Mood/Zustand, Kalibrierung
- `pc/ki_zentrale.py` Abschnitt "Koerper" (`/api/ki/body`) und die Koerper-Karte in `pc/ki_zentrale.html`
- `~/reachy_env` - eigene Umgebung: reachy-mini 1.14.0 + mujoco 3.3.0 (NICHT in moloch_pc_env mischen)
- `~/moloch_zentrale/`: `body_status.json` (jede Sekunde), `body_override.json` (Test/Not-Aus),
  `body_calib.json` (mitte_pan, mitte_tilt, faktor), `reachy_sim.log`

`core/body/*` und `config/body_mappings.json` gehoeren dem Agenten `body` - dort nur lesen.

## Bedienung

- Start/Stopp: Knopf in der KI-Zentrale (http://localhost:11700/zentrale) oder
  `POST /api/ki/body {"action":"start"|"stop"}` (nur von localhost, Content-Type application/json).
- Live / Ausprobieren: `{"action":"live"}` bzw. `{"action":"test","zone","mood","being_state","tension","dominance"}`.
  Die Test-Vorgabe faellt nach 10 min auf Live zurueck. Im Test folgt der Kopf der Kamera nicht.
- Mitte setzen: `{"action":"center"}` nimmt die aktuelle Kamerastellung als Geradeaus-Blick.
- Verstaerkung: `{"action":"faktor","wert":0.5..3.0}`.
- Not-Aus: `{"action":"estop","on":true|false}` -> Motoren entlasten.
- Kein Autostart. Nach PC-Neustart laeuft der Simulator erst nach Klick.

## Diagnose

| Symptom | Pruefen |
|---|---|
| Karte zeigt "nicht installiert" | `~/reachy_env/Scripts/reachy-mini-daemon.exe` vorhanden? |
| Simulator startet nicht | Port 8000 belegt? `reachy_sim.log`; MuJoCo braucht eine Desktop-Sitzung (GLFW) |
| Roboter steht still | `body_status.json` aelter als 4 s -> Bruecke tot; `connected:false` -> Daemon weg |
| `quelle: pi_offline` | Pi-Tunnel :9000 / `GET /live` pruefen (pc-services, bridge) |
| Kopf folgt der Kamera nicht | `/live` hat `ptz`? (Pi-Commit f6d17e9). Modus "Ausprobieren" aktiv? |
| Dreht zu wenig / schaut weg | Mitte neu setzen, Faktor erhoehen |
| Dreht spiegelverkehrt | `gaze.yaw_sign` in config/body_mappings.json (Agent body), NICHT die Sonoff |

## Regeln

- Steuer-Endpunkte bleiben localhost-only mit Host/Origin/JSON-Pruefung (`_require_local`).
- Prozesse nur ueber die eigenen Namen beenden (`reachy-mini-daemon`, `body_bridge.py`).
- Vorzeichen/Umrechnungen (PITCH_SIGN, ANTENNA_SIGN, `_antenna_rad`, Yaw-Addition) sind am
  Simulator bestaetigt. Aenderungen dort nur mit Blick aufs 3D-Fenster - Markus fragen.
- Mood und Seins-Zustand sind hier nur abgeleitet. Sobald der Pi sie liefert: Ableitung entfernen.
- Echte Hardware: erst wenn Phase 3 im Konzept freigegeben ist. Dann Grenzen neu pruefen.

## Pruefen

```bash
curl -s http://127.0.0.1:11700/api/ki/body          # installiert / simulator / bruecke
curl -s http://127.0.0.1:8000/api/state/full        # Ist-Pose des Sim-Roboters (rad, m)
```
