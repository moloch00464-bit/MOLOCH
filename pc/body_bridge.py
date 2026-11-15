"""Koerper-Bruecke: Molochs Zustand -> core/body -> Reachy-Mini-Simulator (Phase 2 im Konzept).

    Pi /live (Zone, Tension, Dominance, Person)      oder Test-Vorgabe aus der KI-Zentrale
        -> core.body.compute_pose()  (Fables Mapping, config/body_mappings.json)
        -> core.body.BodyController  (Grenzen, Tempo, Not-Aus)
        -> ReachyHttpBackend         (REST des reachy-mini-daemon, :8000)

Start:  moloch_pc_env\\Scripts\\python.exe pc\\body_bridge.py
Der Simulator laeuft getrennt:  reachy_env\\Scripts\\reachy-mini-daemon.exe --sim --no-media

Dateien in ~/moloch_zentrale:
    body_override.json  {"zone","mood","being_state","tension","dominance","until": epoch, "estop": bool}
                        geschrieben von der KI-Zentrale; nach "until" gilt wieder der Live-Zustand
    body_status.json    jede Sekunde: Quelle, Zustand, Pose, Verbindung

NAEHERUNGEN (am Simulator pruefen, bevor echte Hardware dranhaengt):
- Der Pi liefert weder Mood noch Seins-Zustand; beides wird hier aus Zone/Tension/Person abgeleitet.
- Vorzeichen Kopf-Pitch und Antennen sind aus der SDK-Doku abgeleitet, nicht am Geraet belegt
  (PITCH_SIGN, ANTENNA_SIGN unten).
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.body import BodyController, BodyPose, BodyState, Priority, compute_pose, load_mappings  # noqa: E402
from core.body.pose import AXES  # noqa: E402

SIM_URL = os.environ.get("MOLOCH_REACHY_URL", "http://127.0.0.1:8000").rstrip("/")
PI_URL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000").rstrip("/")
DIR = Path(os.environ.get("MOLOCH_ZENTRALE_DIR", str(Path.home() / "moloch_zentrale")))
OVERRIDE_FILE = DIR / "body_override.json"
STATUS_FILE = DIR / "body_status.json"

LOOP_HZ = 20.0          # Ziel-Stream an den Simulator
LIVE_EVERY_S = 0.5      # Pi-Abfrage
PITCH_SIGN = -1.0       # BodyPose: + = Kopf hoch; Reachy (y nach links): + = Nase runter
ANTENNA_SIGN = (-1.0, 1.0)  # (rechts, links); Ruhewerte im Sim sind (-0.18, +0.18) rad

ZONES = ("guardian", "shadow", "berserker")
MOODS = ("calm", "focused", "alert", "agitated", "euphoric", "dark")
STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")


def _antenna_rad(value_deg: float) -> float:
    """Mapping-Wert -> Kippwinkel der Antenne.

    body_mappings.json meint "Aufrichtung": 90 = senkrecht hoch (alert), 0 = waagerecht,
    negativ = haengend (dark). Beim Reachy ist 0 rad = senkrecht hoch und der Betrag
    der Kippwinkel nach aussen. Am Simulator gesehen 2026-11-15: ohne Umrechnung
    kippten die Antennen umso weiter weg, je aufmerksamer Moloch war.
    """
    return math.radians(max(0.0, min(135.0, 90.0 - value_deg)))


class ReachyHttpBackend:
    """MotionBackend ueber die REST-Schnittstelle des reachy-mini-daemon.

    goto() merkt sich nur Ziel und Dauer; step() interpoliert und streamt per
    /api/move/set_target. So ueberlagern sich Atmung und Antennen-Wippen fluessig.
    """

    def __init__(self, url: str = SIM_URL):
        self._url = url
        self._http = httpx.Client(timeout=1.5)
        self._connected = False
        self._cur: Optional[dict] = None     # zuletzt gesendete Achswerte
        self._start: Optional[dict] = None
        self._target: Optional[dict] = None
        self._t0 = 0.0
        self._dur = 1.0

    def connect(self) -> bool:
        try:
            r = self._http.get(f"{self._url}/api/daemon/status")
            self._connected = r.status_code == 200
            if self._connected:
                self._http.post(f"{self._url}/api/motors/set_mode/enabled")
        except httpx.HTTPError:
            self._connected = False
        return self._connected

    def is_connected(self) -> bool:
        return self._connected

    def goto(self, pose: BodyPose) -> None:
        target = {a: float(getattr(pose, a)) for a in AXES}
        self._start = dict(self._cur) if self._cur else dict(target)
        self._target = target
        self._t0 = time.monotonic()
        self._dur = max(0.05, float(pose.duration_s))
        self.step()

    def stop(self) -> None:
        self._target = None
        self._http.post(f"{self._url}/api/motors/set_mode/disabled").raise_for_status()

    def step(self) -> None:
        """Einen interpolierten Zwischenschritt senden. Wirft bei Verbindungsverlust."""
        if not self._connected or self._target is None:
            return
        k = min(1.0, (time.monotonic() - self._t0) / self._dur)
        k = k * k * (3.0 - 2.0 * k)  # weich an- und abbremsen
        cur = {a: self._start[a] + (self._target[a] - self._start[a]) * k for a in AXES}
        body = {
            "target_head_pose": {
                "x": 0.0, "y": 0.0, "z": cur["head_z"] / 1000.0,
                "roll": math.radians(cur["head_roll"]),
                "pitch": math.radians(cur["head_pitch"]) * PITCH_SIGN,
                "yaw": math.radians(cur["head_yaw"]),
            },
            "target_antennas": [_antenna_rad(cur["antenna_r"]) * ANTENNA_SIGN[0],
                                _antenna_rad(cur["antenna_l"]) * ANTENNA_SIGN[1]],
            "target_body_yaw": math.radians(cur["body_yaw"]),
        }
        try:
            self._http.post(f"{self._url}/api/move/set_target", json=body).raise_for_status()
        except httpx.HTTPError:
            self._connected = False
            raise
        self._cur = cur


def _clamp(v, lo, hi, default=0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v)) if math.isfinite(v) else default


def derive_state(live: Optional[dict]) -> Optional[dict]:
    """Pi /live -> Felder fuer BodyState. Mood und Seins-Zustand sind abgeleitet (Naeherung)."""
    if not live:
        return None
    core = live.get("core") or {}
    zone = core.get("zone") if core.get("zone") in ZONES else "guardian"
    tension = _clamp(core.get("tension"), 0.0, 1.0)   # Pi liefert -1..+1, der Koerper nutzt 0..1
    if live.get("face_id"):
        being = "engaged"
    elif live.get("person_detected"):
        being = "observing"
    else:
        being = "idle"
    if zone == "berserker":
        mood = "agitated"
    elif zone == "shadow":
        mood = "dark" if tension < 0.4 else "alert"
    elif tension > 0.6:
        mood = "alert"
    elif tension > 0.3:
        mood = "focused"
    else:
        mood = "calm"
    # Blick folgt der Sonoff: Pan/Tilt des Trackers in Grad. Vorzeichen und Skalierung
    # stehen in config/body_mappings.json (gaze.*), nicht hier - die Sonoff bleibt unangetastet.
    ptz = live.get("ptz") or {}
    pan, tilt = ptz.get("pan"), ptz.get("tilt")
    return {"zone": zone, "mood": mood, "being_state": being, "tension": tension,
            "dominance": _clamp(core.get("dominance"), -1.0, 1.0),
            "gaze_yaw": _clamp(pan, -180.0, 180.0) if isinstance(pan, (int, float)) else None,
            "gaze_pitch": _clamp(tilt, -90.0, 90.0) if isinstance(tilt, (int, float)) else None}


def read_override() -> Optional[dict]:
    try:
        o = json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return o if isinstance(o, dict) else None


def override_state(o: dict) -> dict:
    return {
        "zone": o.get("zone") if o.get("zone") in ZONES else "guardian",
        "mood": o.get("mood") if o.get("mood") in MOODS else "calm",
        "being_state": o.get("being_state") if o.get("being_state") in STATES else "idle",
        "tension": _clamp(o.get("tension"), 0.0, 1.0),
        "dominance": _clamp(o.get("dominance"), -1.0, 1.0),
    }


def write_status(data: dict) -> None:
    try:
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATUS_FILE)
    except OSError:
        pass


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    mappings = load_mappings()
    backend = ReachyHttpBackend()
    ctrl = BodyController(backend, mappings)
    pi = httpx.Client(timeout=1.5)
    live_state: Optional[dict] = None
    last_live = last_status = 0.0
    estop_on = False
    t_start = time.monotonic()

    while True:
        loop_t = time.monotonic()
        if loop_t - last_live >= LIVE_EVERY_S:
            last_live = loop_t
            try:
                r = pi.get(f"{PI_URL}/live")
                live_state = derive_state(r.json()) if r.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                live_state = None

        o = read_override() or {}
        want_estop = bool(o.get("estop"))
        if want_estop != estop_on:
            ctrl.estop() if want_estop else ctrl.release_estop()
            estop_on = want_estop

        if _clamp(o.get("until"), 0, 1e12) > time.time():
            source, state = "test", override_state(o)
        elif live_state:
            source, state = "live", live_state
        else:
            source, state = "pi_offline", {"zone": "guardian", "mood": "calm",
                                           "being_state": "offline_anchor",
                                           "tension": 0.0, "dominance": 0.0}

        pose = compute_pose(BodyState(**state), mappings, loop_t - t_start)
        ctrl.submit(Priority.MANUAL if source == "test" else Priority.IDLE, pose, ttl_s=1.0)
        result = ctrl.tick()
        try:
            backend.step()
        except httpx.HTTPError:
            pass  # der Controller verbindet beim naechsten tick() neu

        if loop_t - last_status >= 1.0:
            last_status = loop_t
            write_status({"ts": time.time(), "quelle": source, "zustand": state,
                          "ergebnis": result, **ctrl.status()})

        time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - loop_t)))


if __name__ == "__main__":
    main()
