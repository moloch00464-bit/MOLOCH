"""Koerper-Bruecke: Molochs Zustand -> core/body -> Reachy-Mini-Simulator (Phase 2 im Konzept).

    Pi /live (Zone, Tension, Dominance, Person)      oder Test-Vorgabe aus der KI-Zentrale
        -> core.body.compute_pose()  (Fables Mapping, config/body_mappings.json)
        -> core.body.BodyController  (Grenzen, Tempo, Not-Aus)
        -> ReachyHttpBackend         (REST des reachy-mini-daemon, :8000)
        -> POST Pi /body/report      (was mit dem Soma-Zustand passiert ist, Issue #5)

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
CALIB_FILE = DIR / "body_calib.json"

LOOP_HZ = 20.0          # Ziel-Stream an den Simulator
LIVE_EVERY_S = 0.5      # Pi-Abfrage
PITCH_SIGN = -1.0       # BodyPose: + = Kopf hoch; Reachy (y nach links): + = Nase runter
ANTENNA_SIGN = (-1.0, 1.0)  # (rechts, links); Ruhewerte im Sim sind (-0.18, +0.18) rad

EVENT_HOLD_S = 8.0      # so lange praegt ein Ausloeser die Haltung nach dem Satz
# Anlass -> (Zone, Mood, Seins-Zustand) fuer die Dauer der Reaktion
EVENT_LOOK = {
    "markus_arrived": ("guardian", "alert", "engaged"),      # aufrichten, Antennen senkrecht
    "markus_left": ("guardian", "dark", "idle"),             # Kopf sinkt, Antennen haengen
    "unknown_person": ("shadow", "alert", "observing"),      # Kopf tief, Antennen steil, lauernd
    "spontan": ("guardian", "calm", "engaged"),              # normaler Kommentar: zuwenden
}

ZONES = ("guardian", "shadow", "berserker")
MOODS = ("calm", "focused", "alert", "agitated", "euphoric", "dark")
STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")
SOMA = ("ok", "strained", "distressed", "recovering", "unknown")
REPORT_EVERY_S = 5.0    # Rueckmeldung an den Pi: sofort bei neuer Kennung, sonst so oft
# Begruessung (Pi /live.begruessung, core/bridge/begruessung.py) -> Clip aus der Emotions-Bibliothek
CLIP_DATASET = "pollen-robotics/reachy-mini-emotions-library"
CLIP_MAX_S = 20.0       # spaetestens dann uebernimmt die Dauerhaltung wieder
CLIP_ALTER_MAX_S = 30.0  # aeltere Begruessungen (z. B. nach Bridge-Neustart) nicht nachholen
# Ueber den SSH-Tunnel (Loopback) braucht der Pi kein Token; direkt im LAN schon, wenn gesetzt.
REPORT_HEADERS = ({"X-Moloch-Token": os.environ["MOLOCH_API_TOKEN"]}
                  if os.environ.get("MOLOCH_API_TOKEN") else {})


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
                # BodyPose.head_yaw ist relativ zum Koerper gedacht (compute_pose teilt den
                # Blick auf); der Reachy nimmt die Kopfdrehung absolut -> Koerperanteil addieren.
                "yaw": math.radians(cur["head_yaw"] + cur["body_yaw"]),
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


def read_calib() -> dict:
    """Blick-Kalibrierung aus der KI-Zentrale ("Kamera schaut auf mich").

    mitte_pan/mitte_tilt: Kamerastellung, bei der der Roboter geradeaus schaut.
    faktor: Verstaerkung der Kamerabewegung (1.0 = Grad fuer Grad).
    """
    try:
        c = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        c = {}
    return {"mitte_pan": _clamp(c.get("mitte_pan"), -180.0, 180.0),
            "mitte_tilt": _clamp(c.get("mitte_tilt"), -90.0, 90.0),
            "faktor": _clamp(c.get("faktor"), 0.1, 3.0, default=1.0)}


def derive_state(live: Optional[dict], calib: Optional[dict] = None) -> Optional[dict]:
    """Pi /live -> Felder fuer BodyState. Mood und Seins-Zustand sind abgeleitet (Naeherung)."""
    if not live:
        return None
    calib = calib or {"mitte_pan": 0.0, "mitte_tilt": 0.0, "faktor": 1.0}
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
    # Proaktiver Satz vom Pi (/live.speech): waehrend er spricht nickt der Kopf leicht,
    # und der Anlass faerbt die Haltung fuer EVENT_HOLD_S Sekunden (Priority.DIALOGUE).
    dialog = None
    sp = live.get("speech") or {}
    age = sp.get("age_s")
    speaking = bool(sp.get("speaking")) and isinstance(age, (int, float)) and age < 30
    if isinstance(age, (int, float)) and (speaking or age < EVENT_HOLD_S):
        dialog = sp.get("event") or "spontan"
        zone, mood, being = EVENT_LOOK.get(dialog, (zone, mood, "engaged"))

    # Blick folgt der Sonoff: Pan/Tilt des Trackers in Grad. Vorzeichen und Skalierung
    # stehen in config/body_mappings.json (gaze.*), nicht hier - die Sonoff bleibt unangetastet.
    ptz = live.get("ptz") or {}
    pan, tilt = ptz.get("pan"), ptz.get("tilt")
    # Technischer Koerperzustand (Pi core/awareness/soma.py). Fehlt er -> unknown, kein Einfluss.
    soma = live.get("soma") or {}
    soma_zustand = soma.get("zustand") if soma.get("zustand") in SOMA else "unknown"
    return {"zone": zone, "mood": mood, "being_state": being, "tension": tension,
            "dominance": _clamp(core.get("dominance"), -1.0, 1.0),
            "speaking": speaking, "_dialog": dialog,
            "soma": soma_zustand, "_soma_kennung": soma.get("kennung"),
            "gaze_yaw": _clamp((pan - calib["mitte_pan"]) * calib["faktor"], -180.0, 180.0)
            if isinstance(pan, (int, float)) else None,
            "gaze_pitch": _clamp((tilt - calib["mitte_tilt"]) * calib["faktor"], -90.0, 90.0)
            if isinstance(tilt, (int, float)) else None}


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
        "soma": o.get("soma") if o.get("soma") in SOMA else "ok",
    }


def neue_begruessung(live_raw, letzte_kennung: Optional[str], jetzt: float) -> Optional[dict]:
    """Begruessung aus Pi /live, wenn neu (andere Kennung) und frisch genug fuer einen Clip."""
    b = live_raw.get("begruessung") if isinstance(live_raw, dict) else None
    if not isinstance(b, dict) or not b.get("clip") or b.get("kennung") == letzte_kennung:
        return None
    if jetzt - float(b.get("ts") or 0) > CLIP_ALTER_MAX_S:
        return None
    return b


def soma_report(kennung: Optional[str], zustand: str, quelle: str, ergebnis: str) -> Optional[dict]:
    """Rueckmeldung fuer den Pi. Nur fuer einen echten Soma-Zustand vom Pi (Kennung vorhanden).

    Bei Test-Vorgabe der KI-Zentrale hat der Koerper den Pi-Zustand NICHT dargestellt ->
    ergebnis "ueberschrieben" (der Pi wertet das als ACCEPTED, nicht EXECUTED).
    """
    if not kennung:
        return None
    return {"kennung": kennung, "zustand": zustand, "ts": time.time(),
            "ergebnis": ergebnis if quelle == "live" else "ueberschrieben"}


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
    live_fehler = 0   # Pi-Abfragen ohne Ergebnis in Folge
    live_state: Optional[dict] = None
    last_live = last_status = 0.0
    estop_on = False
    t_start = time.monotonic()
    soma_kennung, soma_zustand = None, "unknown"
    last_report, versucht_kennung, wirksam, wirksam_kennung = 0.0, None, "offline", None
    letzte_meldung = None   # HTTP-Status oder Fehler der letzten Rueckmeldung (body_status.json)
    sim = httpx.Client(timeout=3.0)
    begr_kennung, clip_bis, clip_seit, letzter_clip_check, clip_uuid = None, 0.0, 0.0, 0.0, None

    while True:
        loop_t = time.monotonic()
        if loop_t - last_live >= LIVE_EVERY_S:
            last_live = loop_t
            try:
                r = pi.get(f"{PI_URL}/live")
                raw = r.json() if r.status_code == 200 else None
                live_state = derive_state(raw, read_calib()) if raw else None
            except (httpx.HTTPError, ValueError):
                raw, live_state = None, None
            b = neue_begruessung(raw, begr_kennung, time.time())
            if b:
                begr_kennung = b.get("kennung")
                try:
                    rp = sim.post(f"{SIM_URL}/api/move/play/recorded-move-dataset/{CLIP_DATASET}/{b['clip']}")
                    if rp.status_code == 200:
                        clip_uuid = (rp.json() or {}).get("uuid")
                        clip_seit, clip_bis = loop_t, loop_t + CLIP_MAX_S
                except (httpx.HTTPError, ValueError):
                    pass
            # Nach einem Neustart von Pi oder Tunnel kann eine Keep-alive-Verbindung haengen
            # bleiben (2026-11-17: Bridge blieb auf pi_offline, frischer Client ging sofort).
            # Nach 10 Fehlversuchen in Folge (5 s) den Client neu aufbauen.
            live_fehler = 0 if live_state else live_fehler + 1
            if live_fehler >= 10:
                pi.close()
                pi = httpx.Client(timeout=1.5)
                live_fehler = 0
            soma_kennung = (live_state or {}).get("_soma_kennung")
            soma_zustand = (live_state or {}).get("soma", "unknown")

        o = read_override() or {}
        want_estop = bool(o.get("estop"))
        if want_estop != estop_on:
            ctrl.estop() if want_estop else ctrl.release_estop()
            estop_on = want_estop
            if want_estop and clip_bis and clip_uuid:
                clip_bis = 0.0
                try:   # Begruessungs-Clip laeuft im Daemon, nicht im Controller
                    sim.post(f"{SIM_URL}/api/move/stop", json={"uuid": clip_uuid})
                except httpx.HTTPError:
                    pass

        if _clamp(o.get("until"), 0, 1e12) > time.time():
            source, state = "test", override_state(o)
        elif live_state:
            source, state = "live", live_state
        else:
            source, state = "pi_offline", {"zone": "guardian", "mood": "calm",
                                           "being_state": "offline_anchor",
                                           "tension": 0.0, "dominance": 0.0}

        state = dict(state)
        dialog = state.pop("_dialog", None)
        state.pop("_soma_kennung", None)
        # Laeuft ein Begruessungs-Clip, keine Dauerhaltung streamen (die wuerde ihn ueberfahren).
        # Ende: Daemon meldet keinen laufenden Clip mehr (fruehestens nach 1 s) oder CLIP_MAX_S.
        if clip_bis and not estop_on and loop_t - letzter_clip_check >= 0.5:
            letzter_clip_check = loop_t
            try:
                if loop_t - clip_seit > 1.0 and not sim.get(f"{SIM_URL}/api/move/running").json():
                    clip_bis = 0.0
            except (httpx.HTTPError, ValueError):
                pass
        if clip_bis and loop_t < clip_bis and not estop_on:
            result = "clip"
            state["ereignis"] = "begruessung"
        else:
            clip_bis = 0.0
            pose = compute_pose(BodyState(**state), mappings, loop_t - t_start)
            prio = (Priority.MANUAL if source == "test"
                    else Priority.DIALOGUE if dialog else Priority.IDLE)
            ctrl.submit(prio, pose, ttl_s=1.0)
            state["ereignis"] = dialog
            result = ctrl.tick()
            try:
                backend.step()
            except httpx.HTTPError:
                pass  # der Controller verbindet beim naechsten tick() neu
        if result not in ("rate_limited", "clip"):
            # rate_limited sagt nichts: die letzte Pose steht weiter. Das Ergebnis gehoert
            # zu der Kennung, mit der diese Pose berechnet wurde.
            wirksam, wirksam_kennung = result, soma_kennung

        # Erst melden, wenn eine Pose MIT der aktuellen Kennung gelaufen ist - sonst ginge das
        # Ergebnis der alten Pose als EXECUTED fuer den neuen Zustand raus. Neue Kennung: ein
        # Versuch sofort, danach (auch nach Fehlern, z. B. 401) nur alle REPORT_EVERY_S.
        if soma_kennung and wirksam_kennung == soma_kennung and (
                soma_kennung != versucht_kennung or loop_t - last_report >= REPORT_EVERY_S):
            rep = soma_report(soma_kennung, soma_zustand, source, wirksam)
            last_report, versucht_kennung = loop_t, soma_kennung
            try:
                letzte_meldung = pi.post(f"{PI_URL}/body/report", json=rep,
                                         headers=REPORT_HEADERS).status_code
            except httpx.HTTPError as e:
                letzte_meldung = type(e).__name__  # Pi weg: dort nach 20 s UNBESTAETIGT

        if loop_t - last_status >= 1.0:
            last_status = loop_t
            write_status({"ts": time.time(), "quelle": source, "zustand": state,
                          "ergebnis": result, "soma_meldung": letzte_meldung, **ctrl.status()})

        time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - loop_t)))


if __name__ == "__main__":
    main()
