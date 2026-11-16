#!/usr/bin/env python3
"""
Sprechbewegung des Koerpers (Paket P4): Sprechstatus aus /live.speech -> Nicken.

Prueft die Kette Body-Bridge (derive_state) -> compute_pose ohne Roboter.
Braucht httpx (PC-Abhaengigkeit der Body-Bridge), sonst uebersprungen.

Run:
    python3 -m pytest tests/test_body_speaking.py -v
"""

import os
import sys

import pytest

pytest.importorskip("httpx")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

import body_bridge as bb                                   # noqa: E402
from core.body import BodyState, compute_pose, load_mappings   # noqa: E402

CALIB = {"mitte_pan": 0.0, "mitte_tilt": 0.0, "faktor": 1.0}


def _live(speech=None):
    live = {"core": {"tension": 0.2, "dominance": 0.0, "zone": "guardian"},
            "ptz": {"pan": 0.0, "tilt": 0.0}}
    if speech is not None:
        live["speech"] = speech
    return live


def _state(live):
    try:
        return bb.derive_state(live, CALIB)
    except TypeError:
        return bb.derive_state(live)


def _pitch_swing(state):
    """Wie stark der Kopf ueber eine Sekunde nickt."""
    maps = load_mappings()
    s = {k: v for k, v in state.items() if not k.startswith("_")}
    pitches = [compute_pose(BodyState(**s), maps, t / 20).head_pitch for t in range(20)]
    return max(pitches) - min(pitches)


def test_chat_answer_makes_the_head_nod():
    """Sprechstatus aus POST /speech/state (event 'chat') wirkt wie ein proaktiver Satz."""
    state = _state(_live({"event": "chat", "speaking": True, "age_s": 1.0, "source": "pc_zentrale"}))
    assert state["speaking"] is True
    assert state["being_state"] == "engaged"
    assert _pitch_swing(state) > 2.0


def test_no_speech_means_no_nodding():
    state = _state(_live())
    assert state["speaking"] is False
    assert _pitch_swing(state) < 0.5


def test_speaking_end_stops_nodding():
    state = _state(_live({"event": "chat", "speaking": False, "age_s": 0.5}))
    assert state["speaking"] is False


def test_forgotten_end_expires():
    """Kommt nie ein 'Ende', nickt der Koerper trotzdem nicht ewig."""
    state = _state(_live({"event": "chat", "speaking": True, "age_s": 45.0}))
    assert state["speaking"] is False


# === Soma (Issue #5): /live.soma -> BodyState.soma -> Rueckmeldung ===

def test_soma_aus_live():
    live = _live()
    live["soma"] = {"zustand": "strained", "kennung": "k1"}
    st = bb.derive_state(live, CALIB)
    assert st["soma"] == "strained" and st["_soma_kennung"] == "k1"


def test_soma_fehlt_oder_kaputt_ist_unknown():
    assert bb.derive_state(_live(), CALIB)["soma"] == "unknown"
    live = _live()
    live["soma"] = {"zustand": "brennt"}
    assert bb.derive_state(live, CALIB)["soma"] == "unknown"


def test_soma_erreicht_die_pose():
    maps = load_mappings()
    live = _live()
    live["soma"] = {"zustand": "distressed", "kennung": "k2"}
    st = bb.derive_state(live, CALIB)
    st.pop("_dialog"); st.pop("_soma_kennung")
    ok = dict(st, soma="ok")
    assert compute_pose(BodyState(**st), maps, 0.0).head_pitch < compute_pose(BodyState(**ok), maps, 0.0).head_pitch


def test_soma_report():
    assert bb.soma_report(None, "ok", "live", "sent") is None
    r = bb.soma_report("k3", "strained", "live", "sent")
    assert r["kennung"] == "k3" and r["ergebnis"] == "sent"
    assert bb.soma_report("k3", "strained", "test", "sent")["ergebnis"] == "ueberschrieben"
