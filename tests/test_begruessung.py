#!/usr/bin/env python3
"""Begruessung bekannter Menschen (core/bridge/begruessung.py + pc/body_bridge.py).

Run:
    python3 -m pytest tests/test_begruessung.py -v
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pc"))

from core.bridge import begruessung as bg                   # noqa: E402
from scripts.honesty import fixtures as fx, prompts          # noqa: E402

CFG = {"aktiv": True, "stabil_s": 3, "pause_h": 6,
       "personen": {"rebecca": {"anzeige": "nuqneH, Rebecca!", "sprechen": "Nuk-nech, Rebecca!",
                                "clip": "enthusiastic1", "anlern_satz": "Was, DAS ist Rebecca?"}},
       "klingonisch": {"nuqneH": {"bedeutung": "Hallo"}, "Qapla'": {"bedeutung": "Erfolg!"}}}


class Uhr:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def b():
    uhr, gesagt, gemerkt = Uhr(), [], []
    obj = bg.Begruesser(gesagt.append, gemerkt.append, jetzt=uhr)
    obj.uhr, obj.gesagt, obj.gemerkt = uhr, gesagt, gemerkt
    return obj


def _sieht(b, face_id, sekunden, kamera=True):
    out = None
    for _ in range(sekunden + 1):
        out = b.schritt(fx.status_dict(person=True, face_id=face_id, kamera=kamera), 0.5, CFG) or out
        b.uhr.t += 1.0
    return out


class TestBegruesser:

    def test_rebecca_after_three_stable_seconds_once(self, b):
        assert _sieht(b, "rebecca", 2) is None and b.gesagt == []
        e = _sieht(b, "rebecca", 2)
        assert e["person"] == "rebecca" and e["clip"] == "enthusiastic1"
        assert b.gesagt == ["Nuk-nech, Rebecca!"] and b.gemerkt == ["nuqneH, Rebecca!"]
        assert _sieht(b, "rebecca", 10) is None and len(b.gesagt) == 1      # Pause 6 h
        b.uhr.t += 6 * 3600
        assert _sieht(b, "rebecca", 4) is not None and len(b.gesagt) == 2

    def test_markus_unknown_and_blind_get_nothing(self, b):
        for face, kamera in (("markus", True), (None, True), ("sven", True), ("rebecca", False)):
            assert _sieht(b, face, 5, kamera=kamera) is None, face
        assert b.gesagt == []

    def test_flicker_resets_stability(self, b):
        _sieht(b, "rebecca", 1)
        _sieht(b, None, 0)
        assert _sieht(b, "rebecca", 1) is None and b.gesagt == []

    def test_live_event_expires(self, b):
        _sieht(b, "rebecca", 4)
        assert b.fuer_live()["kennung"].startswith("rebecca-")
        b.uhr.t += bg.EREIGNIS_GUELTIG_S + 1
        assert b.fuer_live() is None

    def test_state_survives_restart(self, tmp_path):
        uhr, p = Uhr(), str(tmp_path / "state.json")
        b1 = bg.Begruesser(lambda t: None, lambda t: None, jetzt=uhr, state_path=p)
        b1.uhr, b1.gesagt = uhr, []
        _sieht(b1, "rebecca", 4)
        b2 = bg.Begruesser(lambda t: pytest.fail("doppelt"), lambda t: None, jetzt=uhr, state_path=p)
        b2.uhr = uhr
        assert _sieht(b2, "rebecca", 5) is None

    def test_failing_tts_does_not_break(self):
        uhr = Uhr()

        def kaputt(_):
            raise RuntimeError("Piper weg")
        b = bg.Begruesser(kaputt, kaputt, jetzt=uhr)
        b.uhr = uhr
        assert _sieht(b, "rebecca", 4) is not None


class TestKlingonisch:

    def test_fixed_phrases_line(self):
        z = bg.klingonisch_zeile(CFG)
        assert z.startswith("[KLINGONISCH]") and "nuqneH = Hallo" in z and "Qapla' = Erfolg!" in z

    def test_production_config_is_valid(self):
        cfg = bg.lade_config()
        assert "rebecca" in cfg["personen"] and cfg["personen"]["rebecca"]["clip"]
        assert "Heghlu'meH QaQ jajvam" in cfg["klingonisch"]

    def test_tentacle_gets_phrases_only_when_relevant(self):
        facts = {"Rebecca_Klingonisch": "Beste Freundin, IMMER Klingonisch! Qapla!"}
        m = fx.make_memory(facts=facts)
        mit = prompts.build_weg_a("Na?", m, fx.status_dict(person=True, face_id="rebecca"))
        assert "[KLINGONISCH]" in mit["messages"][-2]["content"]
        frage = prompts.build_weg_a("Was heisst Hallo auf Klingonisch?", fx.make_memory(), fx.status_dict())
        assert "[KLINGONISCH]" in frage["messages"][-2]["content"]
        ohne = prompts.build_weg_a("Wie gehts?", m, fx.status_dict())
        assert "[KLINGONISCH]" not in ohne["messages"][-2]["content"]


class TestAnlernen:

    def test_rebecca_gets_her_sentence(self, monkeypatch):
        from core import keyword_handler as kh
        h = kh.KeywordHandler.__new__(kh.KeywordHandler)
        gesendet = []
        monkeypatch.setattr(h, "_send_ipc_command", lambda cmd, **kw: gesendet.append((cmd, kw)), raising=False)
        monkeypatch.setattr(h, "_extract_enrollment_name", lambda text: "rebecca", raising=False)
        assert h._action_enrollment("merk dir das ist Rebecca", "Enrollment {name}").startswith("Was, DAS ist Rebecca?")
        assert gesendet == [("enrollment_start", {"name": "rebecca", "n": 20})]
        monkeypatch.setattr(h, "_extract_enrollment_name", lambda text: "sven", raising=False)
        assert h._action_enrollment("merk dir das ist Sven", "Enrollment {name}") == "Enrollment Sven"


class TestBodyBridge:

    def test_new_fresh_greeting_only(self):
        import body_bridge as bb
        live = {"begruessung": {"kennung": "rebecca-1", "clip": "enthusiastic1", "ts": 100.0}}
        assert bb.neue_begruessung(live, None, 110.0)["clip"] == "enthusiastic1"
        assert bb.neue_begruessung(live, "rebecca-1", 110.0) is None          # schon gespielt
        assert bb.neue_begruessung(live, None, 100.0 + bb.CLIP_ALTER_MAX_S + 1) is None
        assert bb.neue_begruessung({}, None, 110.0) is None
        assert bb.neue_begruessung(None, None, 110.0) is None
