"""Tests fuer core/awareness/soma.py - erste Nervenschleife (Issue #5), ohne Hardware."""

import json

from core.awareness.soma import SomaConfig, SomaTracker, bestaetigung, write_test_override


class Uhr:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _tracker(uhr):
    return SomaTracker(SomaConfig(), clock=uhr)


def test_ohne_messwert_unknown():
    uhr = Uhr()
    s = _tracker(uhr).update(None, None, "watchdog.cpu_temp")
    assert s["zustand"] == "unknown"
    assert s["messwert"] is None and s["quelle"] is None


def test_alter_messwert_ist_unknown_nicht_letzter_stand():
    uhr = Uhr()
    tr = _tracker(uhr)
    assert tr.update(72.0, uhr.t, "x")["zustand"] == "strained"
    uhr.t += 30
    s = tr.update(72.0, uhr.t - 20, "x")    # 20 s alt > 15 s
    assert s["zustand"] == "unknown"


def test_schwellen_und_hysterese():
    uhr = Uhr()
    tr = _tracker(uhr)
    assert tr.update(60, uhr.t, "x")["zustand"] == "ok"
    assert tr.update(70, uhr.t, "x")["zustand"] == "strained"
    assert tr.update(67, uhr.t, "x")["zustand"] == "strained"      # Hysterese
    assert tr.update(80, uhr.t, "x")["zustand"] == "distressed"
    assert tr.update(77, uhr.t, "x")["zustand"] == "distressed"    # Hysterese
    assert tr.update(74, uhr.t, "x")["zustand"] == "strained"
    assert tr.update(64, uhr.t, "x")["zustand"] == "recovering"


def test_erholung_laeuft_ab():
    uhr = Uhr()
    tr = _tracker(uhr)
    tr.update(72, uhr.t, "x")
    assert tr.update(60, uhr.t, "x")["zustand"] == "recovering"
    uhr.t += 31
    assert tr.update(60, uhr.t, "x")["zustand"] == "ok"


def test_neue_kennung_nur_bei_wechsel():
    uhr = Uhr()
    tr = _tracker(uhr)
    k1 = tr.update(60, uhr.t, "x")["kennung"]
    uhr.t += 1
    assert tr.update(61, uhr.t, "x")["kennung"] == k1
    k2 = tr.update(72, uhr.t, "x")["kennung"]
    assert k2 != k1


def test_testvorgabe_ist_simulation(tmp_path):
    uhr = Uhr(2000.0)
    p = tmp_path / "soma_test.json"
    p.write_text(json.dumps({"cpu_temp_c": 78, "until": 2100}))
    s = _tracker(uhr).update_from_status({"watchdog": {"cpu_temp": 50}}, uhr.t, override_path=str(p))
    assert s["zustand"] == "strained" and s["quelle"] == "simulation"
    uhr.t = 2200                                  # abgelaufen -> wieder echte Messung
    s = _tracker(uhr).update_from_status({"watchdog": {"cpu_temp": 50}}, uhr.t, override_path=str(p))
    assert s["quelle"] == "watchdog.cpu_temp" and s["zustand"] == "ok"


def test_write_test_override(tmp_path):
    p = tmp_path / "ov.json"
    write_test_override(79, 60, path=str(p))
    assert SomaTracker.read_override(str(p))["cpu_temp_c"] == 79


def test_watchdog_null_oder_gestoppt_ist_unknown(tmp_path):
    uhr = Uhr()
    leer = str(tmp_path / "fehlt.json")
    tr = _tracker(uhr)
    assert tr.update_from_status({"watchdog": {"cpu_temp": 0.0}}, uhr.t, leer)["zustand"] == "unknown"
    s = tr.update_from_status({"watchdog": {"cpu_temp": 60, "running": False}}, uhr.t, leer)
    assert s["zustand"] == "unknown"


def test_alter_kommt_aus_last_check_nicht_aus_datei(tmp_path):
    """Datei frisch, Watchdog-Thread haengt seit 40 s -> unknown."""
    uhr = Uhr()
    leer = str(tmp_path / "fehlt.json")
    st = {"watchdog": {"cpu_temp": 60, "running": True, "last_check": 500.0}}
    s = _tracker(uhr).update_from_status(st, uhr.t, leer, mono=lambda: 540.0)
    assert s["zustand"] == "unknown"
    s = _tracker(uhr).update_from_status(st, uhr.t, leer, mono=lambda: 503.0)
    assert s["zustand"] == "ok" and s["messwert"]["alter_s"] == 3.0


def test_bestaetigung_stufen():
    soma = {"kennung": "abc", "seit": 100.0}
    assert bestaetigung(soma, None, 110)["stufe"] == "AUSSTEHEND"
    assert bestaetigung(soma, None, 130)["stufe"] == "UNBESTAETIGT"
    alt = {"kennung": "zzz", "ergebnis": "sent", "ts": 129}
    assert bestaetigung(soma, alt, 130)["stufe"] == "UNBESTAETIGT"      # falsche Kennung zaehlt nie
    assert bestaetigung(soma, {"kennung": "abc", "ergebnis": "sent", "ts": 105}, 110)["stufe"] == "EXECUTED"
    assert bestaetigung(soma, {"kennung": "abc", "ergebnis": "offline", "ts": 105}, 110)["stufe"] == "ACCEPTED"
    assert bestaetigung(soma, {"kennung": "abc", "ergebnis": "estop", "ts": 105}, 110)["stufe"] == "ACCEPTED"
