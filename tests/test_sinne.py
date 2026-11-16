"""Tests fuer core/awareness/sinne.py - Kamera weg darf nie als "niemand im Bild" im Prompt landen."""

from core.awareness.sinne import kamera_zustand


def _st(reachable=True, frame_age=0.2):
    return {"watchdog": {"camera_reachable": reachable}, "frame_age": frame_age}


def test_alles_da():
    k = kamera_zustand(_st(), 1.0)
    assert k["ok"] is True and k["zeile"] is None


def test_kamera_nicht_erreichbar():
    k = kamera_zustand(_st(reachable=False), 1.0)
    assert k["ok"] is False and "NICHTS" in k["zeile"]


def test_pipeline_liefert_nichts():
    assert kamera_zustand(_st(frame_age=-1), 1.0)["ok"] is False
    k = kamera_zustand(_st(frame_age=340.0), 1.0)
    assert k["ok"] is False and "340" in k["grund"]


def test_status_veraltet_oder_fehlt_ist_unbekannt():
    assert kamera_zustand(_st(), 60.0)["ok"] is None
    assert kamera_zustand(None, None)["ok"] is None
    k = kamera_zustand({"watchdog": {}}, 1.0)
    assert k["ok"] is None and k["zeile"]
