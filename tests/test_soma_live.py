"""Ende-zu-Ende ohne Hardware (Issue #5): Status-JSON -> /live.soma -> Body-Bridge-Rueckmeldung -> /live.koerper.

Braucht die Chat-Server-Abhaengigkeiten (fastapi, httpx), sonst uebersprungen.
Das echte /dev/shm wird nie beschrieben: open/getmtime werden auf tmp_path umgelenkt.
"""

import builtins
import json
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import core.awareness.soma as soma_mod  # noqa: E402
import core.bridge.chat_server as cs  # noqa: E402

REAL = "/dev/shm/moloch_status.json"


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = tmp_path / "status.json"
    real_open, real_mtime = builtins.open, os.path.getmtime

    def _open(path, *a, **kw):
        return real_open(str(fake) if path == REAL else path, *a, **kw)

    def _mtime(path):
        return real_mtime(str(fake) if path == REAL else path)

    monkeypatch.setattr(builtins, "open", _open)
    monkeypatch.setattr(os.path, "getmtime", _mtime)
    monkeypatch.setattr(soma_mod.SomaTracker, "read_override", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(soma_mod, "_tracker", soma_mod.SomaTracker())
    cs._BODY_REPORT.clear()

    def setze(temp, running=True):
        fake.write_text(json.dumps({"watchdog": {"cpu_temp": temp, "running": running,
                                                 "last_check": time.monotonic()}}))

    c = TestClient(cs.app, client=("127.0.0.1", 50000))
    c.setze = setze
    return c


def test_kette_temperatur_koerper_rueckmeldung(client):
    client.setze(55.0)
    live = client.get("/live").json()
    assert live["soma"]["zustand"] == "ok" and live["soma"]["quelle"] == "watchdog.cpu_temp"
    assert live["koerper"]["stufe"] == "AUSSTEHEND"

    client.setze(76.0)
    live = client.get("/live").json()
    k = live["soma"]["kennung"]
    assert live["soma"]["zustand"] == "strained"

    # alte Kennung zaehlt nicht
    assert client.post("/body/report", json={"kennung": "veraltet", "ergebnis": "sent"}).status_code == 200
    assert client.get("/live").json()["koerper"]["stufe"] == "AUSSTEHEND"

    client.post("/body/report", json={"kennung": k, "zustand": "strained", "ergebnis": "offline"})
    assert client.get("/live").json()["koerper"]["stufe"] == "ACCEPTED"

    client.post("/body/report", json={"kennung": k, "zustand": "strained", "ergebnis": "sent"})
    live = client.get("/live").json()
    assert live["koerper"]["stufe"] == "EXECUTED" and live["koerper"]["kennung"] == k


def test_watchdog_aus_ist_unknown(client):
    client.setze(55.0, running=False)
    assert client.get("/live").json()["soma"]["zustand"] == "unknown"


def test_body_report_validierung(client):
    assert client.post("/body/report", json={}).status_code == 422
    assert client.post("/body/report", json={"kennung": "x" * 40}).status_code == 422


def test_live_sinne_kamera(client, tmp_path):
    client.setze(55.0)
    k = client.get("/live").json()["sinne"]["kamera"]
    assert k["ok"] is None            # Testdatei hat weder camera_reachable noch frame_age -> unbekannt
    assert "NICHTS" in k["zeile"]
