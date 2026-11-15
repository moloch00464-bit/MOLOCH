#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruefung nach dem Merge von ccr-5df7b577-w92x94 - auf dem Pi ausfuehren.

Testet genau das, was in der Cloud-Session NICHT pruefbar war. Rein lesend:
schreibt nichts nach Qdrant, aendert keine Einstellungen, bewegt keine Kamera.
Einzige Netzwerk-Aktion: ein paar kurze Test-Saetze an das PC-Ollama zum Einbetten.

    cd ~/moloch && python3 scripts/check_after_merge.py

Ersetzt NICHT moloch_audit.py --auto.
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATUS_PATH = "/dev/shm/moloch_status.json"
results = []


def check(name):
    """Dekorator: fuehrt die Pruefung aus, faengt alles ab, merkt das Ergebnis."""
    def wrap(fn):
        try:
            level, detail = fn()
        except Exception as e:
            level, detail = "FAIL", f"{type(e).__name__}: {str(e)[:160]}"
        results.append(level)
        print(f"[{level:4}] {name}: {detail}")
        return fn
    return wrap


@check("Import core.vision (war seit 30.03. kaputt)")
def _():
    import core.vision  # noqa: F401
    return "PASS", "ok"


@check("Import core.inference_engine")
def _():
    import core.inference_engine  # noqa: F401
    return "PASS", "ok"


@check("Status-Datei frisch")
def _():
    age = time.time() - os.stat(STATUS_PATH).st_mtime
    return ("PASS" if age < 5 else "FAIL"), f"{age:.1f}s alt"


@check("PTZ-Position im Status (braucht die PTZ-Klemmung)")
def _():
    with open(STATUS_PATH) as f:
        ptz = json.load(f).get("ptz", {}) or {}
    pan, tilt = ptz.get("current_pan"), ptz.get("current_tilt")
    if pan is None or tilt is None:
        return "WARN", "current_pan/current_tilt fehlen - Klemmung greift nur ohne engere Grenzen"
    return "PASS", f"pan={pan} tilt={tilt}"


@check("PTZ-Grenzen aus settings.json")
def _():
    from core.agent.tools import hardware as hw
    return "PASS", f"pan={hw._axis_limits('pan')} tilt={hw._axis_limits('tilt')}"


@check("Qdrant-Collection moloch_memory")
def _():
    import requests
    r = requests.get("http://localhost:6333/collections/moloch_memory", timeout=5)
    if r.status_code == 404:
        return "WARN", "Collection fehlt - anlegen nur mit MOLOCH_QDRANT_CREATE_COLLECTION=1"
    res = r.json().get("result", {})
    size = res.get("config", {}).get("params", {}).get("vectors", {}).get("size")
    level = "PASS" if size == 384 else "FAIL"
    return level, f"Dimension {size}, {res.get('points_count')} Punkte"


def _provider():
    """Provider wie konfiguriert, aber unabhaengig vom enabled-Schalter."""
    from core.memory import embedding_provider as ep
    cfg = ep.load_config()
    if not cfg.get("base_url"):
        raise RuntimeError("kein Host (embedding_bridge.host oder tentacle_llm.host)")
    return ep.RemoteEmbeddingProvider(cfg, 384), cfg


@check("Embedding vom PC-Ollama (Antwortformat, Dimension, Dauer)")
def _():
    p, cfg = _provider()
    t0 = time.monotonic()
    vec = p.embed("Hallo Moloch, das ist ein Test.")
    ms = (time.monotonic() - t0) * 1000
    if vec is None:
        return "FAIL", (f"keine Antwort von {cfg['base_url']}{cfg['path']} - laeuft Ollama, "
                        f"ist '{cfg['model']}' geladen (ollama pull {cfg['model']})?")
    return "PASS", f"{len(vec)} Dimensionen in {ms:.0f} ms, Modell {cfg['model']}"


@check("Deutsch-Test: findet die Frage den richtigen Fakt?")
def _():
    p, cfg = _provider()
    facts = [
        "Markus_Lieblingsmusik: Dark Wave und Industrial",
        "Rebecca: spricht Klingonisch, beste Freundin von Markus",
        "Markus_Beruf: arbeitet mit Industrieanlagen und Automatisierung",
        "Moloch_Kamera: Sonoff CAM-PT2 mit Schwenk- und Neigefunktion",
        "Markus_Haustier: hat eine Katze namens Luna",
    ]
    questions = [
        ("Welche Musik hoert Markus gern?", 0),
        ("Wer kann Klingonisch?", 1),
        ("Was macht Markus beruflich?", 2),
        ("Mit welcher Kamera siehst du?", 3),
        ("Hat Markus ein Tier?", 4),
    ]
    fvecs = [p.embed(f) for f in facts]
    if any(v is None for v in fvecs):
        return "FAIL", "Embedding nicht verfuegbar"
    hits, low, lines = 0, 0, []
    for q, want in questions:
        qv = p.embed(q)
        scores = [sum(a * b for a, b in zip(qv, fv)) for fv in fvecs]
        best = max(range(len(scores)), key=scores.__getitem__)
        hits += best == want
        low += scores[want] < float(cfg.get("recall_min_score", 0.45))
        lines.append(f"      {'ok ' if best == want else 'NEIN'} {scores[want]:.2f}  {q}")
    print("\n".join(lines))
    detail = (f"{hits}/5 richtig, {low}/5 unter der Recall-Schwelle "
              f"{cfg.get('recall_min_score', 0.45)}")
    if hits == 5 and low == 0:
        return "PASS", detail
    if hits >= 4:
        return "WARN", detail + " - Schwelle senken oder mehrsprachiges Modell pruefen"
    return "FAIL", detail + " - Modell taugt nicht fuer Deutsch, mehrsprachiges 384-dim-Modell nehmen"


if __name__ == "__main__":
    fails, warns = results.count("FAIL"), results.count("WARN")
    print(f"\n{results.count('PASS')} PASS, {warns} WARN, {fails} FAIL")
    print("Danach: python3 ~/moloch/moloch_audit.py --auto")
    sys.exit(1 if fails else 0)
