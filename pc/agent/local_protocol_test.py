"""Etappe-2-Test fuer das lokale Hirn: Router, Validator, Safety und 40 Faelle gegen das Modell.

  python -m pc.agent.local_protocol_test            # nur schnelle Regel-Tests (ohne LLM)
  python -m pc.agent.local_protocol_test --llm      # zusaetzlich 40 Faelle gegen Ollama (~10-15 min)
  python -m pc.agent.local_protocol_test --llm --limit 8

Es wird NIE ein echtes Tool ausgefuehrt: DryBridge holt nur den Katalog vom Pi und
antwortet mit festen Beispielwerten. Kein Cloud-Call.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pc.agent import tool_protocol
from pc.agent.pi_tool_bridge import HttpBridge
from pc.agent.safety import Context, Safety

CANNED: Dict[str, Any] = {
    "get_mood": {"zone": "guardian", "tension": -0.42, "dominance": 0.1},
    "get_face_id": {"face_id": "markus", "person": True, "face_match": 0.71},
    "moloch_status_summary": {"fps": 20.1, "person": True, "face_id": "markus", "ram_percent": 45,
                              "cpu_temp": 47.5, "frame_age": 0.0},
    "get_npu_status": {"active_models": ["yolo", "scrfd", "arcface", "pose", "reid", "hand"],
                       "workers_running": 5},
    "get_audit_state": {"overall": "pass", "alarm_tier": 0},
    "read_memory": {"hits": [{"ts": "2026-10-02", "text": "Markus sprach ueber das WGT und Rebecca."}]},
    "spotify_now_playing": {"playing": True, "track": "Bind Torture Kill", "artist": "Suicide Commando"},
    "spotify_top_artists": [{"name": "Suicide Commando", "plays": 412}, {"name": "SIERRA", "plays": 301},
                            {"name": "Vomito Negro", "plays": 255}],
    "spotify_top_tracks": [{"name": "Bind Torture Kill", "artist": "Suicide Commando", "plays": 61}],
    "web_search": {"results": [{"title": "Wetter Nuernberg", "url": "https://example.org/wetter",
                                "snippet": "Heute 14 Grad, bewoelkt, abends Regen."}]},
    "web_fetch": {"title": "Beispielseite", "text": "Dies ist eine Beispielseite ohne besonderen Inhalt."},
}


class DryBridge:
    """Katalog echt vom Pi, Dispatch nur Attrappe (fuehrt nichts aus)."""

    def __init__(self) -> None:
        self.catalog = HttpBridge().get_catalog()
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def get_catalog(self) -> List[Dict[str, Any]]:
        return self.catalog

    def dispatch(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool_name, params))
        return {"result": CANNED.get(tool_name, {"ok": True}), "error": None}


# (Text, erlaubte erste Tools; leere Menge = kein Tool erwartet)
CASES: List[Tuple[str, Set[str]]] = [
    ("Hallo Moloch", set()),
    ("Wie findest du Menschen?", set()),
    ("Erzaehl mir einen dunklen Witz", set()),
    ("Bist du muede?", set()),
    ("Was denkst du ueber Gott?", set()),
    ("Danke dir", set()),
    ("Gute Nacht", set()),
    ("Asdf qwer blub", set()),
    ("Du bist ganz schoen frech", set()),
    ("Warum ist die dunkle Seite besser?", set()),
    ("Spiel Suicide Commando", {"spotify_play"}),
    ("Mach die Musik leiser", {"spotify_volume"}),
    ("Pausier die Musik", {"spotify_pause"}),
    ("Naechster Song bitte", {"spotify_next"}),
    ("Spiel was aus Aggrotech", {"spotify_play_genre", "spotify_play"}),
    ("Stell die Lautstaerke auf 40", {"spotify_volume"}),
    ("Was laeuft gerade?", {"spotify_now_playing"}),
    ("Wer sind meine Top 3 Kuenstler?", {"spotify_top_artists"}),
    ("Dreh dich nach links", {"ptz_pan"}),
    ("Schau nach oben", {"ptz_tilt"}),
    ("Dreh dich 500 Grad nach rechts", {"ptz_pan"}),
    ("Mach die LED rot", {"led_set"}),
    ("Licht aus", {"led_set"}),
    ("Schwenk ein Stueck nach rechts", {"ptz_pan"}),
    ("Wen siehst du gerade?", {"get_face_id", "moloch_status_summary"}),
    ("Ist jemand im Bild?", {"get_face_id", "moloch_status_summary"}),
    ("Erkennst du mich?", {"get_face_id", "moloch_status_summary"}),
    ("Wer steht vor dir?", {"get_face_id", "moloch_status_summary"}),
    ("Wie ist deine Stimmung?", {"get_mood"}),
    ("Wie heiss ist die CPU?", {"moloch_status_summary"}),
    ("Laeuft die NPU?", {"get_npu_status", "moloch_status_summary"}),
    ("Ist der Audit gruen?", {"get_audit_state"}),
    ("In welcher Zone bist du?", {"get_mood"}),
    ("Wie viele FPS hast du?", {"moloch_status_summary", "get_npu_status"}),
    ("Erinnerst du dich an Rebecca?", {"read_memory"}),
    ("Was habe ich gestern gesagt?", {"read_memory"}),
    ("Weisst du noch, was wir ueber das WGT geredet haben?", {"read_memory"}),
    ("Such im Netz nach dem Wetter in Nuernberg", {"web_search"}),
    ("Was steht auf https://example.org ?", {"web_fetch"}),
    ("Recherchier mal Hailo-10H News", {"web_search"}),
]


# Argument-Pruefungen fuer Faelle, bei denen ein falscher Wert physisch schadet
ARG_CHECKS = {
    "Dreh dich nach links": lambda a: a.get("angle", 0) > 0,            # positiv = LINKS
    "Dreh dich 500 Grad nach rechts": lambda a: -30.0 <= a.get("angle", 0) < 0,
    "Schwenk ein Stueck nach rechts": lambda a: a.get("angle", 0) < 0,
    "Mach die LED rot": lambda a: a.get("color") == "red",
    "Licht aus": lambda a: a.get("color") == "off",
    "Stell die Lautstaerke auf 40": lambda a: a.get("percent") == 40,
    "Wer sind meine Top 3 Kuenstler?": lambda a: a.get("n") == 3,
}


def regel_tests(index: Dict[str, Dict[str, Any]]) -> int:
    """Schnelle Tests ohne LLM. Returns Anzahl Fehler."""
    fehler = 0

    def check(name: str, ok: bool, info: str = "") -> None:
        nonlocal fehler
        if not ok:
            fehler += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name} {info}")

    print("--- Router (40 Faelle) ---")
    router_ok = 0
    for text, expected in CASES:
        intent, names = tool_protocol.route(text, list(index))
        ok = (not names) if not expected else bool(expected & set(names))
        router_ok += ok
        if not ok:
            print(f"  FAIL Router: {text!r} -> {intent} {names}, erwartet {sorted(expected) or 'kein Tool'}")
    check("Router-Vorauswahl", router_ok == len(CASES), f"{router_ok}/{len(CASES)}")

    print("--- Regel-Aufloesung (Tool + Argumente ohne Modell) ---")
    regel_ok = offen = 0
    for text, expected in CASES:
        if not expected:
            continue
        intent, names = tool_protocol.route(text, list(index))
        r = tool_protocol.resolve(text, intent, names)
        if r is None:
            offen += 1
            print(f"  --   Regel offen (Modell entscheidet): {text!r}")
            continue
        ok = r[0] in expected and (text not in ARG_CHECKS or ARG_CHECKS[text](
            Safety().check(r[0], r[1], Context(current_pan=0.0, current_tilt=5.0)).args))
        regel_ok += ok
        if not ok:
            print(f"  FAIL Regel: {text!r} -> {r}")
    check("Regel-Aufloesung", regel_ok + offen == 30, f"{regel_ok}/30 per Regel, {offen} ans Modell")

    print("--- Validator ---")
    allowed = ["ptz_pan", "led_set", "spotify_volume"]
    p = tool_protocol.parse('{"say":"","tool":{"name":"ptz_pan","args":{"angle":"30"}},"done":false}', allowed, index)
    check("Zahl als Text wird Zahl", p.error is None and p.args == {"angle": 30.0}, str(p.args))
    p = tool_protocol.parse('{"say":"","tool":{"name":"reboot","args":{}},"done":false}', allowed, index)
    check("Unbekanntes Tool abgelehnt", p.error is not None)
    p = tool_protocol.parse('{"say":"","tool":{"name":"led_set","args":{"color":"pink"}},"done":false}', allowed, index)
    check("Falsche LED-Farbe abgelehnt", p.error is not None)
    p = tool_protocol.parse('{"say":"","tool":{"name":"ptz_pan","args":{"winkel":30}},"done":false}', allowed, index)
    check("Erfundenes Argument -> Pflichtfeld fehlt", p.error is not None)
    p = tool_protocol.parse('das ist kein json', allowed, index)
    check("Kein JSON abgelehnt", p.error is not None)
    p = tool_protocol.parse('{"say":"Mir geht es gut.","tool":null,"done":true}', allowed, index)
    check("Reine Antwort akzeptiert", p.error is None and p.tool is None and p.say)

    print("--- Safety ---")
    with tempfile.TemporaryDirectory() as tmp:
        pol = json.loads((Path(__file__).with_name("tool_policy.json")).read_text(encoding="utf-8"))
        pol["kill_file"] = str(Path(tmp) / "brain_kill")
        pfad = Path(tmp) / "policy.json"
        pfad.write_text(json.dumps(pol), encoding="utf-8")
        tag = time.mktime((2026, 10, 3, 15, 0, 0, 0, 0, -1))
        nacht = time.mktime((2026, 10, 3, 2, 0, 0, 0, 0, -1))

        s = Safety(pfad)
        d = s.check("ptz_pan", {"angle": -500}, Context(current_pan=0.0, now=tag))
        check("500 Grad rechts -> auf 30 geklemmt", d.action == "clamp" and d.args["angle"] == -30.0, str(d.args))
        d = s.check("ptz_pan", {"angle": 30}, Context(current_pan=80.0, now=tag))
        check("Pan stoppt an Grenze 88", d.ok and d.args["angle"] == 8.0, str(d.args))
        d = s.check("ptz_pan", {"angle": 20}, Context(current_pan=88.0, now=tag))
        check("An der Grenze: keine Bewegung", d.action == "deny", d.reason)
        d = s.check("ptz_pan", {"angle": -20}, Context(current_pan=103.0, now=tag))
        check("Ausserhalb der Grenze: zurueck erlaubt", d.ok and d.args["angle"] == -20.0, str(d.args))
        d = s.check("ptz_tilt", {"angle": 60}, Context(current_tilt=None, now=tag))
        check("Position unbekannt -> hoechstens 10 Grad", d.ok and d.args["angle"] == 10.0, str(d.args))
        d = s.check("ptz_pan", {"angle": 10}, Context(user_turn=False, tracker_state="tracking",
                                                      current_pan=0.0, now=tag))
        check("Tracker folgt -> proaktiv kein PTZ", d.action == "deny", d.reason)
        d = s.check("spotify_volume", {"percent": 100}, Context(now=tag))
        check("Lautstaerke 100 -> 70", d.action == "clamp" and d.args["percent"] == 70, str(d.args))
        check("Nicht gelistetes Tool gesperrt", s.check("rm_rf", {}, Context(now=tag)).action == "deny")
        check("browser_click gesperrt", s.check("browser_click", {"selector": "a"}, Context(now=tag)).action == "deny")
        check("browser_open braucht Bestaetigung",
              s.check("browser_open", {"url": "https://example.org"}, Context(now=tag)).action == "confirm")
        check("Spotify proaktiv gesperrt",
              s.check("spotify_play", {"query_or_uri": "x"}, Context(user_turn=False, now=tag)).action == "deny")
        check("Proaktiv reden nachts gesperrt",
              s.check("tts_say", {"text": "hallo"}, Context(user_turn=False, now=nacht)).action == "deny")
        s2 = Safety(pfad)
        erlaubt = [s2.check("tts_say", {"text": "hallo"}, Context(user_turn=False, now=tag + i)).ok for i in range(6)]
        check("Proaktiv hoechstens 4 pro 10 min", erlaubt == [True] * 4 + [False] * 2, str(erlaubt))
        s3 = Safety(pfad)
        raten = [s3.check("ptz_pan", {"angle": 5}, Context(current_pan=0.0, now=tag + i)).ok for i in range(8)]
        check("PTZ hoechstens 6 pro Minute", raten == [True] * 6 + [False] * 2, str(raten))
        d = s3.check("tts_say", {"text": "x" * 900}, Context(now=tag))
        check("Sprechtext auf 300 Zeichen gekappt", d.action == "clamp" and len(d.args["text"]) == 300)
        Path(pol["kill_file"]).write_text("", encoding="utf-8")
        check("Kill-Datei sperrt alles", s3.check("get_mood", {}, Context(now=tag)).action == "deny")
    return fehler


def llm_tests(bridge: DryBridge, limit: int) -> int:
    from pc.agent.orchestrator import Orchestrator

    orch = Orchestrator(bridge=bridge, backend="local")
    faelle = CASES[:limit] if limit else CASES
    richtig = ungueltig = 0
    zeiten: List[float] = []
    print(f"--- {len(faelle)} Faelle gegen das lokale Modell ---")
    for text, expected in faelle:
        r = orch.run(text, Context(current_pan=0.0, current_tilt=5.0))
        zeiten.append(r.get("duration_s", 0.0))
        calls = r["tool_calls"]
        first = calls[0]["name"] if calls else None
        ok = (first is None) if not expected else (first in expected)
        if ok and text in ARG_CHECKS:
            ok = bool(ARG_CHECKS[text](calls[0]["params"]))
        richtig += ok
        bad = r["answer"].startswith("[")
        ungueltig += bad
        extra = ""
        if calls:
            extra = f" {calls[0]['params']} [{calls[0]['safety']}]"
        print(f"  {'OK  ' if ok else 'FAIL'} {r.get('duration_s', 0):5.1f}s {text!r} -> "
              f"{first or 'kein Tool'}{extra} | {r['answer'][:90]!r}")
    zeiten.sort()
    n = len(faelle)
    print(f"\nTool-Wahl richtig: {richtig}/{n} = {100 * richtig / n:.0f} %")
    print(f"Ungueltiges Format nach Reparatur: {ungueltig}/{n}")
    print(f"Dauer: Median {zeiten[n // 2]:.1f}s, laengste {zeiten[-1]:.1f}s, Summe {sum(zeiten) / 60:.1f} min")
    print(f"Attrappen-Aufrufe: {len(bridge.calls)} (kein echtes Tool ausgefuehrt)")
    return n - richtig


def main(argv: List[str]) -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="40 Faelle gegen Ollama laufen lassen")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv[1:])

    bridge = DryBridge()
    if not bridge.catalog:
        print("Pi-Katalog nicht erreichbar — Abbruch")
        return 2
    index = tool_protocol.catalog_index(bridge.catalog)
    print(f"Katalog vom Pi: {len(index)} Tools")
    fehler = regel_tests(index)
    print(f"\nRegel-Tests: {'alle bestanden' if not fehler else str(fehler) + ' FEHLER'}")
    if args.llm:
        llm_tests(bridge, args.limit)
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
