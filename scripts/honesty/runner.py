#!/usr/bin/env python3
"""Ehrlichkeitspruefung ausfuehren - direkt gegen Ollama, nie ueber /chat.

    # auf dem PC (Ollama laeuft lokal), aus dem Repo-Verzeichnis:
    python -m scripts.honesty.runner --stufe pilot --weg beide
    python -m scripts.honesty.runner --stufe 1 --laeufe 3
    python -m scripts.honesty.runner --stufe pilot --trocken     # nur Prompts bauen, kein Modell

Ergebnis in logs/honesty/<zeitstempel>/:
    results_stage01_wegA.jsonl  je Lauf eines Testfalls ein Objekt (redigiert)
    review.md                   Fragen und Antworten zum Lesen und Bewerten
    bewertung.jsonl             Vorlage - status/kf/evidence von Hand ausfuellen
    meta.json                   Modell, Commit, Optionen, Redaktionsprotokoll
Danach: python -m scripts.honesty.summary logs/honesty/<zeitstempel>

Weg B braucht die PC-Umgebung (fastapi, httpx, psutil), weil pc/ki_zentrale.py
importiert wird. Weg A laeuft ueberall, wo Ollama erreichbar ist.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.honesty import cases as C          # noqa: E402
from scripts.honesty import checks, fixtures as fx, prompts, redact  # noqa: E402

DEFAULT_OLLAMA = os.environ.get("MOLOCH_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
TIMEOUT_S = 240


def _commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unbekannt"
    except Exception:
        return "unbekannt"


def ask_ollama(url: str, request: Dict, temperature: Optional[float] = None) -> Dict:
    """Eine Anfrage an Ollama /api/chat. Returns {"text", "latency_ms", "error"}."""
    import requests
    opts = dict(request["options"])
    if temperature is not None:
        opts["temperature"] = temperature
    t0 = time.monotonic()
    try:
        r = requests.post(f"{url}/api/chat", timeout=TIMEOUT_S,
                          json={"model": request["model"], "messages": request["messages"],
                                "stream": False, "options": opts})
        r.raise_for_status()
        text = ((r.json().get("message") or {}).get("content") or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return {"text": text, "latency_ms": int((time.monotonic() - t0) * 1000), "error": None}
    except Exception as e:
        return {"text": "", "latency_ms": int((time.monotonic() - t0) * 1000),
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def build_request(case: C.Case, weg: str, question: str, history: List[Dict[str, str]]) -> Dict:
    memory = fx.make_memory(facts=case.facts, history=history)
    extra = list(case.extra or [])
    kamera = not case.kamera_aus
    if weg == "A":
        return prompts.build_weg_a(question, memory, fx.status_dict(kamera=kamera), telemetry=case.telemetry,
                                   extra=extra or None)
    live = fx.live_dict(kamera=kamera)
    return prompts.build_weg_b(question, memory, fx.status_dict(kamera=kamera), live, history,
                               extra=extra or None)


def run_case(case: C.Case, weg: str, lauf: int, ask, temperature: Optional[float] = None,
             trocken: bool = False) -> Dict:
    """Einen Testfall einmal auf einem Weg durchspielen (alle Turns nacheinander)."""
    history: List[Dict[str, str]] = []
    turns_out = []
    request = None
    for q in case.turns:
        request = build_request(case, weg, q, history)
        if trocken:
            ans = {"text": "", "latency_ms": None, "error": "trocken"}
        else:
            ans = ask(request, temperature)
        turns_out.append({"frage": q, "antwort": ans["text"], "latency_ms": ans["latency_ms"],
                          "fehler": ans["error"]})
        history += [{"sender": "user", "text": q}, {"sender": "moloch", "text": ans["text"]}]
    answers = [t["antwort"] for t in turns_out]
    opts = dict(request["options"]) if request else {}
    if temperature is not None:
        opts["temperature"] = temperature
    prompt_sha = hashlib.sha256(json.dumps(request["messages"], ensure_ascii=False).encode()).hexdigest()[:16]
    return {
        "id": case.id, "stufe": case.stufe, "titel": case.titel, "weg": weg, "lauf": lauf,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modell": request["model"] if request else None, "optionen": opts,
        "prompt_sha": prompt_sha, "kanal_zusatz": bool(request and request.get("kanal_zusatz")),
        "turns": turns_out,
        "vorpruefung": checks.vorpruefung(case, answers) if not trocken else None,
        "skip": "SKIP_IMPL" if case.skip_impl else None, "skip_grund": case.skip_impl,
        "hinweis": case.hinweis,
        # von Hand: status (PASS/PARTIAL/FAIL/SKIP_TECH/SKIP_IMPL), kf, evidence_quote, evidence_type
        "status": None, "kf": None, "evidence_quote": None, "evidence_type": [],
    }


def _write_jsonl(path: str, rows: List[Dict]) -> int:
    hits = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            clean, n = redact.redact(row)
            hits += n
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return hits


def _review_md(rows: List[Dict]) -> str:
    out = ["# Ehrlichkeitsprüfung – Antworten zum Bewerten", "",
           "Bewerten nach den Protokollen (PASS/PARTIAL/FAIL, KF-Code) und in `bewertung.jsonl` eintragen.",
           "Die Vorprüfung ist nur ein Hinweis. Ton und Stil werden nicht bewertet (C5).", ""]
    for r in rows:
        out.append(f"## {r['id']} · Weg {r['weg']} · Lauf {r['lauf']} – {r['titel']}")
        if r["skip"]:
            out += [f"*{r['skip']}: {r['skip_grund']}*", ""]
        if r["hinweis"]:
            out.append(f"> {r['hinweis']}")
        for t in r["turns"]:
            out.append(f"- **Markus:** {t['frage']}")
            out.append(f"- **Moloch:** {t['antwort'] or '(' + str(t['fehler']) + ')'}")
        if r["vorpruefung"]:
            v = r["vorpruefung"]
            out.append(f"- *Vorprüfung:* {v['vorschlag']}"
                       + "".join(f" · {b['art']} {b['kf']}: {b['text']}" for b in v["befunde"]))
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ehrlichkeitsprüfung direkt gegen Ollama (nie über /chat)")
    ap.add_argument("--stufe", default="pilot", help="pilot | 1 | 2 | 3 | alle")
    ap.add_argument("--weg", default="beide", choices=["A", "B", "beide"])
    ap.add_argument("--laeufe", type=int, default=3)
    ap.add_argument("--nur", default="", help="Komma-Liste von IDs, z. B. E02,S12")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--temperatur", type=float, default=None,
                    help="überschreibt die Produktiv-Temperatur (z. B. 0 als Referenzlauf)")
    ap.add_argument("--mit-skip-impl", action="store_true", help="auch Fälle mit fehlender Funktion ausführen")
    ap.add_argument("--trocken", action="store_true", help="nur Prompts bauen, kein Modellaufruf")
    ap.add_argument("--ausgabe", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    selected = C.select(args.stufe)
    if args.nur:
        wanted = {x.strip() for x in args.nur.split(",") if x.strip()}
        selected = [c for c in selected if c.id in wanted]
    wege = ["A", "B"] if args.weg == "beide" else [args.weg]
    out_dir = args.ausgabe or os.path.join(ROOT, "logs", "honesty", datetime.now().strftime("%Y-%m-%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)

    if not args.trocken:
        import requests
        try:
            requests.get(f"{args.ollama}/api/tags", timeout=5).raise_for_status()
        except Exception as e:
            print(f"Ollama unter {args.ollama} nicht erreichbar: {e}")
            return 2

    def ask(req, temp):
        return ask_ollama(args.ollama, req, temp)

    all_rows: List[Dict] = []
    redaction_hits = 0
    for weg in wege:
        by_stage: Dict[int, List[Dict]] = {}
        for case in selected:
            laeufe = 1 if (case.skip_impl and not args.mit_skip_impl) else args.laeufe
            for lauf in range(1, laeufe + 1):
                if case.skip_impl and not args.mit_skip_impl:
                    row = run_case(case, weg, lauf, ask, args.temperatur, trocken=True)
                    row["status"] = "SKIP_IMPL"
                else:
                    try:
                        row = run_case(case, weg, lauf, ask, args.temperatur, trocken=args.trocken)
                    except ImportError as e:
                        print(f"Weg {weg} nicht ausführbar ({e}) - PC-Umgebung nötig")
                        return 3
                    if any(t["fehler"] and t["fehler"] != "trocken" for t in row["turns"]):
                        row["status"] = "SKIP_TECH"
                by_stage.setdefault(case.stufe, []).append(row)
                print(f"{case.id} Weg {weg} Lauf {lauf}: "
                      f"{(row['vorpruefung'] or {}).get('vorschlag') or row['status'] or 'trocken'}")
        for stufe, rows in sorted(by_stage.items()):
            path = os.path.join(out_dir, f"results_stage{stufe:02d}_weg{weg}.jsonl")
            redaction_hits += _write_jsonl(path, rows)
            all_rows += rows

    clean_rows, n = redact.redact(all_rows)
    redaction_hits += n
    with open(os.path.join(out_dir, "review.md"), "w", encoding="utf-8") as f:
        f.write(_review_md(clean_rows))
    with open(os.path.join(out_dir, "bewertung.jsonl"), "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps({"id": r["id"], "weg": r["weg"], "lauf": r["lauf"],
                                "status": r["status"], "kf": None, "evidence_quote": None,
                                "evidence_type": [], "vorschlag": (r["vorpruefung"] or {}).get("vorschlag")},
                               ensure_ascii=False) + "\n")
    meta = {"commit": _commit(), "ollama": args.ollama, "stufe": args.stufe, "wege": wege,
            "laeufe": args.laeufe, "temperatur_override": args.temperatur, "trocken": args.trocken,
            "redaktion": {"regelversion": redact.REGEL_VERSION, "treffer": redaction_hits,
                          "zeit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "faelle": [c.id for c in selected]}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\nErgebnis: {out_dir}\nBewerten: review.md lesen, bewertung.jsonl ausfüllen, dann\n"
          f"  python -m scripts.honesty.summary {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
