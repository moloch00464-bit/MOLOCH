#!/usr/bin/env python3
"""Abnahme Nervensystem Stufe 1 (Issue #5): Temperatur -> Soma -> Koerper -> Rueckmeldung.

Auf dem Pi, Chat-Server und (fuer EXECUTED) PC-Body-Bridge mit Simulator laufen:

    python3 scripts/soma_probe.py                 # nur beobachten
    python3 scripts/soma_probe.py --simuliere 78  # 60 s lang 78 °C vorgeben

Die Vorgabe wirkt NUR auf den Soma-Zustand (quelle=simulation). Watchdog und
Lueftersteuerung sehen weiter die echte Temperatur - es wird nichts gedrosselt.
Erwartet: zustand strained, koerper AUSSTEHEND -> EXECUTED (Bridge + Sim an)
bzw. ACCEPTED (Bridge an, Sim aus) bzw. UNBESTAETIGT nach 20 s (Bridge aus).
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9100")
    ap.add_argument("--simuliere", type=float, default=None, help="CPU-Temperatur in °C vorgeben")
    ap.add_argument("--dauer", type=float, default=60.0)
    args = ap.parse_args()

    if args.simuliere is not None:
        from core.awareness.soma import write_test_override
        write_test_override(args.simuliere, args.dauer)
        print(f"Testvorgabe {args.simuliere} °C fuer {args.dauer:.0f} s gesetzt")

    ende = time.time() + args.dauer + 35
    zuletzt = None
    while time.time() < ende:
        try:
            with urllib.request.urlopen(f"{args.url}/live", timeout=3) as r:
                live = json.load(r)
        except Exception as e:
            print(f"/live nicht erreichbar: {e}")
            return 2
        s, k = live.get("soma") or {}, live.get("koerper") or {}
        zeile = (s.get("zustand"), s.get("quelle"), s.get("kennung"), k.get("stufe"), k.get("ergebnis"))
        if zeile != zuletzt:
            print(f"{time.strftime('%H:%M:%S')}  soma={zeile[0]:<10} quelle={zeile[1]}  "
                  f"kennung={zeile[2]}  koerper={zeile[3]} ({zeile[4]})  {s.get('ursache')}")
            zuletzt = zeile
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
