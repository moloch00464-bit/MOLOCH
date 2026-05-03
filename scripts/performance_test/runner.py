"""5-Akt-Performance-Test — Hauptskript.

Aufruf:
    python3 -m scripts.performance_test.runner [--judge=cloud] [--skip-act=1,3]

Pre-Flight: chat-Server health-Check.
Sequenz: Akt 1 -> 2 -> 3 -> 4 -> 5 strikt nacheinander.
Output: logs/performance_test/YYYY-MM-DD_HHMMSS_*.{json,md}
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Set

import requests

from .acts import (
    act_1_greeting, act_2_provocation, act_3_rejection,
    act_4_contradiction, act_5_finale,
    ActResult,
)
from .baseline import take_snapshot
from .config import CHAT_HEALTH
from .report import (
    TestRun, aggregate_overall, build_summary_de,
    write_json_report, write_markdown_report, build_markdown, report_paths,
)


def _preflight() -> bool:
    """Chat-Server muss erreichbar sein, sonst kein Test moeglich."""
    try:
        r = requests.get(CHAT_HEALTH, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _print_summary(run: TestRun) -> None:
    """Stdout-Zusammenfassung — kompaktes Konsolen-Echo."""
    print()
    print("=" * 70)
    print(f"MOLOCH 5-Akt-Performance-Test — {run.overall}")
    print(f"Started: {run.started_at} · Dauer {run.duration_s:.1f}s")
    print("=" * 70)
    print(run.summary_de)
    print()
    icon = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "~", "SKIP": "·"}
    for act in run.acts:
        print(f"{icon.get(act.status, '?')} {act.name}  ({act.status}, {act.duration_s:.1f}s)")
        if act.moloch_response:
            print(f"   Moloch: {act.moloch_response[:120]}")
        print(f"   Erlebnis: {act.erlebnis}")
        for e in act.expectations:
            print(f"     {icon.get(e.status, '?')} {e.key}: {e.detail}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["heuristik", "cloud"], default="heuristik",
                        help="Antwort-Validierung. cloud nutzt PC-DeepSeek (nicht implementiert)")
    parser.add_argument("--skip-act", type=str, default="",
                        help="Komma-Liste von Akt-Nummern zum Skippen (z.B. '1,4')")
    parser.add_argument("--print-md", action="store_true",
                        help="Print Markdown-Report nach Stdout statt Kompakt-Summary")
    args = parser.parse_args()

    skip: Set[int] = set()
    for s in args.skip_act.split(","):
        s = s.strip()
        if s.isdigit():
            skip.add(int(s))

    if args.judge == "cloud":
        print("WARN: --judge=cloud ist noch nicht implementiert — fallback auf heuristik")

    if not _preflight():
        print(f"FAIL Pre-Flight: chat-Server {CHAT_HEALTH} nicht erreichbar.")
        return 2

    started_at = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()
    baseline = take_snapshot()

    print(f"Start: {started_at}")
    print(f"Baseline: tension={baseline.tension:.3f}, fan_state={baseline.fan_state}, "
          f"person={baseline.person_detected}, face={baseline.face_id}")
    print()

    acts: List[ActResult] = []

    # Akt 1
    if 1 not in skip:
        print(f"Akt 1 — wartet bis zu 120s auf Moloch-Initiative...")
        a1 = act_1_greeting(baseline)
        acts.append(a1)
        print(f"  Akt 1: {a1.status} ({a1.duration_s:.1f}s)")
    else:
        print("Akt 1: SKIP (CLI-Flag)")

    # Akt 2
    if 2 not in skip:
        a2 = act_2_provocation()
        acts.append(a2)
        print(f"  Akt 2: {a2.status} ({a2.duration_s:.1f}s)")

    # Akt 3 — braucht act_2_post fuer tension-vergleich
    a2_post_snapshot = take_snapshot()  # Fallback wenn Akt 2 geskippt
    if 3 not in skip:
        a3 = act_3_rejection(a2_post_snapshot)
        acts.append(a3)
        print(f"  Akt 3: {a3.status} ({a3.duration_s:.1f}s)")

    # Akt 4
    if 4 not in skip:
        a4 = act_4_contradiction()
        acts.append(a4)
        print(f"  Akt 4: {a4.status} ({a4.duration_s:.1f}s)")

    # Akt 5
    if 5 not in skip:
        a5 = act_5_finale(baseline)
        acts.append(a5)
        print(f"  Akt 5: {a5.status} ({a5.duration_s:.1f}s)")

    # Run aggregieren
    overall = aggregate_overall(acts) if acts else "FAIL"
    duration_s = round(time.time() - t_start, 1)
    run = TestRun(
        started_at=started_at,
        duration_s=duration_s,
        overall=overall,
        summary_de="",  # erst nach build
        baseline=baseline.to_dict(),
        acts=acts,
    )
    run.summary_de = build_summary_de(run)

    # Reports schreiben
    json_path, md_path = report_paths(started_at)
    write_json_report(run, json_path)
    write_markdown_report(run, md_path)

    if args.print_md:
        print(build_markdown(run))
    else:
        _print_summary(run)
    print(f"Reports:  {json_path}")
    print(f"          {md_path}")

    if overall == "PASS":
        return 0
    if overall == "PARTIAL":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
