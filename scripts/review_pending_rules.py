#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Review Pending Rules — CLI

Welle 1 / W1.4 von ThreeBrain FineTune Loop.

Markus-Review-Gate fuer character_patch.json. Geht durch alle pending_rules,
zeigt Trigger + Verhalten + Quell-Events, fragt approve/reject/skip.

Usage:
  python3 scripts/review_pending_rules.py            # interactive review
  python3 scripts/review_pending_rules.py --list     # nur listen, kein prompt
  python3 scripts/review_pending_rules.py --status   # patch + ledger summary

Spaeter (W3.4): --samples flag fuer LoRA-Sample-Review.
"""

import argparse
import os
import sys
from typing import Optional

# Allow running from scripts/ — add moloch root to sys.path
_MOLOCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MOLOCH_ROOT not in sys.path:
    sys.path.insert(0, _MOLOCH_ROOT)


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_rule(rule: dict, idx: int, total: int) -> None:
    print()
    print(f"[{idx}/{total}] {rule.get('id', '?')}  (vorgeschlagen {rule.get('proposed_at', '?')})")
    print(f"  Trigger:   {rule.get('trigger', '?')}")
    print(f"  Verhalten: {rule.get('behavior', '?')}")
    sources = rule.get("source_event_ids") or []
    if sources:
        print(f"  Quelle:    {', '.join(sources[:5])}" + ("  ..." if len(sources) > 5 else ""))
    print(f"  Vorgeschlagen von: {rule.get('proposed_by', '?')}")


def cmd_status() -> int:
    """Patch + Ledger Summary."""
    from core.memory.character_patch import get_patch
    from core.memory.behavior_mutation_ledger import get_ledger

    p = get_patch()
    L = get_ledger()
    pst = p.get_state()
    lst = L.get_state()

    _print_header("MOLOCH Character Patch — STATUS")
    print(f"  Updated:    {pst['updated_at']}")
    print(f"  Active:     {pst['active_count']} Regeln")
    print(f"  Pending:    {pst['pending_count']} Regeln (warten auf Review)")
    print(f"  Rejected:   {pst['rejected_count']} Regeln (verworfen)")
    print(f"  Next ID:    rule_{pst['next_rule_id']:08d}")

    print()
    print(f"  Ledger:     {lst['last_id']} Eintraege total")
    print(f"  Ledger-File: {lst['file']}")

    # Letzte 5 Ledger-Events
    recent = L.read_recent(5)
    if recent:
        print()
        print("  Letzte Ledger-Events:")
        for e in recent:
            ts = e.get("ts", "?")[:19]
            print(f"    [{ts}] {e.get('event', '?')}  {e.get('meta', {})}")

    # Aktive Regeln zeigen
    active = p.get_active_rules()
    if active:
        print()
        print("  AKTIVE REGELN:")
        for r in active:
            print(f"    - {r.get('id')}: Wenn {r.get('trigger', '')[:60]} → {r.get('behavior', '')[:60]}")

    return 0


def cmd_list() -> int:
    """Nur listen ohne Prompt."""
    from core.memory.character_patch import get_patch
    p = get_patch()
    pending = p.get_pending_rules()

    _print_header(f"PENDING RULES ({len(pending)})")
    if not pending:
        print()
        print("  Keine pending Rules. Alles reviewed oder Distiller hat noch nichts vorgeschlagen.")
        return 0

    for i, r in enumerate(pending, 1):
        _print_rule(r, i, len(pending))
    return 0


def _ask(prompt: str) -> str:
    """Input mit Fallback bei EOF (z.B. Pipe-Aufruf)."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "s"


def cmd_review() -> int:
    """Interaktiver Review-Modus."""
    from core.memory.character_patch import get_patch
    p = get_patch()
    pending = p.get_pending_rules()

    _print_header(f"REVIEW PENDING RULES ({len(pending)})")
    if not pending:
        print()
        print("  Keine pending Rules zu reviewen.")
        print()
        return 0

    counts = {"a": 0, "r": 0, "s": 0}
    for i, r in enumerate(pending, 1):
        _print_rule(r, i, len(pending))
        ans = _ask("  [a]pprove / [r]eject / [s]kip / [q]uit > ")
        if ans == "q":
            print("  Abbruch — Rest bleibt pending.")
            break
        if ans == "a":
            ok = p.approve(r["id"], by="markus")
            counts["a"] += 1 if ok else 0
            print(f"  → APPROVED")
        elif ans == "r":
            reason = _ask("  Grund (kurz) > ")
            ok = p.reject(r["id"], reason=reason or "kein Grund", by="markus")
            counts["r"] += 1 if ok else 0
            print(f"  → REJECTED ({reason})")
        else:
            counts["s"] += 1
            print(f"  → SKIP (bleibt pending)")

    _print_header("REVIEW DONE")
    print(f"  Approved: {counts['a']}")
    print(f"  Rejected: {counts['r']}")
    print(f"  Skipped:  {counts['s']}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MOLOCH Character Patch Review CLI")
    parser.add_argument("--status", action="store_true", help="Nur Status anzeigen")
    parser.add_argument("--list", action="store_true", help="Pending listen ohne Prompt")
    parser.add_argument("--samples", action="store_true",
                        help="(Phase W3.4) Sample-Review — noch nicht implementiert")
    args = parser.parse_args()

    if args.samples:
        print("Sample-Review kommt in Welle 3 (W3.4). Bitte Patch-Review nutzen.", file=sys.stderr)
        return 2
    if args.status:
        return cmd_status()
    if args.list:
        return cmd_list()
    return cmd_review()


if __name__ == "__main__":
    sys.exit(main())
