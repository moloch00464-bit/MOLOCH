"""CLI fuer Token-Budget-Bericht.

Aufruf:
  python -m pc.agent.token_budget_report          # heute
  python -m pc.agent.token_budget_report --week   # letzte 7 Tage
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from pc.agent.token_budget import (
    DAILY_HARD_CAP,
    PRICING,
    estimate_usd,
    get_state,
    report,
)


def _week_report() -> str:
    state = get_state()
    daily = state.get("daily_buckets", {})
    today = datetime.now(timezone.utc).date()
    lines = ["=== TOKEN-BUDGET — letzte 7 Tage ==="]
    total_tokens = 0
    total_usd = 0.0
    for i in range(7):
        d = today - timedelta(days=i)
        key = d.strftime("%Y-%m-%d")
        b = daily.get(key, {})
        if not b:
            lines.append(f"  {key}  (keine Calls)")
            continue
        in_t = b.get("input", 0)
        out_t = b.get("output", 0)
        usd = b.get("usd", 0.0)
        calls = b.get("calls", 0)
        total_tokens += in_t + out_t
        total_usd += usd
        lines.append(f"  {key}  {in_t+out_t:>8} tokens  ${usd:.4f}  ({calls} calls)")
    lines.append(f"  {'-' * 60}")
    lines.append(f"  Total 7 Tage: {total_tokens:>8} tokens  ${total_usd:.4f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Token-Budget-Bericht")
    parser.add_argument("--week", action="store_true", help="Letzte 7 Tage")
    parser.add_argument("--json", action="store_true", help="JSON-Output")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(get_state(), indent=2, ensure_ascii=False))
        return 0
    if args.week:
        print(_week_report())
    else:
        print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
