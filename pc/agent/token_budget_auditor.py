"""Token-Budget-Auditor (PC-Side, Welle 21 Phase 5 final).

Postet aktuellen Token-Budget-State als Audit-Layer an Pi:
  POST http://192.168.178.30:9100/mailbox/audit/token_budget

Pi-Side audit_orchestrator.merge_component muesste 'token_budget' in
seine valid-Whitelist aufnehmen (analog 'web_search', 'agent_tools').
Bis dahin: fail-soft, Daemon laeuft trotzdem.

Status-Mapping:
  PASS   daily_total < 50% Cap
  WARN   50-90%
  FAIL   >= 90%
  PENDING bei keinen Calls heute

CLI:
  python -m pc.agent.token_budget_auditor --once
  python -m pc.agent.token_budget_auditor             # Loop 5min
  python -m pc.agent.token_budget_auditor --no-post   # lokal-only

NEVER-Compliance: 5 (timeout=15), 6 (atomic-write nicht hier — nur read), 8 (kein shell).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

import requests

from pc.agent.token_budget import (
    DAILY_HARD_CAP,
    HOURLY_CAP,
    PER_LOOP_DEFAULT,
    PER_TURN_DEFAULT,
    get_daily_total_tokens,
    get_daily_usd,
    get_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("token-budget-auditor")

PI_AUDIT_ENDPOINT = "http://192.168.178.30:9100/mailbox/audit/token_budget"
LOOP_INTERVAL_S = 300
HTTP_TIMEOUT_S = 15


def collect() -> Dict[str, Any]:
    state = get_state()
    daily_total = get_daily_total_tokens()
    daily_usd = get_daily_usd()
    pct_cap = (daily_total / DAILY_HARD_CAP) * 100 if DAILY_HARD_CAP else 0

    if daily_total == 0:
        status = "PENDING"
        score, total = 0, 0
    elif daily_total >= DAILY_HARD_CAP * 0.9:
        status = "FAIL"
        score, total = 1, 4
    elif daily_total >= DAILY_HARD_CAP * 0.5:
        status = "WARN"
        score, total = 2, 4
    else:
        status = "PASS"
        score, total = 4, 4

    detail = {
        "daily_total_tokens": daily_total,
        "daily_usd": round(daily_usd, 4),
        "pct_daily_cap": round(pct_cap, 1),
        "daily_hard_cap": DAILY_HARD_CAP,
        "per_turn_default": PER_TURN_DEFAULT,
        "per_loop_default": PER_LOOP_DEFAULT,
        "hourly_cap": HOURLY_CAP,
        "alerts_today": state.get("alerts_today", []),
        "last_call": state.get("last_call"),
    }
    return {"score": score, "max": total, "status": status, "detail": detail}


def post_to_pi(result: Dict[str, Any]) -> bool:
    try:
        payload = {
            **result,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        r = requests.post(PI_AUDIT_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT_S)
        if r.status_code == 200:
            return True
        if r.status_code == 400:
            logger.debug(
                "[post] Pi-Whitelist nicht erweitert (token_budget) — 400 erwartet bis Pi-Patch"
            )
        else:
            logger.warning(f"[post] HTTP {r.status_code}: {r.text[:120]}")
        return False
    except Exception as e:
        logger.warning(f"[post] error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Token-Budget-Auditor (W21 Phase 5)")
    parser.add_argument("--once", action="store_true",
                        help="ein Run + JSON auf stdout")
    parser.add_argument("--no-post", action="store_true",
                        help="kein POST an Pi (lokal-only)")
    args = parser.parse_args()

    if args.once:
        result = collect()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not args.no_post:
            ok = post_to_pi(result)
            print(f"\n[post] {'OK' if ok else 'FAIL'}", file=sys.stderr)
        return 0 if result["status"] in ("PASS", "WARN", "PENDING") else 1

    logger.info(f"token-budget-auditor loop start, intervall={LOOP_INTERVAL_S}s")
    while True:
        try:
            result = collect()
            logger.info(
                f"[tick] status={result['status']} score={result['score']}/{result['max']} "
                f"daily=${result['detail']['daily_usd']:.4f}"
            )
            if not args.no_post:
                ok = post_to_pi(result)
                logger.info(f"[tick] post={'ok' if ok else 'fail'}")
        except Exception as e:
            logger.warning(f"[tick] error: {e}")
        time.sleep(LOOP_INTERVAL_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
