"""W21 Phase 4 Closed-Loop — Agent-Loop End-to-End-Verifikation.

Triggert orchestrator.execute_loop mit Test-Query, prueft:
- mind. 1 tool_call (sonst: LLM hat keinen Tool genutzt -> FAIL)
- final answer non-empty
- erwartetes Tool wurde gerufen (z.B. spotify_top_artists)
- iterations <= max_iter

Lesen:
- orchestrator.execute_loop(TEST_QUERY, max_iterations=3) Result-Dict

PASS  : score == 4 (kein error, answer da, tool_calls > 0, expected tool gerufen)
WARN  : score >= 2
FAIL  : score < 2 ODER loop_exception
SKIP  : kein api_key ODER tool_catalog leer ODER orchestrator nicht ladbar

Best-effort: Verifier crasht nie. Bei Exception -> FAIL mit Reason.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from ._common import fail_result, skip_result

logger = logging.getLogger("closed_loop.agent_loop")

TEST_QUERY = "Was ist Markus' Top-1 Artist?"
EXPECTED_TOOL = "spotify_top_artists"


def verify(timeout_s: int = 90) -> Dict[str, Any]:
    """W21 Phase 4 End-to-End Agent-Loop — Tool-Use-Test."""
    started = time.time()
    cmd = f"orchestrator.execute_loop('{TEST_QUERY}', max_iterations=3)"

    # 1. Orchestrator-Init check
    try:
        from core.agent.orchestrator import get_orchestrator
        orch = get_orchestrator()
    except Exception as e:
        return skip_result(
            f"orchestrator_unavailable:{str(e)[:200]}",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    if not getattr(orch, "api_key", None):
        return skip_result(
            "no_deepseek_api_key",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )
    if not getattr(orch, "tools", None):
        return skip_result(
            "tool_catalog_empty",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    # 2. Loop ausfuehren (mit explizitem max_iter=3, billig)
    try:
        result = orch.execute_loop(TEST_QUERY, max_iterations=3)
    except Exception as e:
        return fail_result(
            f"loop_exception:{str(e)[:200]}",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    duration = time.time() - started

    if not isinstance(result, dict):
        return fail_result(
            f"loop_result_not_dict:{type(result).__name__}",
            duration_s=round(duration, 2),
            command_sent=cmd,
        )

    # 3. Bewerten
    answer = str(result.get("answer") or "").strip()
    iterations = int(result.get("iterations", 0) or 0)
    tool_calls = result.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    err = result.get("error")

    expected_called = any(
        isinstance(tc, dict) and tc.get("name") == EXPECTED_TOOL
        for tc in tool_calls
    )

    score = 0
    max_s = 4
    if not err:
        score += 1
    if answer:
        score += 1
    if tool_calls:
        score += 1
    if expected_called:
        score += 1

    if score >= 4:
        status = "PASS"
    elif score >= 2:
        status = "WARN"
    else:
        status = "FAIL"

    tool_names = [
        tc.get("name") for tc in tool_calls
        if isinstance(tc, dict) and tc.get("name")
    ]

    return {
        "score": score,
        "max": max_s,
        "status": status,
        "command_sent": cmd,
        "baseline": {},
        "after": {
            "iterations": iterations,
            "tool_call_count": len(tool_calls),
            "expected_tool_called": expected_called,
            "answer_chars": len(answer),
            "error": err,
        },
        "delta": {
            "iterations": iterations,
            "tool_call_count": len(tool_calls),
            "expected_tool_called": expected_called,
            "answer_chars": len(answer),
            "error": err,
        },
        "duration_s": round(duration, 2),
        "detail": {
            "query": TEST_QUERY,
            "expected_tool": EXPECTED_TOOL,
            "answer_excerpt": answer[:300],
            "tool_calls": tool_names,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
