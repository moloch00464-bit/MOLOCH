"""Smoketest fuer Orchestrator (mit MockBridge — Pi-unabhaengig).

Aufruf: python -m pc.agent.orchestrator_test
"""
from __future__ import annotations

import json
import sys

from pc.agent.orchestrator import Orchestrator
from pc.agent.pi_tool_bridge import MockBridge


CASES = [
    {
        "query": "Welche P-Bands spielen aufm WGT 2026?",
        "expected_tools": ["web_search", "web_fetch"],
        "expected_keywords": ["Portion", "Perturbator", "Phosgore"],
    },
    {
        "query": "Wer sind meine Top-5-Artists auf Spotify?",
        "expected_tools": ["spotify_top_artists"],
        "expected_keywords": ["Suicide Commando"],
    },
    {
        "query": "Hi wie geht's?",
        "expected_tools": [],  # smalltalk braucht keine Tools
        "expected_keywords": [],
    },
]


def main() -> int:
    bridge = MockBridge()
    orch = Orchestrator(bridge=bridge, max_iter=4, verbose=True)

    fail = 0
    for i, case in enumerate(CASES, 1):
        print(f"\n{'='*60}\nCASE {i}: {case['query']!r}\n{'='*60}")
        try:
            result = orch.run(case["query"])
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            fail += 1
            continue

        tool_names = [tc["name"] for tc in result["tool_calls"]]
        answer = result["answer"]

        tool_ok = all(t in tool_names for t in case["expected_tools"]) \
            if case["expected_tools"] \
            else True
        kw_ok = all(k.lower() in answer.lower() for k in case["expected_keywords"]) \
            if case["expected_keywords"] \
            else True

        status = "PASS" if (tool_ok and kw_ok) else "FAIL"
        print(f"  status={status} iter={result['iterations']} "
              f"tokens={result['total_tokens']}")
        print(f"  tools_used={tool_names}")
        print(f"  expected_tools={case['expected_tools']} -> "
              f"{'OK' if tool_ok else 'MISS'}")
        if case["expected_keywords"]:
            print(f"  expected_keywords={case['expected_keywords']} -> "
                  f"{'OK' if kw_ok else 'MISS'}")
        print(f"  answer-preview: {answer[:200]}")
        if status == "FAIL":
            fail += 1

    print(f"\n{'='*60}\nResult: {len(CASES) - fail}/{len(CASES)} PASS")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
