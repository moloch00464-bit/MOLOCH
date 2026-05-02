"""MOLOCH Orchestrator (Welle 21 Phase 2).

DeepSeek-LLM mit function-calling-Loop. User-Query rein -> Tool-Use-Iterations
-> finale Antwort. Tools werden via Pi-Bridge dispatched.

Aufruf:
  python -m pc.agent.orchestrator "wieviel P-Bands aufm WGT 2026?"
  python -m pc.agent.orchestrator --max-iter 3 "spiel was Hartes"

Konfig:
  --mock        Erzwingt MockBridge (Pi-unabhaengig)
  --max-iter N  Max Tool-Use-Iterations (default 5)
  --verbose     Tool-Calls + Token-Counts ausgeben

NEVER 5: requests-timeout. NEVER 7: kein Runtime-State committen.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from pc.agent import deepseek_client
from pc.agent import token_budget
from pc.agent.pi_tool_bridge import MockBridge, ToolBridge, get_bridge

logger = logging.getLogger("orchestrator")

DEFAULT_MAX_ITER = 5
SYSTEM_PROMPT = (
    "Du bist Moloch — anatomisches AI-System auf Markus' Pi+PC. "
    "Du hast Zugriff auf Tools fuer Web-Suche, URL-Fetch, Spotify, "
    "Mood/Zone und Hardware-Aktoren. Nutze Tools wenn noetig — recherchiere "
    "selbst statt zu raten. Antworte deutsch, knapp, direkt. Kein 'natuerlich gerne'. "
    "Wenn ein Tool fehlt, sag das klar."
)


class Orchestrator:
    def __init__(
        self,
        bridge: Optional[ToolBridge] = None,
        max_iter: int = DEFAULT_MAX_ITER,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: bool = False,
    ) -> None:
        self.bridge = bridge or get_bridge()
        self.max_iter = max_iter
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.tools = self.bridge.get_catalog()
        if not self.tools:
            logger.warning("[orch] Tool-Catalog leer — DeepSeek antwortet ohne Tools")

    def run(self, user_query: str) -> Dict[str, Any]:
        """Loop bis finale Antwort. Returns {answer, iterations, tool_calls, total_tokens}."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        tool_calls_log: List[Dict[str, Any]] = []
        total_tokens = 0

        for iteration in range(1, self.max_iter + 1):
            # Welle 21 Phase 5: Per-Loop-Budget-Check
            if total_tokens >= token_budget.PER_LOOP_DEFAULT:
                logger.warning(
                    f"[orch] Per-Loop-Budget exhausted ({total_tokens} >= "
                    f"{token_budget.PER_LOOP_DEFAULT}) — abort"
                )
                return {
                    "answer": "[token-budget exhausted — kuerz die Frage oder reset Limit]",
                    "iterations": iteration - 1,
                    "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens,
                }
            if token_budget.is_over_daily_cap():
                logger.warning("[orch] Daily-Cap exceeded — Cloud-Calls blocked")
                return {
                    "answer": "[daily token-cap erreicht — bitte morgen weiter oder MOLOCH_DAILY_TOKEN_HARD_CAP env hochsetzen]",
                    "iterations": iteration - 1,
                    "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens,
                }
            if self.verbose:
                logger.info(f"[orch] iter {iteration}/{self.max_iter}")
            response = deepseek_client.complete(messages, tools=self.tools)
            usage = deepseek_client.extract_usage(response)
            total_tokens += usage.get("total_tokens", 0)
            assistant_msg = deepseek_client.extract_message(response)
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                # Finale Antwort
                return {
                    "answer": assistant_msg.get("content", ""),
                    "iterations": iteration,
                    "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens,
                }
            for tc in tool_calls:
                tname = tc.get("function", {}).get("name", "?")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    targs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    targs = {}
                if self.verbose:
                    logger.info(f"[orch]   tool_call: {tname}({targs})")
                bridge_result = self.bridge.dispatch(tname, targs)
                tool_calls_log.append(
                    {"name": tname, "params": targs, "result": bridge_result}
                )
                # Tool-Result als string fuer LLM
                tool_content = json.dumps(bridge_result, ensure_ascii=False)[:5000]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": tool_content,
                    }
                )
        return {
            "answer": "[max iterations reached without final answer]",
            "iterations": self.max_iter,
            "tool_calls": tool_calls_log,
            "total_tokens": total_tokens,
        }


def main(argv: List[str]) -> int:
    # Windows-cp1252-Fallback: stdout auf UTF-8 reconfigure damit Emojis
    # (✅, etc.) nicht UnicodeEncodeError werfen
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MOLOCH Orchestrator (Welle 21)")
    parser.add_argument("query", nargs="+", help="User-Query")
    parser.add_argument("--mock", action="store_true",
                        help="Erzwingt MockBridge (kein Pi-Bezug)")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    user_query = " ".join(args.query)
    bridge = MockBridge() if args.mock else get_bridge()
    orch = Orchestrator(bridge=bridge, max_iter=args.max_iter, verbose=args.verbose)
    result = orch.run(user_query)
    print("=" * 60)
    print(f"ANTWORT (nach {result['iterations']} iter, {result['total_tokens']} tokens):")
    print(result["answer"])
    if result["tool_calls"] and args.verbose:
        print("\nTOOL-CALLS:")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}({tc['params']}) -> "
                  f"{'OK' if tc['result'].get('error') is None else 'ERR: ' + str(tc['result'].get('error'))[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
