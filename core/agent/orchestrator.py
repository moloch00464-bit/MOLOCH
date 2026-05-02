"""W21 Agent-Orchestrator — Cloud-LLM mit Tool-Catalog + function-calling.

Phase 1: Loop-Skeleton + Tool-Dispatch + DeepSeek-Roundtrip.
Phase 2: chat_server-Integration (klassifikator-Bypass agent_loop).
"""
from __future__ import annotations
import json
import logging
import os
import requests
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("agent.orchestrator")

MOLOCH_DIR = Path(os.path.expanduser("~/moloch"))
TOOL_CATALOG_PATH = MOLOCH_DIR / "config" / "tool_catalog.json"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_ITERATIONS = 5
MAX_TOKENS_PER_TURN = 4000
SUBPROCESS_TIMEOUT_S = 90


def _load_api_key() -> Optional[str]:
    try:
        with open(MOLOCH_DIR / "config" / "api_keys.json") as f:
            keys = json.load(f)
        return (keys.get("deepseek") or {}).get("api_key")
    except Exception:
        return None


def _load_tool_catalog() -> List[Dict[str, Any]]:
    try:
        with open(TOOL_CATALOG_PATH) as f:
            cat = json.load(f)
        return cat.get("tools", [])
    except Exception as e:
        logger.warning(f"tool_catalog load failed: {e}")
        return []


def _dispatch_tool(name: str, arguments: Dict[str, Any]) -> Any:
    try:
        from core.agent.tools import TOOL_REGISTRY  # type: ignore
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return {"error": f"unknown_tool:{name}"}
        return fn(**arguments)
    except Exception as e:
        return {"error": f"tool_dispatch_error:{str(e)[:200]}"}


class AgentOrchestrator:
    """DeepSeek-API + Tool-Catalog-Loop."""

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt or (
            "Du bist M.O.L.O.C.H. Du hast Tools verfuegbar: web_search, web_fetch, "
            "spotify_top_artists, spotify_play, get_mood. Nutze sie um Markus' Frage zu beantworten. "
            "Multi-Step erlaubt — max 5 Tool-Calls. Antworte am Ende deutsch, kurz, direkt."
        )
        self.tools = _load_tool_catalog()
        self.api_key = _load_api_key()

    def execute_loop(
        self,
        user_query: str,
        max_iterations: int = MAX_ITERATIONS,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "answer": None,
                "error": "no_deepseek_api_key",
                "iterations": 0,
                "tool_calls": [],
            }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        tool_calls_log: List[Dict[str, Any]] = []

        for i in range(max_iterations):
            try:
                resp = requests.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": DEEPSEEK_MODEL,
                        "messages": messages,
                        "tools": self.tools,
                        "max_tokens": MAX_TOKENS_PER_TURN,
                        "temperature": 0.4,
                    },
                    timeout=SUBPROCESS_TIMEOUT_S,
                )
                if not resp.ok:
                    return {
                        "answer": None,
                        "error": f"deepseek_http_{resp.status_code}",
                        "iterations": i,
                        "tool_calls": tool_calls_log,
                    }
                data = resp.json()
                msg = data["choices"][0]["message"]
            except Exception as e:
                return {
                    "answer": None,
                    "error": f"deepseek_error:{str(e)[:200]}",
                    "iterations": i,
                    "tool_calls": tool_calls_log,
                }

            tcalls = msg.get("tool_calls") or []
            if not tcalls:
                return {
                    "answer": msg.get("content") or "",
                    "iterations": i + 1,
                    "tool_calls": tool_calls_log,
                    "error": None,
                }

            messages.append(msg)
            for tc in tcalls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    fn_args = {}
                logger.info(f"[orchestrator] tool_call: {fn_name}({fn_args})")
                result = _dispatch_tool(fn_name, fn_args)
                tool_calls_log.append({
                    "name": fn_name,
                    "args": fn_args,
                    "result_preview": str(result)[:200],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False)[:8000],
                })

        return {
            "answer": None,
            "error": "max_iterations_reached",
            "iterations": max_iterations,
            "tool_calls": tool_calls_log,
        }


_singleton: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _singleton
    if _singleton is None:
        _singleton = AgentOrchestrator()
    return _singleton


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    orch = get_orchestrator()
    result = orch.execute_loop(args.query, max_iterations=args.max_iter)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
