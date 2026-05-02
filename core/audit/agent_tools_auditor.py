"""W21 B4 Agent-Tools-Auditor — Smoketest aller registrierten Tools.

Pullt Tool-Liste aus core.agent.tool_dispatcher und ruft jedes Tool mit
einem Mini-Param. Erwartet result != None UND error == None.

Schreibt audit_state.layers.agent_tools Schema:
  {tool_count, tools_pass, tools_fail, tools_total, per_tool: {name: {status, duration_ms, error}}}

Status-Logik:
- PASS: alle Tools haben result + kein error
- WARN: 1 Tool fail (result=None oder error!=None)
- FAIL: 2+ Tools fail
- PENDING: tool_dispatcher nicht importierbar
"""
from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger("audit.agent_tools")

# Mini-Smoketest-Params pro Tool — leichtgewichtig, kein Side-Effect
_SMOKE_PARAMS = {
    "web_search": {"query": "test", "max_results": 1},
    "web_fetch": {"url": "https://example.com", "max_chars": 500},
    "spotify_top_artists": {"n": 3},
    "spotify_play": None,  # SKIP — Side-Effect (wuerde Playback triggern)
    "get_mood": {},
}


def _smoke_one(tool: str) -> Dict[str, Any]:
    """Ein Tool mit Smoke-Param testen. Returns per-tool-Dict."""
    params = _SMOKE_PARAMS.get(tool)
    out: Dict[str, Any] = {"status": "SKIP", "error": None, "duration_ms": 0.0}
    if params is None:
        out["error"] = "skip_side_effect"
        return out
    try:
        from core.agent.tool_dispatcher import dispatch
        res = dispatch(tool, params)
        out["duration_ms"] = round(res.get("duration_ms", 0.0), 1)
        if res.get("error"):
            out["status"] = "FAIL"
            out["error"] = res["error"]
        elif res.get("result") is None:
            out["status"] = "FAIL"
            out["error"] = "result_none"
        else:
            out["status"] = "PASS"
    except Exception as e:
        out["status"] = "FAIL"
        out["error"] = f"dispatch_exception:{str(e)[:200]}"
    return out


def collect() -> Dict[str, Any]:
    """Sammelt agent_tools-Layer-Daten. Returns audit_state.layers.agent_tools-Dict."""
    detail: Dict[str, Any] = {}
    try:
        from core.agent.tool_dispatcher import list_tools
        tools = list_tools()
    except Exception as e:
        return {"score": 0, "max": 0, "status": "PENDING",
                "tool_count": 0, "tools_pass": 0, "tools_fail": 0,
                "detail": {"error": f"dispatcher_unavailable:{str(e)[:200]}"}}

    per_tool: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        per_tool[t] = _smoke_one(t)

    pass_count = sum(1 for r in per_tool.values() if r["status"] == "PASS")
    fail_count = sum(1 for r in per_tool.values() if r["status"] == "FAIL")
    skip_count = sum(1 for r in per_tool.values() if r["status"] == "SKIP")
    total = len(per_tool)
    testable = total - skip_count

    if total == 0:
        status = "PENDING"
    elif fail_count == 0:
        status = "PASS"
    elif fail_count == 1:
        status = "WARN"
    else:
        status = "FAIL"

    detail["per_tool"] = per_tool
    detail["skip_count"] = skip_count

    return {
        "score": pass_count,
        "max": testable,
        "status": status,
        "tool_count": total,
        "tools_pass": pass_count,
        "tools_fail": fail_count,
        "detail": detail,
    }


if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(collect(), indent=2, ensure_ascii=False))
    sys.exit(0)
