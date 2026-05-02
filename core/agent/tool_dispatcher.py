"""W21 Tool-Dispatcher — wrapt TOOL_REGISTRY mit Validation + Timeout + Fail-soft.

Kern-API:
    dispatch(tool_name: str, params: dict) -> dict
        Returns: {"result": Any, "error": Optional[str], "duration_ms": float, "tool": str}

Catalog-Source: config/tool_catalog.json (function-calling-Schema).
Tool-Source: core.agent.tools.TOOL_REGISTRY (callable per Name).
"""
from __future__ import annotations
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.tool_dispatcher")

MOLOCH_DIR = Path(os.path.expanduser("~/moloch"))
TOOL_CATALOG_PATH = MOLOCH_DIR / "config" / "tool_catalog.json"
TOOL_TIMEOUT_S = 30  # NEVER 5: subprocess/network timeout


def _load_catalog() -> Dict[str, Dict[str, Any]]:
    """Returns {tool_name: full_definition_dict}. Fail-soft: leeres dict bei Fehler."""
    try:
        with open(TOOL_CATALOG_PATH, "r", encoding="utf-8") as f:
            cat = json.load(f)
        out: Dict[str, Dict[str, Any]] = {}
        for tool in cat.get("tools", []) or []:
            fn = tool.get("function") or {}
            name = fn.get("name")
            if name:
                out[name] = tool
        return out
    except Exception as e:
        logger.warning(f"tool_catalog load failed: {e}")
        return {}


_CATALOG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_CATALOG_LOCK = threading.Lock()


def get_catalog() -> Dict[str, Dict[str, Any]]:
    """Lazy-loaded singleton."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        with _CATALOG_LOCK:
            if _CATALOG_CACHE is None:
                _CATALOG_CACHE = _load_catalog()
    return _CATALOG_CACHE


def list_tools() -> List[str]:
    return list(get_catalog().keys())


def _validate_params(tool_def: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    """Minimal-Validation gegen input_schema. Returns Fehlerstring oder None."""
    fn = tool_def.get("function") or {}
    schema = fn.get("parameters") or {}
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    for r in required:
        if r not in params:
            return f"missing_required_param:{r}"
    for k, v in (params or {}).items():
        prop_schema = properties.get(k)
        if not isinstance(prop_schema, dict):
            continue  # extra Param tolerieren
        expected = prop_schema.get("type")
        if expected == "string" and not isinstance(v, str):
            return f"type_mismatch:{k}_expected_string"
        if expected == "integer" and not isinstance(v, int):
            return f"type_mismatch:{k}_expected_integer"
        if expected == "number" and not isinstance(v, (int, float)):
            return f"type_mismatch:{k}_expected_number"
        if expected == "boolean" and not isinstance(v, bool):
            return f"type_mismatch:{k}_expected_boolean"
    return None


def _dispatch_with_timeout(fn, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run fn(**params) in worker-thread mit Timeout TOOL_TIMEOUT_S."""
    result_box: Dict[str, Any] = {"result": None, "error": None}

    def _worker():
        try:
            result_box["result"] = fn(**(params or {}))
        except Exception as e:
            result_box["error"] = f"tool_exec:{str(e)[:200]}"

    t = threading.Thread(target=_worker, daemon=True, name="tool-dispatch")
    t.start()
    t.join(timeout=TOOL_TIMEOUT_S)
    if t.is_alive():
        return {"result": None, "error": f"tool_timeout_{TOOL_TIMEOUT_S}s"}
    return result_box


def dispatch(tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Top-Level-Dispatch — liest catalog, validiert, ruft tool, returns gestructte Antwort.

    Returns:
        {
            "tool": str,
            "result": Any (nur wenn error=None),
            "error": Optional[str],
            "duration_ms": float
        }
    """
    started = time.time()
    params = params or {}
    out: Dict[str, Any] = {"tool": tool_name, "result": None, "error": None,
                            "duration_ms": 0.0}
    catalog = get_catalog()
    tool_def = catalog.get(tool_name)
    if tool_def is None:
        out["error"] = f"unknown_tool:{tool_name}"
        out["duration_ms"] = (time.time() - started) * 1000
        return out
    err = _validate_params(tool_def, params)
    if err:
        out["error"] = err
        out["duration_ms"] = (time.time() - started) * 1000
        return out
    try:
        from core.agent.tools import TOOL_REGISTRY  # type: ignore
    except Exception as e:
        out["error"] = f"registry_import:{str(e)[:200]}"
        out["duration_ms"] = (time.time() - started) * 1000
        return out
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        out["error"] = f"registry_missing:{tool_name}"
        out["duration_ms"] = (time.time() - started) * 1000
        return out
    inner = _dispatch_with_timeout(fn, params)
    out["result"] = inner.get("result")
    out["error"] = inner.get("error")
    out["duration_ms"] = (time.time() - started) * 1000
    return out


def _main() -> int:
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True)
    p.add_argument("--params", default="{}", help="JSON-String")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        params = json.loads(args.params)
    except Exception as e:
        print(f"params parse fail: {e}", file=sys.stderr)
        return 1
    res = dispatch(args.tool, params)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if not res.get("error") else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
