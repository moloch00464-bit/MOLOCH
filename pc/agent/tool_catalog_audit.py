"""Tool-Catalog-Audit (PC-Side, Welle 21 Phase 3 Vorbereitung).

Vergleicht Pi-Tool-Catalog (`GET /api/agent/tools`) mit erwarteter Tool-Liste
(W21 Plan: 11 Spotify + 2 Web + Mood/Hardware/Vision).

Pro vorhandenes Tool: Smoketest mit Mini-Param.
Pro fehlendes Tool: MISSING-Marker.

CLI:
  python -m pc.agent.tool_catalog_audit             # report
  python -m pc.agent.tool_catalog_audit --json      # JSON
  python -m pc.agent.tool_catalog_audit --post      # zusaetzlich POST an Pi-mailbox/audit/tool_catalog
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

PI_BASE = "http://192.168.178.30:9100"

# Erwartete Tool-Liste laut W21-Plan (Phase 3 voll)
# Map: tool_name -> Mini-Test-Params (None = SKIP-Smoketest, nur presence-Check)
EXPECTED_TOOLS: Dict[str, Dict[str, Any]] = {
    # === Web (Welle 19+20a) ===
    "web_search":            {"category": "web",      "test_params": {"query": "test", "max_results": 1}},
    "web_fetch":             {"category": "web",      "test_params": None},  # vermeidet echten Page-Fetch im Audit

    # === Spotify (Welle 21 Phase 3 — 11 Tools) ===
    "spotify_play":          {"category": "spotify",  "test_params": None},  # vermeidet ungewollten Play
    "spotify_pause":         {"category": "spotify",  "test_params": None},
    "spotify_next":          {"category": "spotify",  "test_params": None},
    "spotify_prev":          {"category": "spotify",  "test_params": None},
    "spotify_volume":        {"category": "spotify",  "test_params": None},
    "spotify_top_artists":   {"category": "spotify",  "test_params": {"n": 3}},
    "spotify_top_tracks":    {"category": "spotify",  "test_params": {"n": 3}},
    "spotify_search":        {"category": "spotify",  "test_params": {"query": "test"}},
    "spotify_recommend":     {"category": "spotify",  "test_params": None},
    "spotify_now_playing":   {"category": "spotify",  "test_params": {}},
    "spotify_play_genre":    {"category": "spotify",  "test_params": None},

    # === Mood / Personality ===
    "get_mood":              {"category": "mood",     "test_params": {}},

    # === Hardware (Welle 21 Phase 3+) ===
    "ptz_pan":               {"category": "hardware", "test_params": None},
    "led_set":               {"category": "hardware", "test_params": None},
    "camera_snapshot":       {"category": "hardware", "test_params": None},

    # === Browser (Welle 22) ===
    "browser_open":          {"category": "browser",  "test_params": None},
    "browser_click":         {"category": "browser",  "test_params": None},
    "browser_screenshot":    {"category": "browser",  "test_params": None},
}


def _fetch_catalog() -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{PI_BASE}/api/agent/tools", timeout=10)
        r.raise_for_status()
        return r.json().get("tools", [])
    except Exception as e:
        return [{"_error": str(e)[:120]}]


def _dispatch(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(
            f"{PI_BASE}/api/agent/dispatch",
            json={"tool_name": tool_name, "params": params},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
        return {"result": None, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"result": None, "error": str(e)[:120]}


def collect() -> Dict[str, Any]:
    catalog_tools = _fetch_catalog()
    if catalog_tools and "_error" in catalog_tools[0]:
        return {
            "score": 0, "max": len(EXPECTED_TOOLS),
            "status": "FAIL",
            "detail": {"error": catalog_tools[0]["_error"]},
        }
    catalog_names = [t.get("function", {}).get("name", "?") for t in catalog_tools]

    results: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    present_count = 0
    smoketest_pass_count = 0
    smoketest_fail_count = 0

    for tool_name, meta in EXPECTED_TOOLS.items():
        cat = meta["category"]
        cat_stats = by_category.setdefault(cat, {"present": 0, "missing": 0})
        if tool_name not in catalog_names:
            results[tool_name] = {"present": False, "category": cat,
                                   "smoketest": "SKIP (missing)"}
            cat_stats["missing"] += 1
            continue
        present_count += 1
        cat_stats["present"] += 1
        # Smoketest
        if meta["test_params"] is None:
            results[tool_name] = {"present": True, "category": cat,
                                   "smoketest": "SKIP (side-effect-tool)"}
            continue
        disp = _dispatch(tool_name, meta["test_params"])
        if disp.get("error") is None and disp.get("result") is not None:
            results[tool_name] = {"present": True, "category": cat,
                                   "smoketest": "PASS",
                                   "result_type": type(disp.get("result")).__name__}
            smoketest_pass_count += 1
        else:
            results[tool_name] = {"present": True, "category": cat,
                                   "smoketest": "FAIL",
                                   "error": str(disp.get("error", "unknown"))[:120]}
            smoketest_fail_count += 1

    # Score: present_count = score, max = len(EXPECTED_TOOLS)
    score = present_count
    total = len(EXPECTED_TOOLS)
    if smoketest_fail_count > 0:
        status = "FAIL"
    elif present_count >= total * 0.9:
        status = "PASS"
    elif present_count >= total * 0.5:
        status = "WARN"
    else:
        status = "WARN"

    return {
        "score": score,
        "max": total,
        "status": status,
        "detail": {
            "catalog_size": len(catalog_names),
            "expected_size": total,
            "present": present_count,
            "missing": total - present_count,
            "smoketest_pass": smoketest_pass_count,
            "smoketest_fail": smoketest_fail_count,
            "by_category": by_category,
            "tools": results,
        },
    }


def report(result: Dict[str, Any]) -> str:
    d = result["detail"]
    if "error" in d:
        return f"=== TOOL-CATALOG-AUDIT FAIL ===\nFehler: {d['error']}"
    lines = ["=== TOOL-CATALOG-AUDIT ==="]
    lines.append(f"Pi-Catalog: {d['catalog_size']} Tools, erwartet: {d['expected_size']}")
    lines.append(f"Score: {result['score']}/{result['max']} status={result['status']}")
    lines.append(f"  smoketest: {d['smoketest_pass']} PASS / {d['smoketest_fail']} FAIL")
    lines.append("")
    by_cat = d["by_category"]
    lines.append("Per Category:")
    for cat, s in sorted(by_cat.items()):
        lines.append(f"  {cat:10} {s['present']:>2} present / {s['missing']:>2} missing")
    lines.append("")
    lines.append("Per Tool:")
    for name, r in d["tools"].items():
        marker = "V" if r["present"] else "X"
        suffix = ""
        if r.get("smoketest") == "PASS":
            suffix = f" -> {r.get('result_type', '?')}"
        elif r.get("smoketest") == "FAIL":
            suffix = f" -> ERR: {r.get('error', '')[:60]}"
        elif r["present"]:
            suffix = f" ({r['smoketest']})"
        lines.append(f"  {marker} {name:25} [{r['category']:10}]{suffix}")
    return "\n".join(lines)


def post_to_pi(result: Dict[str, Any]) -> bool:
    try:
        payload = {**result, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        r = requests.post(
            f"{PI_BASE}/mailbox/audit/tool_catalog",
            json=payload,
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-Catalog-Audit (W21 Phase 3 Vorbereitung)")
    parser.add_argument("--json", action="store_true", help="JSON-Output")
    parser.add_argument("--post", action="store_true",
                        help="Resultat zu Pi /mailbox/audit/tool_catalog posten")
    args = parser.parse_args()

    result = collect()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(report(result))
    if args.post:
        ok = post_to_pi(result)
        print(f"\n[post] {'OK' if ok else 'FAIL'}", file=sys.stderr)

    return 0 if result["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
