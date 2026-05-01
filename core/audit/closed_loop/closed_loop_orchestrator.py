"""Closed-Loop-Orchestrator (Welle 15).

Ruft alle 7 Verifier sequenziell (NICHT parallel — Aktoren konkurrieren).
Schreibt /dev/shm/closed_loop_state.json atomic.

CLI:
  python3 -m core.audit.closed_loop.closed_loop_orchestrator --all
  python3 -m core.audit.closed_loop.closed_loop_orchestrator --ptz
  python3 -m core.audit.closed_loop.closed_loop_orchestrator --led ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List

logger = logging.getLogger("closed_loop_orchestrator")

_STATE_PATH = "/dev/shm/closed_loop_state.json"

# Reihenfolge ist signifikant — Aktoren sequenziell
_VERIFIERS: List[str] = [
    "ptz",
    "led",
    "fan",
    "tts",
    "spotify",
    "memory_recall",
    "bridge_roundtrip",
    "web_search",
]


def _import_verifier(name: str):
    mod_name = f"core.audit.closed_loop.{name}_verify"
    try:
        return __import__(mod_name, fromlist=["verify"])
    except Exception as e:
        logger.error("import %s failed: %s", mod_name, e)
        return None


def _safe_call(name: str) -> Dict[str, Any]:
    mod = _import_verifier(name)
    if mod is None or not hasattr(mod, "verify"):
        return {
            "score": 0, "max": 1, "status": "SKIP",
            "command_sent": "", "baseline": {}, "after": {}, "delta": {},
            "duration_s": 0.0,
            "detail": {"reason": f"verifier_{name}_unavailable"},
        }
    try:
        return mod.verify()
    except Exception as e:
        return {
            "score": 0, "max": 1, "status": "FAIL",
            "command_sent": "", "baseline": {}, "after": {}, "delta": {},
            "duration_s": 0.0,
            "detail": {
                "reason": "verifier_exception",
                "error": str(e)[:200],
                "traceback": traceback.format_exc(limit=3)[:500],
            },
        }


def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(path) or "/dev/shm"
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        # Fallback ohne atomic
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _aggregate(layers: Dict[str, Dict[str, Any]]) -> str:
    """Overall-Status: FAIL > WARN > PASS > SKIP (worst wins)."""
    seen = {l.get("status", "SKIP") for l in layers.values()}
    if "FAIL" in seen:
        return "FAIL"
    if "WARN" in seen:
        return "WARN"
    if "PASS" in seen:
        return "PASS"
    return "SKIP"


def run_all_verifications(only: List[str] | None = None) -> Dict[str, Any]:
    targets = only if only else _VERIFIERS
    layers: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    for name in targets:
        if name not in _VERIFIERS:
            continue
        logger.info("verifier start: %s", name)
        layers[name] = _safe_call(name)
        logger.info("verifier done: %s -> %s", name, layers[name].get("status"))

    overall = _aggregate(layers)
    state = {
        "overall": overall,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "duration_s": round(time.time() - t0, 2),
        "layers": layers,
    }
    _atomic_write(_STATE_PATH, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-Loop Verifier Orchestrator")
    parser.add_argument("--all", action="store_true", help="alle Verifier ausfuehren")
    for name in _VERIFIERS:
        parser.add_argument(f"--{name}", action="store_true", help=f"nur {name}-Verifier")
    parser.add_argument("--quiet", action="store_true", help="kein JSON auf stdout")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    only: List[str] = []
    if not args.all:
        for name in _VERIFIERS:
            if getattr(args, name, False):
                only.append(name)
    if not args.all and not only:
        # Default: alle, wenn nichts spezifiziert
        only = []  # leer -> run_all nimmt _VERIFIERS

    state = run_all_verifications(only or None)

    if not args.quiet:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
