"""Gemeinsame Helfer fuer Closed-Loop-Verifier."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Dict

logger = logging.getLogger("closed_loop")

_STATUS_PATH = "/dev/shm/moloch_status.json"


def now() -> float:
    return time.time()


def is_tracking_active() -> bool:
    """True wenn Moloch gerade auf eine Person tracked (PTZ blockiert)."""
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        # Neuere Pfade: tracker.fsm_state — Fallback: moloch_tracking bool
        tracker = st.get("tracker") or {}
        if isinstance(tracker, dict):
            fsm = str(tracker.get("fsm_state", "")).lower()
            if fsm == "tracking":
                return True
        if bool(st.get("moloch_tracking", False)):
            return True
        ptz = st.get("ptz") or {}
        if isinstance(ptz, dict):
            state = str(ptz.get("state", "")).lower()
            if state in ("tracking", "follow"):
                return True
    except Exception:
        return False
    return False


def write_ipc_cmd(name: str, payload: Dict[str, Any]) -> bool:
    """IPC-Command atomic in /tmp/moloch_cmd_<name>.json schreiben."""
    path = f"/tmp/moloch_cmd_{name}.json"
    payload = dict(payload)
    payload.setdefault("timestamp", now())
    try:
        d = os.path.dirname(path) or "/tmp"
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
    except Exception as e:
        logger.debug("write_ipc_cmd %s failed: %s", name, e)
        return False


def skip_result(reason: str, **extra: Any) -> Dict[str, Any]:
    """Standard-SKIP-Result."""
    base: Dict[str, Any] = {
        "score": 0,
        "max": 1,
        "status": "SKIP",
        "command_sent": "",
        "baseline": {},
        "after": {},
        "delta": {},
        "duration_s": 0.0,
        "detail": {"reason": reason},
    }
    base["detail"].update(extra)
    return base


def fail_result(reason: str, **extra: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "score": 0,
        "max": 1,
        "status": "FAIL",
        "command_sent": "",
        "baseline": {},
        "after": {},
        "delta": {},
        "duration_s": 0.0,
        "detail": {"reason": reason},
    }
    base["detail"].update(extra)
    return base
