"""LED Closed-Loop-Verifier — set_color via IPC -> State-Confirm via /dev/shm.

W18 Cross-Prozess-Fix: liest /dev/shm/moloch_led_state.json (vom Service-Singleton
geschrieben) statt RGBLEDController() im Audit-Subprozess zu instanziieren.

PASS  : color-Diff zeigt Wechsel zu gruen nach IPC-Trigger
WARN  : last_change_ts < 2s alt aber Color != gruen
FAIL  : state-file nicht erreichbar nach Trigger / write_ipc_cmd fehlgeschlagen
SKIP  : state-file fehlt ODER available: false
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from ._common import fail_result, now, skip_result, write_ipc_cmd

logger = logging.getLogger("led_verify")

_STATE_PATH = "/dev/shm/moloch_led_state.json"
_TARGET_COLOR_NAME = "gruen"
_TARGET_RGB = (0, 255, 0)
_SLEEP_AFTER_TRIGGER = 1.0


def _read_state() -> Optional[Dict[str, Any]]:
    """Liest LED-State-File, None wenn fehlt/kaputt."""
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        logger.debug("led state read failed: %s", e)
        return None


def _color_distance(a, b) -> int:
    """Manhattan-Distanz zwischen 2 RGB-Listen, 765 = Maximum."""
    try:
        return sum(abs(int(a[i]) - int(b[i])) for i in range(3))
    except (TypeError, IndexError, ValueError):
        return 765


def verify(timeout_s: int = 5) -> Dict[str, Any]:
    t_start = now()

    baseline = _read_state()
    if baseline is None:
        return skip_result("led_state_file_missing", path=_STATE_PATH)
    if not baseline.get("available", False):
        return skip_result("led_unavailable", available=False)

    base_color = baseline.get("color", [0, 0, 0])
    base_color_name = baseline.get("color_name") or "aus"

    cmd_str = f"led_set(farbe={_TARGET_COLOR_NAME})"
    sent_ok = write_ipc_cmd(
        "led_set",
        {"action": "led_set", "farbe": _TARGET_COLOR_NAME, "modus": "statisch"},
    )
    if not sent_ok:
        return fail_result(
            "ipc_write_failed",
            command_attempted=cmd_str,
            baseline={"color": base_color, "color_name": base_color_name},
        )

    time.sleep(_SLEEP_AFTER_TRIGGER)

    after = _read_state()
    if after is None:
        return fail_result("state_file_missing_after_trigger", command_sent=cmd_str)

    after_color = after.get("color", [0, 0, 0])
    after_color_name = after.get("color_name") or ""
    last_change = float(after.get("last_change_ts", 0.0))
    age_s = max(0.0, time.time() - last_change)

    target_dist = _color_distance(after_color, _TARGET_RGB)
    color_changed = _color_distance(after_color, base_color) > 30
    near_target = target_dist <= 60  # Tolerance fuer Farbnamen-Mapping
    name_match = _TARGET_COLOR_NAME in str(after_color_name).lower()

    # Cleanup: vorigen Farbnamen restoren wenn bekannt
    try:
        if base_color_name and base_color_name != "aus":
            write_ipc_cmd(
                "led_set",
                {"action": "led_set", "farbe": base_color_name, "modus": "statisch"},
            )
    except Exception as e:
        logger.debug("led cleanup failed: %s", e)

    if (near_target or name_match) and color_changed:
        status, score = "PASS", 2
    elif age_s < 2.0 and color_changed:
        status, score = "WARN", 1
    elif age_s < 2.0:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd_str,
        "baseline": {
            "color": base_color,
            "color_name": base_color_name,
            "last_change_ts": baseline.get("last_change_ts", 0.0),
        },
        "after": {
            "color": after_color,
            "color_name": after_color_name,
            "last_change_ts": last_change,
            "age_s": round(age_s, 2),
        },
        "delta": {
            "target_distance": target_dist,
            "color_changed": color_changed,
            "name_match": name_match,
        },
        "duration_s": round(now() - t_start, 2),
        "detail": {
            "target": _TARGET_COLOR_NAME,
            "target_rgb": list(_TARGET_RGB),
            "state_path": _STATE_PATH,
            "note": "W18: state-file-read statt Singleton-Import",
        },
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(verify(), indent=2, ensure_ascii=False))
