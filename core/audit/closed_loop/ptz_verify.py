"""PTZ Closed-Loop-Verifier — Pan-Befehl -> Position-Diff messen.

PASS  : pan-Diff 15-25 deg
WARN  : pan-Diff <15 deg (PTZ traege)
FAIL  : pan-Diff <5 deg ODER Position nicht lesbar
SKIP  : Tracking aktiv ODER Camera-API nicht verfuegbar
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ._common import fail_result, is_tracking_active, now, skip_result, write_ipc_cmd

logger = logging.getLogger("ptz_verify")

_PAN_OFFSET = 20.0  # Grad
_SLEEP_AFTER_CMD = 2.0


def _read_pan() -> float | None:
    try:
        from core.hardware.camera import get_camera_controller  # type: ignore
        cam = get_camera_controller(auto_connect=False)
        pos = cam.get_position()
        if pos is None:
            return None
        return float(pos.pan)
    except Exception as e:
        logger.debug("read_pan failed: %s", e)
        return None


def _move_pan_relative(delta: float) -> str:
    """Versucht move_absolute via API, faellt zurueck auf IPC-File. Returns command-Beschreibung."""
    # Versuch 1: direkter API-Call
    try:
        from core.hardware.camera import get_camera_controller  # type: ignore
        cam = get_camera_controller(auto_connect=False)
        pos = cam.get_position()
        if pos is not None and getattr(cam, "is_connected", False):
            target_pan = float(pos.pan) + delta
            ok = cam.move_absolute(pan_deg=target_pan, tilt_deg=pos.tilt, speed=0.5)
            if ok:
                return f"move_absolute(pan={target_pan:.1f})"
    except Exception as e:
        logger.debug("move_absolute failed: %s", e)
    # Fallback: IPC
    if write_ipc_cmd("ptz_test", {"action": "pan_relative", "value": delta}):
        return f"ipc_pan_relative({delta:+.1f})"
    return ""


def verify(timeout_s: int = 10) -> Dict[str, Any]:
    if is_tracking_active():
        return skip_result("tracking_active")

    t_start = now()

    baseline_pan = _read_pan()
    if baseline_pan is None:
        return fail_result("position_not_readable")

    cmd = _move_pan_relative(_PAN_OFFSET)
    if not cmd:
        return fail_result("command_send_failed", baseline_pan=baseline_pan)

    time.sleep(_SLEEP_AFTER_CMD)

    after_pan = _read_pan()
    if after_pan is None:
        # Cleanup-Versuch best-effort
        _move_pan_relative(-_PAN_OFFSET)
        return fail_result(
            "position_unreadable_after",
            baseline_pan=baseline_pan,
            command_sent=cmd,
        )

    diff = abs(after_pan - baseline_pan)

    # Cleanup: zuruck-bewegen
    _move_pan_relative(-_PAN_OFFSET)

    if diff >= 15.0 and diff <= 25.0:
        status = "PASS"
        score = 2
    elif diff >= 5.0:
        status = "WARN"
        score = 1
    else:
        status = "FAIL"
        score = 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd,
        "baseline": {"pan": round(baseline_pan, 2)},
        "after": {"pan": round(after_pan, 2)},
        "delta": {"pan": round(after_pan - baseline_pan, 2), "abs": round(diff, 2)},
        "duration_s": round(now() - t_start, 2),
        "detail": {
            "expected_offset_deg": _PAN_OFFSET,
            "tolerance_deg": "15-25",
        },
    }
