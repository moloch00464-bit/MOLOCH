"""LED Closed-Loop-Verifier — set_color -> State-Confirm.

PASS  : Soll-Wert via send_command quittiert (return True)
WARN  : Befehl gesendet aber State-Readback fehlt (kein internes State-Tracking)
FAIL  : kein State-Update / Send-Fehler
SKIP  : LED-Controller nicht verfuegbar
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ._common import fail_result, now, skip_result

logger = logging.getLogger("led_verify")


def _get_controller():
    try:
        from core.hardware.rgb_led_controller import RGBLEDController  # type: ignore
        # Singleton-Faktory existiert oft nicht standardisiert -> einfacher Default
        return RGBLEDController()
    except Exception as e:
        logger.debug("led import failed: %s", e)
        return None


def _read_state(led) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for attr in ("_current_color", "_current_state", "_current_mood", "_last_command"):
        if hasattr(led, attr):
            try:
                state[attr.lstrip("_")] = getattr(led, attr)
            except Exception:
                pass
    if hasattr(led, "get_state") and callable(led.get_state):
        try:
            gs = led.get_state()
            if isinstance(gs, dict):
                state.update({f"gs_{k}": v for k, v in gs.items()})
        except Exception:
            pass
    return state


def verify(timeout_s: int = 5) -> Dict[str, Any]:
    led = _get_controller()
    if led is None:
        return skip_result("led_controller_unavailable")

    t_start = now()
    baseline = _read_state(led)
    prev_state = baseline.get("current_state") or baseline.get("last_command")

    cmd = "set_color('gruen','statisch','mittel')"
    sent_ok = False
    try:
        # set_color() liefert kein bool (nur send_command intern), also wir checken send_command-Path
        led.set_color("gruen", "statisch", "mittel")
        sent_ok = True
    except Exception as e:
        return fail_result("set_color_exception", error=str(e)[:120])

    time.sleep(0.5)
    after = _read_state(led)

    delta_keys = [k for k in after if after.get(k) != baseline.get(k)]

    # Cleanup: vorherigen State restoren falls bekannt
    try:
        if prev_state:
            if hasattr(led, "set_state"):
                led.set_state(str(prev_state))
        else:
            # Default-Reset: idle
            if hasattr(led, "set_state"):
                led.set_state("idle")
    except Exception:
        pass

    if not sent_ok:
        status, score = "FAIL", 0
    elif delta_keys:
        status, score = "PASS", 2
    else:
        # send-Aufruf war OK, aber kein internes State-Tracking sichtbar
        status, score = "WARN", 1

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd,
        "baseline": baseline,
        "after": after,
        "delta": {"changed_keys": delta_keys},
        "duration_s": round(now() - t_start, 2),
        "detail": {
            "send_ok": sent_ok,
            "note": "Hardware-Readback nicht verfuegbar - State-Track via Controller-Attribute",
        },
    }
