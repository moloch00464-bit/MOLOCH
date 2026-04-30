"""Fan Closed-Loop-Verifier — PWM hoch -> CPU-Temp-Drop.

Test laeuft NUR wenn Baseline-Temp >50 C (sonst SKIP - sinnlos).
PASS  : Temp-Drop >=1.5 C in 30 s
WARN  : 0.5-1.5 C
FAIL  : kein Drop ODER PWM-API nicht verfuegbar
SKIP  : Baseline <50 C ODER ThermalManager nicht verfuegbar
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, Optional

from ._common import fail_result, now, skip_result

logger = logging.getLogger("fan_verify")

_TEST_DURATION = 30.0
_SAMPLE_INTERVAL = 5.0
_BASELINE_MIN_TEMP = 50.0


def _read_cpu_temp() -> Optional[float]:
    """vcgencmd measure_temp -> float (C). Fallback /sys/class/thermal."""
    try:
        r = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            # "temp=58.4'C\n"
            txt = r.stdout.strip().replace("temp=", "").rstrip("'C")
            return float(txt.replace("C", "").strip())
    except Exception as e:
        logger.debug("vcgencmd failed: %s", e)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            milli = int(f.read().strip())
            return milli / 1000.0
    except Exception:
        return None


def _get_thermal():
    try:
        from core.hardware.thermal_manager import get_thermal_manager  # type: ignore
        return get_thermal_manager()
    except Exception as e:
        logger.debug("thermal import failed: %s", e)
        return None


def _read_fan_pwm(tm) -> Optional[int]:
    if tm is None:
        return None
    try:
        # current_fan_level ist standardisiert (0-100 oder Stufen)
        return int(tm.current_fan_level)
    except Exception:
        try:
            st = tm.get_status()
            if isinstance(st, dict):
                return int(st.get("fan_level") or st.get("fan_pwm") or 0)
        except Exception:
            pass
    return None


def _set_fan_pwm(tm, value: int) -> bool:
    """Versucht verschiedene Setter-Namen — je nach Implementation."""
    if tm is None:
        return False
    for name in ("set_fan_pwm", "set_fan_level", "set_target_pwm", "force_fan_level"):
        fn = getattr(tm, name, None)
        if callable(fn):
            try:
                fn(value)
                return True
            except Exception as e:
                logger.debug("%s failed: %s", name, e)
    return False


def verify(timeout_s: int = 60) -> Dict[str, Any]:
    tm = _get_thermal()
    if tm is None:
        return skip_result("thermal_manager_unavailable")

    baseline_temp = _read_cpu_temp()
    if baseline_temp is None:
        return fail_result("temp_unreadable")

    if baseline_temp < _BASELINE_MIN_TEMP:
        return skip_result(
            "baseline_temp_too_low",
            baseline_c=round(baseline_temp, 1),
            min_required=_BASELINE_MIN_TEMP,
        )

    t_start = now()
    baseline_pwm = _read_fan_pwm(tm)

    cmd = "set_fan_pwm(100)"
    if not _set_fan_pwm(tm, 100):
        return fail_result(
            "fan_pwm_api_unavailable",
            baseline_temp=round(baseline_temp, 1),
            baseline_pwm=baseline_pwm,
        )

    samples = [(0.0, baseline_temp)]
    elapsed = 0.0
    while elapsed < _TEST_DURATION:
        time.sleep(_SAMPLE_INTERVAL)
        elapsed = now() - t_start
        t = _read_cpu_temp()
        if t is not None:
            samples.append((round(elapsed, 1), round(t, 2)))

    final_temp = samples[-1][1]
    drop = baseline_temp - final_temp

    # Cleanup: Fan-PWM zuruecksetzen
    if baseline_pwm is not None:
        _set_fan_pwm(tm, baseline_pwm)

    if drop >= 1.5:
        status, score = "PASS", 2
    elif drop >= 0.5:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd,
        "baseline": {"temp_c": round(baseline_temp, 2), "pwm": baseline_pwm},
        "after": {"temp_c": round(final_temp, 2)},
        "delta": {"temp_drop_c": round(drop, 2)},
        "duration_s": round(now() - t_start, 2),
        "detail": {
            "samples": samples,
            "interval_s": _SAMPLE_INTERVAL,
            "duration_s": _TEST_DURATION,
        },
    }
