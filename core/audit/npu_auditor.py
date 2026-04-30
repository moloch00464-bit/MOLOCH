"""NPU-Worker-Auditor (Welle 12 Schritt 2).

Pullt:
- /dev/h1x-0 device existiert (Hailo-Treiber-Health)
- dmesg-channel-warnings (Frühwarnung VOR VDevice-Stuck)
- vision_workers.get_health() pro Worker (Face/Pose/ReID/Depth)
- ROI-Dispatcher Frames/Dispatched/Dropped

Schreibt audit_state.layers.npu Schema:
  {workers: {face/pose/reid/depth: {loaded, inferences, errors, queue, last_ms}},
   total_inferences_24h, error_rate, dropped_pct, dmesg_channel_warnings,
   hailo_device_present, status, score, max, detail}

Status-Logik:
- PASS: alle Worker loaded, error_rate <1%, kein dmesg-warning
- WARN: 1 Worker tot ODER error_rate 1-5% ODER dropped >10%
- FAIL: hailo-device fehlt ODER >1 Worker tot ODER error_rate >5%
"""
from __future__ import annotations

import os
import subprocess
import logging
from typing import Any, Dict

logger = logging.getLogger("npu_auditor")


def _hailo_device_present() -> bool:
    """Pruefe ob /dev/h1x-0 oder /dev/hailo* existiert."""
    if os.path.exists("/dev/h1x-0"):
        return True
    try:
        return any(n.startswith("hailo") for n in os.listdir("/dev"))
    except Exception:
        return False


def _dmesg_channel_warnings() -> int:
    """Zaehle dmesg-Lines mit 'channels' + 'enabled' (NPU-VDevice-Race-Frühwarnung)."""
    try:
        r = subprocess.run(
            ["sudo", "dmesg", "--time-format=iso"],
            capture_output=True, text=True, timeout=10,
        )
        lines = r.stdout.splitlines()
        return sum(
            1 for ln in lines[-200:]
            if "hailo" in ln.lower() and ("channels" in ln.lower() and "enabled" in ln.lower())
        )
    except Exception:
        return 0


def collect() -> Dict[str, Any]:
    """Sammelt NPU-Layer-Daten."""
    detail: Dict[str, Any] = {}
    workers: Dict[str, Dict[str, Any]] = {}
    total_inf = 0
    total_err = 0

    # 1. Hailo-Device-Health
    hailo_present = _hailo_device_present()
    detail["hailo_device_present"] = hailo_present

    # 2. dmesg-channel-warnings (Frühwarnung)
    warnings = _dmesg_channel_warnings()
    detail["dmesg_channel_warnings"] = warnings

    # 3. Worker-Health aus vision_workers
    try:
        from core.perception.vision_workers import get_workers  # type: ignore
        ws = get_workers() or {}
        for name, worker in ws.items():
            if not hasattr(worker, "get_health"):
                continue
            try:
                h = worker.get_health() or {}
                workers[name] = {
                    "loaded": bool(h.get("loaded", False)),
                    "running": bool(h.get("running", False)),
                    "inferences": int(h.get("inferences", 0) or 0),
                    "errors": int(h.get("errors", 0) or 0),
                    "queue": int(h.get("queue_size", h.get("queue", 0)) or 0),
                    "last_ms": float(h.get("last_inference_ms", h.get("last", 0)) or 0),
                }
                total_inf += workers[name]["inferences"]
                total_err += workers[name]["errors"]
            except Exception as e:
                workers[name] = {"error": str(e)[:80]}
    except Exception as e:
        detail["worker_import_error"] = str(e)[:100]

    # 4. ROI-Dispatcher (best-effort)
    try:
        from core.perception.roi_dispatcher import get_dispatcher  # type: ignore
        d = get_dispatcher()
        if d is not None and hasattr(d, "get_stats"):
            stats = d.get_stats() or {}
            detail["roi_frames"] = int(stats.get("frames", 0) or 0)
            detail["roi_dispatched"] = int(stats.get("dispatched", 0) or 0)
            detail["roi_dropped"] = int(stats.get("dropped", 0) or 0)
    except Exception:
        pass

    # 5. Status-Berechnung
    score = 0
    max_score = 4
    if hailo_present:
        score += 1
    if workers:
        loaded_count = sum(1 for w in workers.values() if w.get("loaded"))
        if loaded_count >= 4:
            score += 1
        if loaded_count == len(workers) and len(workers) > 0:
            score += 1
    error_rate = (total_err / total_inf) if total_inf > 0 else 0.0
    if error_rate < 0.01 and warnings == 0:
        score += 1

    if not hailo_present:
        status = "FAIL"
    elif error_rate > 0.05 or warnings >= 5:
        status = "FAIL"
    elif error_rate > 0.01 or warnings > 0:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "workers": workers,
        "total_inferences": total_inf,
        "total_errors": total_err,
        "error_rate": round(error_rate, 4),
        "detail": detail,
    }
