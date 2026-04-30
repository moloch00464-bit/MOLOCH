"""Vision-Pipeline-Auditor (Welle 12 Schritt 1).

Pullt:
- /dev/shm/moloch_status.json: fps, frame_age, frozen_restarts, active_models
- core/system_watchdog.get_watchdog().get_status(): pipeline_restarts, frame_freeze-State

Schreibt audit_state.layers.vision Schema:
  {fps_total, fps_per_worker, frame_age_s, pipeline_running, dropped_frames_24h,
   frozen_restarts, active_models, status, score, max, detail}

Status-Logik:
- PASS: FPS ≥10, frame_age <2s, pipeline_running=True
- WARN: FPS 5-10 ODER frame_age 2-8s
- FAIL: FPS <5 ODER frame_age ≥8s ODER pipeline_running=False
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger("vision_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"


def collect() -> Dict[str, Any]:
    """Sammelt Vision-Layer-Daten. Returns audit_state.layers.vision-Dict."""
    detail: Dict[str, Any] = {}
    fps_total = 0.0
    frame_age = 99.0
    pipeline_running = False
    frozen_restarts = 0
    active_models = []

    # 1. moloch_status.json (Live-Pipeline-Daten)
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        fps = st.get("fps") or {}
        if isinstance(fps, dict):
            fps_total = float(fps.get("total", 0) or 0)
            detail["fps_per_worker"] = {
                k: round(float(v), 1)
                for k, v in fps.items()
                if k != "total" and isinstance(v, (int, float))
            }
        # WICHTIG: 0.0 or 99 == 99 (Python-Truthiness!), explizit None-Check
        fa_raw = st.get("frame_age")
        frame_age = float(fa_raw) if fa_raw is not None else 99.0
        frozen_restarts = int(st.get("frozen_restarts", 0) or 0)
        active_models = list(st.get("active_models", []) or [])
    except Exception as e:
        detail["status_json_error"] = str(e)[:100]

    # 2. system_watchdog get_status (best-effort)
    pipeline_restarts = 0
    onvif_errors = 0
    active_pains = []
    try:
        from core.system_watchdog import get_watchdog  # type: ignore
        ws = get_watchdog().get_status() or {}
        pipeline_running = bool(ws.get("pipeline_running", fps_total > 0))
        pipeline_restarts = int(ws.get("pipeline_restarts", 0) or 0)
        onvif_errors = int(ws.get("onvif_consecutive_errors", 0) or 0)
        active_pains = list(ws.get("active_pains", {}) or {})
    except Exception:
        # Fallback: Pipeline-Running aus FPS ableiten
        pipeline_running = fps_total >= 1.0

    detail["pipeline_restarts"] = pipeline_restarts
    detail["onvif_errors"] = onvif_errors
    detail["active_pains"] = active_pains

    # 3. Status berechnen
    score = 0
    max_score = 4
    if fps_total >= 10:
        score += 2
    elif fps_total >= 5:
        score += 1
    if 0 <= frame_age < 2:
        score += 1
    if pipeline_running:
        score += 1

    if fps_total < 5 or frame_age >= 8 or not pipeline_running:
        status = "FAIL"
    elif fps_total < 10 or frame_age >= 2 or pipeline_restarts > 5:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "fps_total": round(fps_total, 1),
        "frame_age_s": round(frame_age, 2),
        "pipeline_running": pipeline_running,
        "frozen_restarts": frozen_restarts,
        "active_models": active_models,
        "detail": detail,
    }
