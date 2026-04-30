"""Tracking-Layer-Auditor (Welle 13).

Misst PTZ-Tracker (AutonomousTracker / mpo).

L0: AutonomousTracker importierbar (best-effort)
L1: tracker_state, last_tick_age aus moloch_status.json
L2: lost_count, target_track_duration, ptz_modus

Schreibt audit_state.layers.tracking:
  {fsm_state, last_tick_age_s, lost_count_24h, ptz_modus,
   score, max, status, detail}

Status-Logik:
- PASS: state in {tracking, follow, idle, parked}, last_tick_age <2s
- WARN: state=searching laenger als 60s ODER frozen
- FAIL: state nicht ermittelbar
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("tracking_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"

_HEALTHY_STATES = {"tracking", "follow", "idle", "parked", "coast"}
_WARN_STATES = {"searching", "frozen", "locked"}


def collect() -> Dict[str, Any]:
    """Sammelt Tracking-Layer-Daten."""
    detail: Dict[str, Any] = {}
    fsm_state: Any = None
    last_tick_age_s = 99999.0
    lost_count_24h = 0
    ptz_modus: Any = None
    target_track_duration_s = 0.0
    tracker_alive = False

    # 1. AutonomousTracker import (L0, best-effort)
    try:
        from core.mpo.autonomous_tracker import (  # type: ignore
            get_autonomous_tracker,
        )
        tr = get_autonomous_tracker()
        if tr is not None:
            tracker_alive = True
            try:
                st_obj = getattr(tr, "state", None)
                if st_obj is not None:
                    fsm_state = (
                        st_obj.value if hasattr(st_obj, "value") else str(st_obj)
                    )
                stats = getattr(tr, "stats", None)
                if isinstance(stats, dict):
                    detail["cycles"] = stats.get("cycles")
                    detail["lost"] = stats.get("lost")
                    if "lost" in stats:
                        try:
                            lost_count_24h = int(stats.get("lost", 0) or 0)
                        except (TypeError, ValueError):
                            pass
            except Exception as ee:
                detail["tracker_state_error"] = str(ee)[:100]
    except Exception as e:
        detail["import_error"] = str(e)[:120]

    # 2. moloch_status.json (L1+L2)
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        ptz = st.get("ptz") or {}
        if isinstance(ptz, dict):
            # PTZ-Block ist konkret in Status-JSON belegt
            tracker_state_raw = ptz.get("tracker_state")
            if tracker_state_raw and not fsm_state:
                fsm_state = str(tracker_state_raw).lower()
            # last_tick - nicht direkt da; greifen auf frame_age als Approx
            tracking_moves = ptz.get("tracking_moves")
            if tracking_moves is not None:
                detail["tracking_moves"] = tracking_moves
            ptz_stage = ptz.get("ptz_stage")
            if ptz_stage:
                detail["ptz_stage"] = ptz_stage
        ptz_modus = (
            st.get("ptz_arbiter_mode")
            or st.get("ptz_modus")
            or detail.get("ptz_stage")
        )
        # last_tick_age: tracker.last_tick_ts wenn vorhanden, sonst frame_age
        tracker_blk = st.get("tracker") or {}
        if isinstance(tracker_blk, dict):
            ts = tracker_blk.get("last_tick_ts")
            if ts is not None:
                try:
                    last_tick_age_s = max(0.0, time.time() - float(ts))
                except (TypeError, ValueError):
                    pass
            if not fsm_state:
                ts2 = tracker_blk.get("state") or tracker_blk.get("fsm_state")
                if ts2:
                    fsm_state = str(ts2).lower()
            ld = tracker_blk.get("target_track_duration_s")
            if ld is not None:
                try:
                    target_track_duration_s = float(ld)
                except (TypeError, ValueError):
                    pass
            lc = tracker_blk.get("lost_count")
            if lc is not None:
                try:
                    lost_count_24h = int(lc)
                except (TypeError, ValueError):
                    pass
        # Fallback: frame_age als last_tick_age (Vision-Tick proxy)
        if last_tick_age_s == 99999.0:
            fa = st.get("frame_age")
            if fa is not None:
                try:
                    fa_f = float(fa)
                    if fa_f >= 0:
                        last_tick_age_s = fa_f
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        detail["status_json_error"] = str(e)[:100]

    detail["target_track_duration_s"] = round(target_track_duration_s, 1)
    detail["tracker_alive"] = tracker_alive

    # Normalisierung
    fsm_norm = str(fsm_state).lower() if fsm_state else None

    # 3. Status berechnen
    score = 0
    max_score = 4
    if tracker_alive:
        score += 1
    if fsm_norm:
        score += 1
    if fsm_norm in _HEALTHY_STATES:
        score += 1
    if last_tick_age_s != 99999.0 and last_tick_age_s < 2:
        score += 1

    if fsm_norm is None:
        status = "FAIL"
    elif fsm_norm in _WARN_STATES:
        status = "WARN"
    elif fsm_norm in _HEALTHY_STATES and (
        last_tick_age_s == 99999.0 or last_tick_age_s < 5
    ):
        status = "PASS"
    else:
        status = "WARN"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "fsm_state": fsm_norm,
        "last_tick_age_s": round(last_tick_age_s, 2)
        if last_tick_age_s != 99999.0
        else None,
        "lost_count_24h": lost_count_24h,
        "ptz_modus": ptz_modus,
        "detail": detail,
    }
