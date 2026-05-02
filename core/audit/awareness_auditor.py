"""Awareness-Layer-Auditor (Welle 13).

Misst Situationsbewusstsein:
- ActivityAnalyzer (core.awareness.activity_analyzer)
- moloch_status.json: activity, zone, motion_state

Schreibt audit_state.layers.awareness:
  {activity_state, zone, motion_state, last_publish_age_s,
   score, max, status, detail}

Status-Logik:
- PASS: state in {alone, working, conversation, party, away}, recent
- WARN: state=away laenger als 24h ODER stale >5min
- FAIL: module nicht importierbar

Idle-Toleranz #11: away ist legitimer Idle-State (Markus nicht im Bild).
Erst nach 24h ohne Update wird's WARN.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("awareness_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"

_HEALTHY_ACTIVITIES = {
    "alone",
    "working",
    "conversation",
    "party",
    "watching",
    "reading",
    "away",  # #11: away ist legitimer Idle-State (Markus nicht im Bild)
}

# Schwelle: away laenger als 24h gilt als zu-lange-stale -> WARN
_AWAY_MAX_AGE_S = 24 * 3600.0


def collect() -> Dict[str, Any]:
    """Sammelt Awareness-Layer-Daten."""
    detail: Dict[str, Any] = {}
    activity_state: Any = None
    zone: Any = None
    motion_state: Any = None
    last_publish_age_s = 99999.0
    activity_alive = False

    # 1. ActivityAnalyzer (L0+L1)
    try:
        from core.awareness.activity_analyzer import (  # type: ignore
            get_activity_analyzer,
        )
        an = get_activity_analyzer()
        if an is not None:
            activity_alive = True
            try:
                state = getattr(an, "_state", None)
                if state:
                    activity_state = str(state)
                last_pub = getattr(an, "_last_publish", None)
                if last_pub is not None:
                    try:
                        lp = float(last_pub)
                        if lp > 0:
                            last_publish_age_s = max(0.0, time.time() - lp)
                    except (TypeError, ValueError):
                        pass
                # get_state() fuer Detail
                try:
                    snap = an.get_state() or {}
                    if isinstance(snap, dict):
                        detail["snapshot"] = {
                            k: v
                            for k, v in snap.items()
                            if isinstance(v, (int, float, str, bool))
                            or v is None
                        }
                        if not motion_state:
                            motion_state = snap.get("motion_state")
                        if not zone:
                            zone = snap.get("zone")
                except Exception:
                    pass
            except Exception as ee:
                detail["activity_state_error"] = str(ee)[:100]
    except Exception as e:
        detail["import_error"] = str(e)[:120]

    # 2. moloch_status.json (L2)
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        # activity (string oder dict)
        ac_raw = st.get("activity")
        if isinstance(ac_raw, dict):
            if not activity_state:
                activity_state = ac_raw.get("activity") or ac_raw.get("state")
            if not motion_state:
                motion_state = ac_raw.get("motion_state")
            if not zone:
                zone = ac_raw.get("zone")
        elif isinstance(ac_raw, str) and not activity_state:
            activity_state = ac_raw
        # Top-level fallbacks
        if not zone:
            zone = st.get("zone")
        if not motion_state:
            motion_state = st.get("motion_state") or (
                (st.get("perception", {}) or {}).get("motion_state")
            )
        # RoomMap update_age (best-effort)
        rm = st.get("room_map")
        if isinstance(rm, dict):
            ts = rm.get("last_update_ts") or rm.get("updated_at")
            if ts is not None:
                try:
                    rm_age = max(0.0, time.time() - float(ts))
                    detail["room_map_age_s"] = round(rm_age, 1)
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        detail["status_json_error"] = str(e)[:100]

    detail["activity_alive"] = activity_alive
    activity_norm = (
        str(activity_state).lower() if activity_state else None
    )

    # 3. Status berechnen
    score = 0
    max_score = 4
    if activity_alive:
        score += 1
    if activity_norm:
        score += 1
    if activity_norm in _HEALTHY_ACTIVITIES:
        score += 1
    if last_publish_age_s != 99999.0 and last_publish_age_s < 300:
        score += 1

    stale = (
        last_publish_age_s != 99999.0 and last_publish_age_s > 300
    )
    # #11: away nur WARN wenn laenger als 24h (sonst legitimer Idle-State)
    away_too_long = (
        activity_norm == "away"
        and last_publish_age_s != 99999.0
        and last_publish_age_s > _AWAY_MAX_AGE_S
    )

    if not activity_alive:
        status = "FAIL"
    elif activity_norm is None:
        status = "FAIL"
    elif stale or away_too_long:
        status = "WARN"
    elif activity_norm in _HEALTHY_ACTIVITIES:
        status = "PASS"
    else:
        status = "WARN"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "activity_state": activity_norm,
        "zone": zone,
        "motion_state": motion_state,
        "last_publish_age_s": round(last_publish_age_s, 1)
        if last_publish_age_s != 99999.0
        else None,
        "detail": detail,
    }
