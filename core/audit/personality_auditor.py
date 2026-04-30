"""Personality-Layer-Auditor (Welle 13).

Misst Stabilitaet und Drift der PersonalityEngine.

L0: PersonalityEngine importierbar
L1: mode in {guardian, shadow, berserker}, last_switch_age aus Engine-State
L2: Drift gegen Baseline (config/perception_weights.json optional)

Schreibt audit_state.layers.personality:
  {mode, tension, zone, last_switch_age_s, drift, score, max, status, detail}

Status-Logik:
- PASS: mode set, tension <0.7, zone konsistent
- WARN: tension 0.7-0.9 ODER mode-switch-rate hoch
- FAIL: PersonalityEngine nicht importierbar ODER mode=None
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger("personality_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"
_WEIGHTS_PATH = os.path.expanduser("~/moloch/config/perception_weights.json")

_VALID_MODES = {"guardian", "shadow", "berserker"}


def collect() -> Dict[str, Any]:
    """Sammelt Personality-Layer-Daten."""
    detail: Dict[str, Any] = {}
    mode: Any = None
    tension = 0.0
    zone: Any = None
    last_switch_age_s = 99999.0
    drift = 0.0
    engine_alive = False

    # 1. moloch_status.json (Live-Werte fuer mode/tension/zone)
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        # mode kann "personality_mode" oder "mode" oder "personality.mode" sein
        mode = (
            st.get("personality_mode")
            or (st.get("personality", {}) or {}).get("mode")
        )
        # tension kann float oder dict sein
        t_raw = st.get("tension")
        if isinstance(t_raw, dict):
            tension = float(t_raw.get("level", 0.0) or 0.0)
        elif t_raw is not None:
            try:
                tension = float(t_raw)
            except (TypeError, ValueError):
                tension = 0.0
        # zone (best-effort, kann fehlen)
        zone = (
            st.get("zone")
            or (st.get("personality", {}) or {}).get("zone")
            or (st.get("activity", {}) or {}).get("zone")
            if isinstance(st.get("activity"), dict) else st.get("zone")
        )
    except Exception as e:
        detail["status_json_error"] = str(e)[:100]

    # 2. PersonalityEngine importieren (L0)
    try:
        from core.personality.personality_engine import (  # type: ignore
            get_personality_engine,
        )
        eng = get_personality_engine()
        engine_alive = True
        # Engine-Mode bevorzugt verwenden falls Status-JSON leer
        try:
            eng_mode = getattr(eng, "mode", None)
            if eng_mode is not None and not mode:
                mode = (
                    eng_mode.value
                    if hasattr(eng_mode, "value")
                    else str(eng_mode)
                )
            # last_switch -> Alter
            last_switch = getattr(eng, "last_switch", None)
            if last_switch is not None:
                last_switch_age_s = max(0.0, time.time() - float(last_switch))
        except Exception as ee:
            detail["engine_state_error"] = str(ee)[:100]
    except Exception as e:
        detail["import_error"] = str(e)[:120]

    # 3. Drift aus perception_weights.json (best-effort)
    try:
        if os.path.exists(_WEIGHTS_PATH):
            with open(_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                w = json.load(f)
            d_raw = w.get("drift") if isinstance(w, dict) else None
            if d_raw is not None:
                try:
                    drift = float(d_raw)
                except (TypeError, ValueError):
                    drift = 0.0
    except Exception as e:
        detail["drift_error"] = str(e)[:80]

    # Normalize mode
    mode_str = str(mode).lower() if mode is not None else None
    mode_valid = mode_str in _VALID_MODES if mode_str else False

    # 4. Status berechnen
    score = 0
    max_score = 4
    if engine_alive:
        score += 1
    if mode_valid:
        score += 1
    if tension < 0.7:
        score += 1
    if last_switch_age_s >= 30 or last_switch_age_s == 99999.0:
        # Kein Hyper-Switch (>=30s seit letztem Wechsel oder unbekannt)
        score += 1

    # Switch-Rate-Heuristik: <30s = WARN
    switch_rate_high = (
        last_switch_age_s != 99999.0 and last_switch_age_s < 30
    )

    if not engine_alive or mode_str is None:
        status = "FAIL"
    elif tension >= 0.9 or not mode_valid:
        status = "FAIL"
    elif tension >= 0.7 or switch_rate_high:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "mode": mode_str,
        "tension": round(tension, 3),
        "zone": zone,
        "last_switch_age_s": round(last_switch_age_s, 1)
        if last_switch_age_s != 99999.0
        else None,
        "drift": round(drift, 3),
        "detail": detail,
    }
