"""Autonomy-Layer-Auditor (Welle 13).

Misst Autonomie-Subsysteme:
- DecisionEngine (core.autonomy.decision_engine)
- Homeostasis (core.autonomy.homeostasis)
- NightCycle (core.autonomy.night_cycle, optional)

Schreibt audit_state.layers.autonomy:
  {decisions_1h, last_decision_age_s, homeostasis_alerts_total,
   night_cycle_state, score, max, status, detail}

Status-Logik:
- PASS: alle 3 Module alive, alerts_total <5 letzte Stunde
- WARN: 1 Modul nicht importierbar
- FAIL: decision_engine nicht alive
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger("autonomy_auditor")


def _parse_ts(ts: Any) -> float:
    """Robust ISO/Float-Timestamp parsen."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.rstrip("Z")).timestamp()
        except Exception:
            try:
                return float(ts)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def collect() -> Dict[str, Any]:
    """Sammelt Autonomy-Layer-Daten."""
    detail: Dict[str, Any] = {}
    decisions_1h = 0
    last_decision_age_s = 99999.0
    homeostasis_alerts_total = 0
    night_cycle_state: Any = None
    decision_alive = False
    homeostasis_alive = False
    night_cycle_alive = False
    modules_alive = 0

    # 1. DecisionEngine (L0+L1)
    try:
        from core.autonomy.decision_engine import (  # type: ignore
            get_decision_engine,
        )
        de = get_decision_engine()
        if de is not None:
            decision_alive = True
            modules_alive += 1
            try:
                last = getattr(de, "_last_decision", None)
                if isinstance(last, dict):
                    ts = _parse_ts(last.get("timestamp") or last.get("ts"))
                    if ts > 0:
                        last_decision_age_s = max(0.0, time.time() - ts)
                # Counter best-effort: prefer get_stats() oder _stats
                stats = getattr(de, "_stats", None)
                if isinstance(stats, dict):
                    detail["decision_stats"] = {
                        k: v
                        for k, v in stats.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                # decisions_1h Approximation: stats.total minus stats.before_1h falls vorhanden
                # Fallback: 1 wenn last_decision <3600s alt
                if last_decision_age_s != 99999.0 and last_decision_age_s < 3600:
                    decisions_1h = max(decisions_1h, 1)
            except Exception as ee:
                detail["decision_state_error"] = str(ee)[:100]
    except Exception as e:
        detail["decision_import_error"] = str(e)[:120]

    # 2. Homeostasis (L1+L2)
    try:
        from core.autonomy.homeostasis import get_homeostasis  # type: ignore
        ho = get_homeostasis()
        if ho is not None:
            homeostasis_alive = True
            modules_alive += 1
            try:
                stats = getattr(ho, "_stats", None)
                if isinstance(stats, dict):
                    homeostasis_alerts_total = int(stats.get("alerts_total", 0) or 0)
                    detail["homeostasis_checks"] = int(stats.get("checks", 0) or 0)
                    detail["homeostasis_heals"] = int(stats.get("heals", 0) or 0)
            except Exception as ee:
                detail["homeostasis_state_error"] = str(ee)[:100]
    except Exception as e:
        detail["homeostasis_import_error"] = str(e)[:120]

    # 3. NightCycle (L2, best-effort)
    try:
        from core.autonomy.night_cycle import get_night_cycle  # type: ignore
        nc = get_night_cycle()
        if nc is not None:
            night_cycle_alive = True
            modules_alive += 1
            try:
                # State auslesen, robust gegen verschiedene Attributnamen
                ncs = getattr(nc, "state", None) or getattr(nc, "_state", None)
                if ncs is not None:
                    night_cycle_state = (
                        ncs.value if hasattr(ncs, "value") else str(ncs)
                    )
            except Exception as ee:
                detail["night_cycle_state_error"] = str(ee)[:100]
    except Exception as e:
        detail["night_cycle_import_error"] = str(e)[:120]

    detail["decision_alive"] = decision_alive
    detail["homeostasis_alive"] = homeostasis_alive
    detail["night_cycle_alive"] = night_cycle_alive
    detail["modules_alive"] = modules_alive

    # 4. Status berechnen
    score = 0
    max_score = 4
    if decision_alive:
        score += 1
    if homeostasis_alive:
        score += 1
    if night_cycle_alive:
        score += 1
    if homeostasis_alerts_total < 5:
        score += 1

    if not decision_alive:
        status = "FAIL"
    elif modules_alive < 3 or homeostasis_alerts_total >= 5:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "decisions_1h": decisions_1h,
        "last_decision_age_s": round(last_decision_age_s, 1)
        if last_decision_age_s != 99999.0
        else None,
        "homeostasis_alerts_total": homeostasis_alerts_total,
        "night_cycle_state": night_cycle_state,
        "detail": detail,
    }
