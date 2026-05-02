"""W21 Tools — Mood (Personality-State)."""
from __future__ import annotations
import logging
import json
from typing import Any, Dict

logger = logging.getLogger("agent.tools.mood")


def get_mood() -> Dict[str, Any]:
    out: Dict[str, Any] = {"zone": None, "tension": None, "mode": None}
    try:
        from core.core_integrator import get_core_integrator
        ci = get_core_integrator()
        if hasattr(ci, "get_personality_zone"):
            out["zone"] = ci.get_personality_zone()
        if hasattr(ci, "get_tension"):
            out["tension"] = float(ci.get_tension())
    except Exception as e:
        out["error"] = str(e)[:200]
    try:
        with open("/dev/shm/moloch_status.json") as f:
            st = json.load(f)
        pers = st.get("personality") or {}
        out["mode"] = pers.get("mode")
    except Exception:
        pass
    return out
