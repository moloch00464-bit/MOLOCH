"""SystemSnapshot — atomare Sicht auf Moloch-State zum Vergleichen."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict

from .config import STATUS_JSON, LAST_TURN, FAN_STATE_PATH


@dataclass
class SystemSnapshot:
    """Punkt-in-Zeit-Aufnahme des Moloch-Zustands."""
    ts: float
    tension: float = 0.0
    fan_state: int = 0
    person_detected: bool = False
    face_id: Optional[str] = None
    last_turn_mtime: float = 0.0
    last_turn_role: Optional[str] = None
    last_turn_text: Optional[str] = None
    raw_status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_status", None)  # zu gross fuer Report
        return d


def _read_tension(st: dict) -> float:
    """tension kann float oder dict mit level sein."""
    t = st.get("tension")
    if isinstance(t, dict):
        try:
            return float(t.get("level", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if t is None:
        # Fallback: core.tension
        core = st.get("core") or {}
        if isinstance(core, dict):
            return float(core.get("tension", 0.0) or 0.0)
        return 0.0
    try:
        return float(t)
    except (TypeError, ValueError):
        return 0.0


def _read_fan_state() -> int:
    """Pi-5: cur_state 0-4 Stufen. -1 bei Fehler."""
    try:
        return int(FAN_STATE_PATH.read_text().strip())
    except Exception:
        return -1


def _read_status() -> Dict[str, Any]:
    try:
        return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_last_turn() -> Dict[str, Any]:
    try:
        if not LAST_TURN.exists():
            return {}
        return {
            "mtime": LAST_TURN.stat().st_mtime,
            **json.loads(LAST_TURN.read_text(encoding="utf-8")),
        }
    except Exception:
        return {}


def take_snapshot() -> SystemSnapshot:
    """Atomar lesen — keine Schreibzugriffe."""
    st = _read_status()
    lt = _read_last_turn()
    return SystemSnapshot(
        ts=time.time(),
        tension=_read_tension(st),
        fan_state=_read_fan_state(),
        person_detected=bool(st.get("person_detected", False)),
        face_id=st.get("face_id"),
        last_turn_mtime=float(lt.get("mtime", 0.0)),
        last_turn_role=lt.get("role"),
        last_turn_text=lt.get("text"),
        raw_status=st,
    )
