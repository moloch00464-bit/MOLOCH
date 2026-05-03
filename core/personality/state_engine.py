#!/usr/bin/env python3
"""
M.O.L.O.C.H. State Engine — Phase 1 Pi-Side (Drei-Hirn-Synthese)
==================================================================

6-State FSM-Layer auf state_vector.py.
Trennt Vector (gewichtete Aktivierung) von FSM (current_state + Transition).

Synthese-Punkte:
- 6 States: idle, observing, engaged, overloaded, withdrawing, offline_anchor
- Tension = Meta-Parameter (NUR Transition-Speed, NICHT Ziel-State) — ChatGPT
- Pi=Reflector, PC-Authority kann via apply_pc_authority() ueberschreiben — Synthese
- Wesen stirbt nie: offline_anchor State + Identity-Phrase — DeepSeek

Ziel-State-Bestimmung: argmax aus state_vector.snapshot().vector.
Transition wird nur ausgefuehrt, wenn TransitionEngine OK gibt
(Min-Duration + Failsafe-Check).

Bei jedem Wechsel: state_logger.log_transition() schreibt JSONL-Entry.

Singleton: get_state_engine()
"""

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger("MolochStateEngine")

VALID_STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")


class StateEngine:
    """6-State FSM mit Vector-Reflector + Transition-Mechanik."""

    def __init__(self):
        self._lock = threading.RLock()
        self._last_tick_ts: float = 0.0

    def tick(self, target_override: Optional[str] = None,
             reason: str = "tick") -> Dict[str, object]:
        """Ein FSM-Tick. Liest Vector, ermittelt Ziel-State, propagiert Transition.

        Args:
            target_override: explizit erzwungener Ziel-State (z.B. PC-Authority).
            reason: Begruendung fuer den Logger.

        Returns:
            snapshot mit current_state + state_vector + transition_speed + ...
        """
        from core.awareness.state_vector import get_state_vector
        from core.personality.transition_engine import get_transition_engine
        from core.personality.state_logger import get_state_logger

        sv = get_state_vector()
        te = get_transition_engine()
        sl = get_state_logger()

        sv.tick()  # Heuristik aktualisiert vector
        snap = sv.snapshot()
        vector: Dict[str, float] = snap["vector"]
        tension = sv.tension_meta()

        # Ziel-State: argmax aus Vector, oder Override
        if target_override and target_override in VALID_STATES:
            target = target_override
        else:
            target = max(vector.items(), key=lambda kv: kv[1])[0]

        # Failsafe-Check: stuck > 300s -> idle
        failsafe = te.failsafe_check()
        if failsafe:
            sl.log_transition(snap["primary"], failsafe, vector=vector,
                              tension=tension, reason="failsafe_stuck")

        prev = te.current()
        applied, why = te.request_transition(target, tension=tension)
        if applied:
            sl.log_transition(prev, target, vector=vector,
                              tension=tension, reason=reason)

        with self._lock:
            self._last_tick_ts = time.time()

        return self.snapshot()

    def apply_pc_authority(self, vector: Dict[str, float],
                           current_state: Optional[str] = None) -> Dict[str, object]:
        """PC-State-Authority laesst Pi-Side den ihrem Vector folgen.

        Setzt state_vector + erzwingt FSM-Transition zum dominanten State.
        """
        from core.awareness.state_vector import get_state_vector
        sv = get_state_vector()
        sv.apply_pc_authority(vector)
        return self.tick(target_override=current_state, reason="pc_authority")

    def snapshot(self) -> Dict[str, object]:
        """PC-Opus-verbindliches Schema fuer /api/state/current.

        Keys: current_state, state_vector, tension, transition_speed,
              last_transition_ts, zone, identity_phrase
        """
        from core.awareness.state_vector import get_state_vector
        from core.personality.transition_engine import get_transition_engine
        from core.personality.identity_phrases import get_phrase

        sv = get_state_vector()
        te = get_transition_engine()
        snap_v = sv.snapshot()
        snap_t = te.snapshot()
        current = snap_t["current_state"]
        vector = snap_v["vector"]

        zone = self._read_zone()

        return {
            "current_state": current,
            "state_vector": vector,
            "tension": float(snap_v["tension_meta"]),
            "transition_speed": float(snap_t["transition_speed"]),
            "last_transition_ts": float(snap_t["last_transition_ts"]),
            "zone": zone,
            "identity_phrase": get_phrase(current),
        }

    def _read_zone(self) -> str:
        """Aktuelle Zone aus moloch_status.json (guardian/shadow/berserker)."""
        try:
            import json
            from pathlib import Path
            p = Path("/dev/shm/moloch_status.json")
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                z = (data.get("zone") or "").lower()
                if z in ("guardian", "shadow", "berserker"):
                    return z
        except Exception:
            pass
        return "guardian"


_instance: Optional[StateEngine] = None
_singleton_lock = threading.Lock()


def get_state_engine() -> StateEngine:
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = StateEngine()
        return _instance


if __name__ == "__main__":
    import pprint
    se = get_state_engine()
    pprint.pprint(se.tick())
    pprint.pprint(se.snapshot())
