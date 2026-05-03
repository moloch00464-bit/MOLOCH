#!/usr/bin/env python3
"""
M.O.L.O.C.H. Transition Engine — Phase 1 (Drei-Hirn-Synthese, ChatGPT-Position)
================================================================================

Sichert Stabilitaet von State-Wechseln im 6-State-FSM.
ChatGPT-Synthese-Punkt: "no rapid oscillation between states, bounded
transition frequency, failsafe fallback to idle bei Inkonsistenz".

Regeln:
- MIN_DURATION_MS: kein Wechsel innerhalb 500ms (debounce)
- BOUNDED_SPEED: max Delta pro tick (Tension moduliert das, nicht den Ziel-State)
- FAILSAFE_TIMEOUT_S: kein State stuck ueber 300s ohne Transition - sonst zu idle

Singleton: get_transition_engine()
"""

import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger("MolochTransitionEngine")

VALID_STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")

MIN_DURATION_MS = 500
FAILSAFE_TIMEOUT_S = 300.0  # State stuck > 5 Min -> Failsafe idle
BASE_SPEED = 0.3   # Default Transition-Speed wenn Tension neutral
MIN_SPEED = 0.05
MAX_SPEED = 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TransitionEngine:
    """Mechanik fuer State-Wechsel mit Min-Duration, Bounded Speed, Failsafe."""

    def __init__(self):
        self._lock = threading.RLock()
        self._current: str = "idle"
        self._last_transition_ts: float = time.time()
        self._proposed: Optional[str] = None
        self._proposed_since: float = 0.0
        self._transition_speed: float = BASE_SPEED

    def request_transition(self, target: str, tension: float = 0.0) -> Tuple[bool, str]:
        """Schlage State-Wechsel vor. Gibt (applied, reason) zurueck.

        Tension wirkt nur auf transition_speed (Meta-Parameter), NICHT auf den
        Ziel-State. Hohe Tension = schnellerer Wechsel, niedrige = traeger.
        """
        if target not in VALID_STATES:
            return False, f"invalid target '{target}'"

        with self._lock:
            now = time.time()
            self._update_speed(tension)

            if target == self._current:
                self._proposed = None
                self._proposed_since = 0.0
                return False, "already in target"

            since_last_ms = (now - self._last_transition_ts) * 1000
            if since_last_ms < MIN_DURATION_MS:
                return False, f"min duration {MIN_DURATION_MS}ms not reached ({since_last_ms:.0f}ms)"

            self._current = target
            self._last_transition_ts = now
            self._proposed = None
            self._proposed_since = 0.0
            logger.info(f"transition -> {target} (speed={self._transition_speed:.2f}, "
                        f"tension={tension:.2f})")
            return True, "ok"

    def failsafe_check(self) -> Optional[str]:
        """Pruefe ob current_state stuck ueber FAILSAFE_TIMEOUT_S.

        Returnt new_state wenn Failsafe gefeuert, sonst None.
        """
        with self._lock:
            stuck_s = time.time() - self._last_transition_ts
            if stuck_s > FAILSAFE_TIMEOUT_S and self._current != "idle":
                old = self._current
                self._current = "idle"
                self._last_transition_ts = time.time()
                logger.warning(f"FAILSAFE: state '{old}' stuck {stuck_s:.0f}s -> idle")
                return "idle"
            return None

    def _update_speed(self, tension: float) -> None:
        try:
            t = float(tension)
        except (TypeError, ValueError):
            t = 0.0
        # Tension [-1..+1] -> Speed Modulation: hoeher Tension = schneller
        modulator = 0.5 + 0.5 * _clamp(t, -1.0, 1.0)  # -1 -> 0.0, +1 -> 1.0
        self._transition_speed = _clamp(BASE_SPEED + modulator * 0.5, MIN_SPEED, MAX_SPEED)

    def current(self) -> str:
        with self._lock:
            return self._current

    def transition_speed(self) -> float:
        with self._lock:
            return self._transition_speed

    def last_transition_ts(self) -> float:
        with self._lock:
            return self._last_transition_ts

    def state_age_s(self) -> float:
        with self._lock:
            return time.time() - self._last_transition_ts

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "current_state": self._current,
                "transition_speed": self._transition_speed,
                "last_transition_ts": self._last_transition_ts,
                "state_age_s": time.time() - self._last_transition_ts,
            }


_instance: Optional[TransitionEngine] = None
_singleton_lock = threading.Lock()


def get_transition_engine() -> TransitionEngine:
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = TransitionEngine()
        return _instance


if __name__ == "__main__":
    te = get_transition_engine()
    print("init:", te.snapshot())
    ok, reason = te.request_transition("observing", tension=0.0)
    print(f"  -> observing: ok={ok} reason={reason}")
    print("after:", te.snapshot())
    # Sofortiger zweiter Versuch (sollte an MIN_DURATION scheitern)
    ok2, reason2 = te.request_transition("engaged", tension=0.5)
    print(f"  -> engaged immediately: ok={ok2} reason={reason2}")
    time.sleep(0.6)
    ok3, reason3 = te.request_transition("engaged", tension=0.5)
    print(f"  -> engaged after 600ms: ok={ok3} reason={reason3}")
    print("final:", te.snapshot())
