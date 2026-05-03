"""DH-6 Safety-Layer (PC-Side).

ChatGPT-Synthese: state_safety_layer verhindert instabile Zustandsoszillation.

Regeln:
- No rapid oscillation: max 2 primary-state-changes pro 5s
- Bounded transition frequency
- Failsafe fallback to 'idle' bei Inkonsistenz
- Detect-and-Reject: wenn primary > N times in M sec wechselt -> stuck auf last good
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

OSCILLATION_WINDOW_SEC = 5.0
MAX_CHANGES_IN_WINDOW = 2

# Wenn nach Trigger: einfrieren auf last-good fuer X Sekunden
FREEZE_DURATION_SEC = 10.0


class SafetyLayer:
    def __init__(self) -> None:
        self._primary_changes: Deque[Tuple[float, str]] = deque(maxlen=20)
        self._last_primary: Optional[str] = None
        self._frozen_until: float = 0.0
        self._last_good_vector: Optional[Dict[str, float]] = None

    def filter(
        self,
        candidate_vector: Dict[str, float],
        candidate_primary: str,
    ) -> Tuple[Dict[str, float], str, bool]:
        """Returnt (filtered_vector, filtered_primary, was_filtered).

        was_filtered=True wenn Safety-Layer eingegriffen hat (Freeze aktiv).
        """
        now = time.time()

        # Bei aktivem Freeze: returne last-good
        if now < self._frozen_until and self._last_good_vector is not None:
            return self._last_good_vector, self._last_primary or "idle", True

        # FIX (Code-Review #3): bei Freeze-Ende primary_changes resetten,
        # damit alte Eintraege nicht sofort wieder Trigger werden
        if self._frozen_until > 0 and now >= self._frozen_until:
            self._primary_changes.clear()
            self._frozen_until = 0.0

        # Track primary-changes
        if candidate_primary != self._last_primary:
            self._primary_changes.append((now, candidate_primary))

        # FIX (Code-Review #2): prune alte Eintraege ausserhalb Window
        cutoff_for_prune = now - OSCILLATION_WINDOW_SEC
        while self._primary_changes and self._primary_changes[0][0] < cutoff_for_prune:
            self._primary_changes.popleft()

        # Count changes in last OSCILLATION_WINDOW_SEC
        cutoff = now - OSCILLATION_WINDOW_SEC
        recent = [t for (t, _) in self._primary_changes if t >= cutoff]
        if len(recent) > MAX_CHANGES_IN_WINDOW:
            # Oszillation entdeckt - Freeze
            self._frozen_until = now + FREEZE_DURATION_SEC
            if self._last_good_vector is not None:
                return self._last_good_vector, self._last_primary or "idle", True
            # Fallback wenn kein last-good: idle
            idle = {"idle": 1.0, "observing": 0.0, "engaged": 0.0, "overloaded": 0.0, "withdrawing": 0.0, "offline_anchor": 0.0}
            return idle, "idle", True

        # Akzeptieren als last-good
        self._last_good_vector = dict(candidate_vector)
        self._last_primary = candidate_primary
        return candidate_vector, candidate_primary, False

    def is_frozen(self) -> bool:
        return time.time() < self._frozen_until

    def stats(self) -> Dict[str, object]:
        return {
            "frozen_until": self._frozen_until,
            "is_frozen": self.is_frozen(),
            "last_primary": self._last_primary,
            "primary_changes_recent": len([
                t for (t, _) in self._primary_changes
                if t >= time.time() - OSCILLATION_WINDOW_SEC
            ]),
        }
