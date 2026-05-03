"""DH-6 Transition-Engine (PC-Side).

ChatGPT-Synthese: Tension = Dynamik (Speed-Modulator), NICHT direkter State-Trigger.

Regeln:
- Min state duration 500ms (kein Hektik-Switch)
- Bounded transition speed: max delta pro tick
- Tension hoch -> schnellere Transition (bis zu 2x)
- Tension niedrig -> langsamere (bis 0.5x)
- Failsafe-Fallback to idle wenn input-Vector inkonsistent (sum != ~1.0, NaN, negative)
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")

MIN_STATE_DURATION_MS = 500
BASE_TRANSITION_SPEED = 0.15  # default delta per tick
MIN_SPEED = 0.05
MAX_SPEED = 0.40

VECTOR_SUM_TOLERANCE = 0.05  # akzeptierte Abweichung von 1.0


class TransitionEngine:
    """Smooth state-Vector-Uebergaenge mit Min-Duration + Tension-Modulation."""

    def __init__(self) -> None:
        self._current: Dict[str, float] = {s: 0.0 for s in STATES}
        self._current["idle"] = 1.0
        self._primary: str = "idle"
        self._primary_since_ms: int = int(time.time() * 1000)
        self._last_tick_ts: float = 0.0

    def tick(
        self,
        target_vector: Dict[str, float],
        tension: float = 0.0,
    ) -> Dict[str, float]:
        """Lerpt _current Richtung target_vector mit tension-modulierter Speed.

        target_vector: gewuenschter Ziel-Vector (z.B. aus Pi state_vector.py + EMA)
        tension: 0..1 (oder negativ als Sentinel) - moduliert Transition-Speed

        Returns: aktueller Vector nach Tick.
        """
        now_ts = time.time()
        now_ms = int(now_ts * 1000)

        # Validate target — fallback to idle bei Inkonsistenz
        target = self._validate_or_failsafe(target_vector)

        # Speed = base * tension-multiplier (hohe Tension = schneller)
        speed = self._tension_modulated_speed(tension)

        # Lerp pro State
        for s in STATES:
            cur = self._current.get(s, 0.0)
            tgt = target.get(s, 0.0)
            self._current[s] = cur + (tgt - cur) * speed

        # Re-Normalize
        total = sum(self._current.values()) or 1.0
        self._current = {s: v / total for s, v in self._current.items()}

        # Primary-State + Min-Duration-Lock
        new_primary = max(self._current.items(), key=lambda kv: kv[1])[0]
        if new_primary != self._primary:
            since_ms = now_ms - self._primary_since_ms
            if since_ms >= MIN_STATE_DURATION_MS:
                self._primary = new_primary
                self._primary_since_ms = now_ms
            # else: bleib bei altem primary (lock)

        self._last_tick_ts = now_ts
        return dict(self._current)

    def primary(self) -> str:
        return self._primary

    def vector(self) -> Dict[str, float]:
        return dict(self._current)

    def _tension_modulated_speed(self, tension: float) -> float:
        # Tension Sentinel (-1.0) = idle, langsame transition
        if tension < 0:
            return MIN_SPEED
        # 0.0-0.5: base, 0.5-1.0: bis 2x base
        if tension <= 0.5:
            multiplier = 1.0
        else:
            multiplier = 1.0 + (tension - 0.5) * 2.0  # bis 2.0 bei tension=1.0
        speed = BASE_TRANSITION_SPEED * multiplier
        return max(MIN_SPEED, min(MAX_SPEED, speed))

    def _validate_or_failsafe(self, vec: Dict[str, float]) -> Dict[str, float]:
        """Returnt validierten Vector oder failsafe-idle bei Inkonsistenz."""
        if not isinstance(vec, dict):
            return self._idle_vector()
        try:
            cleaned = {}
            for s in STATES:
                v = vec.get(s, 0.0)
                if not isinstance(v, (int, float)):
                    v = 0.0
                if math.isnan(v) or math.isinf(v) or v < 0:
                    v = 0.0
                cleaned[s] = float(v)
            total = sum(cleaned.values())
            if total <= 0 or abs(total - 1.0) > (VECTOR_SUM_TOLERANCE + 0.5):
                # zu unsicher
                return self._idle_vector()
            # Re-normalize
            return {s: v / total for s, v in cleaned.items()}
        except Exception:
            return self._idle_vector()

    @staticmethod
    def _idle_vector() -> Dict[str, float]:
        return {"idle": 1.0, "observing": 0.0, "engaged": 0.0, "overloaded": 0.0, "withdrawing": 0.0, "offline_anchor": 0.0}
