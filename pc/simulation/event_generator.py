"""Synthetic Event-Generator fuer Simulation (Phase 2 Plan).

Erzeugt face_detected / voice_input / tension_spike Events ohne Live-Pi-Stream,
damit Scenarios reproduzierbar getestet werden koennen.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SimEvent:
    ts: float
    kind: str  # face_detected | voice_input | tension_spike
    payload: dict = field(default_factory=dict)


class EventGenerator:
    def __init__(self, base_ts: float | None = None):
        self.base_ts = base_ts if base_ts is not None else time.time()
        self._cursor: float = 0.0

    def _next_ts(self, dt: float) -> float:
        self._cursor += dt
        return self.base_ts + self._cursor

    def face_detected(self, person_id: str, conf: float = 0.9, dt: float = 0.0) -> SimEvent:
        return SimEvent(
            ts=self._next_ts(dt),
            kind="face_detected",
            payload={"person_id": person_id, "confidence": conf},
        )

    def voice_input(self, transcript: str, dt: float = 0.0) -> SimEvent:
        return SimEvent(
            ts=self._next_ts(dt),
            kind="voice_input",
            payload={"transcript": transcript},
        )

    def tension_spike(self, delta: float, reason: str = "synthetic", dt: float = 0.0) -> SimEvent:
        return SimEvent(
            ts=self._next_ts(dt),
            kind="tension_spike",
            payload={"delta": delta, "reason": reason},
        )
