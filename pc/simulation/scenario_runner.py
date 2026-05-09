"""Scenario-Runner: orchestriert vordefinierte Event-Sequenzen (Phase 2 Plan).

Scenarios sind JSON-Files unter scenarios/ mit Event-Liste + erwartetem
State-Verlauf. ScenarioRunner laedt + materialisiert sie zu SimEvent-Objekten.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .event_generator import EventGenerator, SimEvent


class ScenarioRunner:
    def __init__(self, scenarios_dir: Path | str):
        self.scenarios_dir = Path(scenarios_dir)
        self._scenario: dict = {}
        self._events: list[SimEvent] = []

    def load_scenario(self, name: str) -> int:
        path = self.scenarios_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Scenario nicht gefunden: {path}")
        with path.open("r", encoding="utf-8") as f:
            self._scenario = json.load(f)

        gen = EventGenerator()
        self._events = []
        for ev in self._scenario.get("events", []):
            kind = ev.get("kind", "")
            dt = float(ev.get("dt", 0.0))
            payload = dict(ev.get("payload", {}))
            if kind == "face_detected":
                self._events.append(
                    gen.face_detected(
                        payload.get("person_id", "unknown"),
                        payload.get("confidence", 0.9),
                        dt,
                    )
                )
            elif kind == "voice_input":
                self._events.append(gen.voice_input(payload.get("transcript", ""), dt))
            elif kind == "tension_spike":
                self._events.append(
                    gen.tension_spike(
                        payload.get("delta", 0.0),
                        payload.get("reason", ""),
                        dt,
                    )
                )
        return len(self._events)

    def __iter__(self) -> Iterator[SimEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def name(self) -> str:
        return str(self._scenario.get("name", ""))

    def expected_state_path(self) -> list[str]:
        return list(self._scenario.get("expected_state_path", []))
