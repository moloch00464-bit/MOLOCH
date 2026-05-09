"""Replay-Engine fuer state_log.jsonl-Sequenzen (Phase 2 Plan).

Liest historische State-Logs (vom Pi-side state_logger erzeugt) und replay'd die
Sequenz, damit Tests deterministisch sind. Speed-Factor erlaubt Beschleunigung.
Pfad zum log wird vom Caller geliefert (Default-Pfade sind Pi-spezifisch).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional


class ReplayEngine:
    def __init__(self, log_path: Path | str, speed_factor: float = 1.0):
        self.log_path = Path(log_path)
        self.speed_factor = speed_factor
        self._entries: list[dict] = []
        self._idx: int = 0

    def load(self) -> int:
        self._entries.clear()
        if not self.log_path.exists():
            raise FileNotFoundError(f"state_log nicht gefunden: {self.log_path}")
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._entries.sort(key=lambda e: e.get("ts", 0.0))
        self._idx = 0
        return len(self._entries)

    def reset(self) -> None:
        self._idx = 0

    def __iter__(self) -> Iterator[dict]:
        return iter(self._entries)

    def next(self) -> Optional[dict]:
        if self._idx >= len(self._entries):
            return None
        entry = self._entries[self._idx]
        self._idx += 1
        return entry

    def __len__(self) -> int:
        return len(self._entries)
