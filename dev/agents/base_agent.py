"""
BaseDevAgent — Basisklasse für alle Entwicklungsagenten.
Nicht zu verwechseln mit core/agents/ (Runtime-Agenten).
Diese hier sind TOOLS für die Entwicklung, keine Runtime-Komponenten.
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path


class TaskStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


@dataclass
class TaskResult:
    agent: str
    task_id: str
    status: TaskStatus
    timestamp: float = field(default_factory=time.time)
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class Task:
    task_id: str
    feature: str
    description: str
    gate: str = "gate_1"
    priority: str = "medium"
    target_files: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseDevAgent:
    """Basisklasse für Entwicklungsagenten."""

    AGENT_NAME = "base"
    LOG_DIR = Path.home() / "moloch" / "logs" / "dev_agents"

    def __init__(self):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def execute(self, task: Task) -> TaskResult:
        """Überschreiben in Subklassen."""
        raise NotImplementedError(f"{self.AGENT_NAME} hat keine execute() Logik")

    def log_result(self, result: TaskResult):
        """Schreibt Ergebnis als JSON in Log-Verzeichnis."""
        logfile = self.LOG_DIR / f"{result.task_id}_{self.AGENT_NAME}.json"
        logfile.write_text(result.to_json(), encoding="utf-8")

    def _make_result(self, task: Task, status: TaskStatus, **kwargs) -> TaskResult:
        """Convenience: erstellt TaskResult mit vorausgefüllten Feldern."""
        return TaskResult(
            agent=self.AGENT_NAME,
            task_id=task.task_id,
            status=status,
            **kwargs,
        )
