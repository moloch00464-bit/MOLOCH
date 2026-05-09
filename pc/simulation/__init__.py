"""MOLOCH PC-Side Simulation-Interface (Phase 2 Plan).

Module:
    replay_engine   - Replay state_log.jsonl Sequenzen (deterministische Tests)
    event_generator - Synthetic events (face_detected / voice_input / tension_spike)
    scenario_runner - Scenario-Files orchestrieren (Provokation, Ruhig, etc.)
"""

from .replay_engine import ReplayEngine
from .event_generator import EventGenerator, SimEvent
from .scenario_runner import ScenarioRunner

__all__ = ["ReplayEngine", "EventGenerator", "SimEvent", "ScenarioRunner"]
