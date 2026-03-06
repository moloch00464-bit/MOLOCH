#!/usr/bin/env python3
"""
M.O.L.O.C.H. Agent-Struktur — Gate 1
======================================

Basis-Klasse und 5 spezialisierte Agenten, alle auf dem Event Bus.
Aktuell nur Struktur — Logik kommt in spaeteren Gates.

Agenten:
  BuilderAgent   — Baut/deployt Code-Aenderungen
  TesterAgent    — Fuehrt Tests aus, validiert Aenderungen
  DebuggerAgent  — Analysiert Fehler, sammelt Diagnostik
  ReviewerAgent  — Code-Review, Qualitaetssicherung
  ChaosAgent     — Stresstest, Fehlerinjektion

Alle subscriben auf den zentralen MolochEventBus.

Author: M.O.L.O.C.H. System (Gate 1)
"""

import logging
import threading
from typing import Optional

from core.moloch_event_bus import (
    get_event_bus,
    MolochEventBus,
    PRIO_SYSTEM,
    PRIO_INFO,
    PRIO_DEBUG,
)

logger = logging.getLogger("MolochAgents")


# ============================================================
# BASIS-KLASSE — Gemeinsame Agent-Infrastruktur
# ============================================================

class BaseAgent:
    """
    Basis fuer alle M.O.L.O.C.H. Agenten.

    Jeder Agent hat:
    - Eigenen Namen und Zustand (idle/active/error)
    - Zugriff auf den Event Bus (subscribe/publish)
    - start()/stop() Lifecycle

    Subklassen ueberschreiben _on_event() fuer ihre Logik.
    """

    def __init__(self, name: str):
        self.name = name
        self._bus: MolochEventBus = get_event_bus()
        self._state = "idle"  # idle, active, error
        self._running = False
        self._subscriptions: list = []  # (topic, callback) fuer Cleanup
        logger.info(f"[AGENT:{self.name}] Erstellt")

    def subscribe(self, topic: str, priority: int = PRIO_INFO):
        """Auf ein Event-Topic subscriben."""
        callback = self._on_event
        self._bus.subscribe(topic, callback, priority=priority)
        self._subscriptions.append((topic, callback))

    def publish(self, event_type: str, payload: dict = None, priority: int = PRIO_INFO):
        """Event auf den Bus publishen."""
        self._bus.publish(
            event_type=event_type,
            payload=payload or {},
            source=f"agent.{self.name}",
            priority=priority,
        )

    def _on_event(self, event: dict):
        """Event-Handler — von Subklassen ueberschrieben."""
        pass

    def start(self):
        """Agent aktivieren."""
        self._running = True
        self._state = "active"
        logger.info(f"[AGENT:{self.name}] Gestartet")

    def stop(self):
        """Agent deaktivieren und Subscriptions aufraeumen."""
        self._running = False
        self._state = "idle"
        for topic, callback in self._subscriptions:
            self._bus.unsubscribe(topic, callback)
        self._subscriptions.clear()
        logger.info(f"[AGENT:{self.name}] Gestoppt")

    def get_status(self) -> dict:
        """Agent-Status fuer IPC/Panel."""
        return {
            "name": self.name,
            "state": self._state,
            "running": self._running,
            "subscriptions": [t for t, _ in self._subscriptions],
        }


# ============================================================
# SPEZIALISIERTE AGENTEN — Nur Struktur, keine Logik
# ============================================================

class BuilderAgent(BaseAgent):
    """Baut und deployt Code-Aenderungen. Subscribt auf build-Events."""

    def __init__(self):
        super().__init__("builder")
        self.subscribe("agent.build_request", priority=PRIO_SYSTEM)


class TesterAgent(BaseAgent):
    """Fuehrt Tests aus und validiert Aenderungen. Subscribt auf test-Events."""

    def __init__(self):
        super().__init__("tester")
        self.subscribe("agent.test_request", priority=PRIO_SYSTEM)


class DebuggerAgent(BaseAgent):
    """Analysiert Fehler und sammelt Diagnostik. Subscribt auf error-Events."""

    def __init__(self):
        super().__init__("debugger")
        self.subscribe("system.error", priority=PRIO_SYSTEM)


class ReviewerAgent(BaseAgent):
    """Code-Review und Qualitaetssicherung. Subscribt auf review-Events."""

    def __init__(self):
        super().__init__("reviewer")
        self.subscribe("agent.review_request", priority=PRIO_INFO)


class ChaosAgent(BaseAgent):
    """Stresstest und Fehlerinjektion. Subscribt auf chaos-Events."""

    def __init__(self):
        super().__init__("chaos")
        self.subscribe("agent.chaos_request", priority=PRIO_DEBUG)


# ============================================================
# CONVENIENCE — Alle Agenten auf einmal
# ============================================================

_all_agents: Optional[dict] = None


def get_agents() -> dict:
    """Alle Agenten als Dict (Singleton, lazy init)."""
    global _all_agents
    if _all_agents is None:
        _all_agents = {
            "builder": BuilderAgent(),
            "tester": TesterAgent(),
            "debugger": DebuggerAgent(),
            "reviewer": ReviewerAgent(),
            "chaos": ChaosAgent(),
        }
    return _all_agents
