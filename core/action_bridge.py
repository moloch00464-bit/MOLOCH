#!/usr/bin/env python3
"""
M.O.L.O.C.H. Action Bridge FSM — Gate 1, T01
================================================

Zentrale Entscheidungsschicht zwischen Perception und Execution.
Jede Transition durchlaeuft: Thought -> Intent -> Action -> Result.

States:
  IDLE         — Niemand da, Kamera geparkt
  SEARCHING    — Person erkannt, Suche laeuft
  TRACKING     — Gesicht bestaetigt, aktives Tracking
  INTERACTION  — Owner erkannt, Interaktionsmodus

Transitions:
  person_detected   -> SEARCHING
  face_confirmed    -> TRACKING
  target_lost+5s    -> IDLE
  owner_detected    -> INTERACTION

Event Bus:
  Priority 1 = Bridge subscribt (Perception-Events rein)
  Priority 2 = Bridge publisht (Aktions-Events raus)

Der autonomous_tracker.py bleibt unangetastet — die Bridge
ist eine PARALLELE Beobachtungs- und Entscheidungsschicht.

Author: M.O.L.O.C.H. System (Gate 1)
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

from core.moloch_event_bus import get_event_bus, PRIO_PERCEPTION, PRIO_ACTION

logger = logging.getLogger("ActionBridge")


# ============================================================
# ACTION BRIDGE FSM
# ============================================================

class BridgeState(Enum):
    """Action Bridge Zustaende."""
    IDLE = "idle"
    SEARCHING = "searching"
    TRACKING = "tracking"
    INTERACTION = "interaction"


@dataclass
class BridgeContext:
    """Aktueller Kontext der Bridge — wird bei jedem Thought aktualisiert."""
    person_detected: bool = False
    face_confirmed: bool = False
    owner_detected: bool = False
    person_confidence: float = 0.0
    face_similarity: float = 0.0
    owner_name: str = ""
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    last_detection_time: float = 0.0
    last_face_time: float = 0.0
    last_owner_time: float = 0.0


class ActionBridge:
    """
    Action Bridge FSM — Thought/Intent/Action/Result Pipeline.

    Subscribt auf Perception-Events (Priority 1).
    Publisht Action-Events (Priority 2).

    Der autonomous_tracker laeuft PARALLEL und unabhaengig.
    Die Bridge ist eine Beobachtungs- und Entscheidungsschicht
    die zusaetzliche Aktionen ausloest (TTS, LED, Logging).
    """

    # Timeout: Kein Target -> zurueck zu IDLE
    TARGET_LOST_TIMEOUT = 5.0
    # Owner-Interaction Timeout: Kein Owner mehr -> zurueck zu TRACKING/IDLE
    INTERACTION_TIMEOUT = 15.0

    def __init__(self):
        self._state = BridgeState.IDLE
        self._prev_state = BridgeState.IDLE
        self._context = BridgeContext()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._bus = get_event_bus()
        self._state_enter_time = time.time()

        # Decision-Log: Jede Transition als Thought->Intent->Action->Result
        self._decision_log: List[dict] = []
        self._max_decisions = 200

        # Bus-Subscriptions (PRIO_PERCEPTION = Perception-Input)
        self._bus.subscribe("perception.person_detected", self._on_person_detected, priority=PRIO_PERCEPTION)
        self._bus.subscribe("perception.face_confirmed", self._on_face_confirmed, priority=PRIO_PERCEPTION)
        self._bus.subscribe("perception.owner_detected", self._on_owner_detected, priority=PRIO_PERCEPTION)
        self._bus.subscribe("perception.target_lost", self._on_target_lost, priority=PRIO_PERCEPTION)

        logger.info(f"[BRIDGE] Initialisiert, State={self._state.value}")

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def start(self):
        """Bridge-Thread starten (1 Hz Tick)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._tick_loop, daemon=True, name="ActionBridge")
        self._thread.start()
        logger.info("[BRIDGE] Gestartet (1 Hz)")

    def stop(self):
        """Bridge stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("[BRIDGE] Gestoppt")

    # ============================================================
    # TICK LOOP — Prueft Timeouts, triggert Transitionen
    # ============================================================

    def _tick_loop(self):
        """1 Hz Loop: Timeout-Pruefung und State-Wartung."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"[BRIDGE] Tick-Fehler: {e}")
            time.sleep(1.0)

    def _tick(self):
        """Ein Tick: Timeouts pruefen, ggf. Transition ausloesen."""
        now = time.time()
        with self._lock:
            state = self._state
            ctx = self._context

        # SEARCHING/TRACKING -> IDLE bei Target-Verlust
        if state in (BridgeState.SEARCHING, BridgeState.TRACKING):
            time_since = now - ctx.last_detection_time if ctx.last_detection_time > 0 else 999
            if time_since > self.TARGET_LOST_TIMEOUT:
                self._transition(
                    BridgeState.IDLE,
                    thought=f"Kein Target seit {time_since:.1f}s",
                    intent="park",
                    action_topic="action.park",
                    action_data={"reason": "target_lost_timeout"},
                )

        # INTERACTION -> TRACKING/IDLE bei Owner-Verlust
        if state == BridgeState.INTERACTION:
            time_since_owner = now - ctx.last_owner_time if ctx.last_owner_time > 0 else 999
            if time_since_owner > self.INTERACTION_TIMEOUT:
                # Zurueck zu TRACKING wenn noch Person da, sonst IDLE
                time_since_person = now - ctx.last_detection_time if ctx.last_detection_time > 0 else 999
                if time_since_person < self.TARGET_LOST_TIMEOUT:
                    self._transition(
                        BridgeState.TRACKING,
                        thought=f"Owner weg seit {time_since_owner:.1f}s, Person noch da",
                        intent="track",
                        action_topic="action.track_continue",
                        action_data={"reason": "owner_left"},
                    )
                else:
                    self._transition(
                        BridgeState.IDLE,
                        thought=f"Owner und Person weg",
                        intent="park",
                        action_topic="action.park",
                        action_data={"reason": "interaction_timeout"},
                    )

    # ============================================================
    # EVENT HANDLER (Priority 1 — Perception-Input)
    # ============================================================

    def _on_person_detected(self, event: dict):
        """Person erkannt (YOLO). IDLE -> SEARCHING."""
        data = event.get("payload", {})
        now = time.time()
        with self._lock:
            self._context.person_detected = True
            self._context.person_confidence = data.get("confidence", 0.0)
            self._context.bbox = data.get("bbox", [0, 0, 0, 0])
            self._context.last_detection_time = now

        if self._state == BridgeState.IDLE:
            self._transition(
                BridgeState.SEARCHING,
                thought=f"Person erkannt (conf={data.get('confidence', 0):.2f})",
                intent="search",
                action_topic="action.search_start",
                action_data={"confidence": data.get("confidence", 0)},
            )

        # ptz_track bei aktivem Tracking mit Person-BBox
        if self._state in (BridgeState.TRACKING, BridgeState.INTERACTION):
            self._publish_ptz_track()

    def _on_face_confirmed(self, event: dict):
        """Gesicht bestaetigt (SCRFD). SEARCHING -> TRACKING, ptz_track publishen."""
        data = event.get("payload", {})
        now = time.time()
        with self._lock:
            self._context.face_confirmed = True
            self._context.face_similarity = data.get("similarity", 0.0)
            self._context.bbox = data.get("bbox", self._context.bbox)
            self._context.last_face_time = now
            self._context.last_detection_time = now

        if self._state in (BridgeState.IDLE, BridgeState.SEARCHING):
            self._transition(
                BridgeState.TRACKING,
                thought=f"Gesicht bestaetigt (sim={data.get('similarity', 0):.2f})",
                intent="track",
                action_topic="action.track_start",
                action_data={"similarity": data.get("similarity", 0)},
            )

        # ptz_track Event mit BBox-Zentrum publishen (TRACKING/INTERACTION)
        self._publish_ptz_track()

    def _on_owner_detected(self, event: dict):
        """Owner erkannt (ArcFace Match). -> INTERACTION."""
        data = event.get("payload", {})
        now = time.time()
        with self._lock:
            self._context.owner_detected = True
            self._context.owner_name = data.get("name", "Markus")
            self._context.face_similarity = data.get("similarity", 0.0)
            self._context.last_owner_time = now
            self._context.last_detection_time = now

        if self._state != BridgeState.INTERACTION:
            self._transition(
                BridgeState.INTERACTION,
                thought=f"Owner erkannt: {data.get('name', 'Markus')} (sim={data.get('similarity', 0):.2f})",
                intent="greet",
                action_topic="action.greet_owner",
                action_data={"name": data.get("name", "Markus"), "similarity": data.get("similarity", 0)},
            )

    def _on_target_lost(self, event: dict):
        """Target verloren — Kontext aktualisieren, Timeout laeuft im Tick."""
        with self._lock:
            self._context.person_detected = False
            self._context.face_confirmed = False
            # last_detection_time bleibt stehen -> Timeout im Tick

    # ============================================================
    # PTZ TRACK — BBox-Zentrum als Zielkoordinaten publishen
    # ============================================================

    def _publish_ptz_track(self):
        """Publisht action.ptz_track mit BBox-Zentrum (normalisiert 0-1)."""
        with self._lock:
            bbox = self._context.bbox
        if not bbox or bbox == [0, 0, 0, 0]:
            return
        # BBox-Zentrum berechnen (x1, y1, x2, y2 normalisiert)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self._bus.publish(
            event_type="action.ptz_track",
            payload={"center_x": round(cx, 4), "center_y": round(cy, 4),
                     "bbox": bbox},
            source="action_bridge",
            priority=PRIO_ACTION,
        )

    # ============================================================
    # TRANSITION — Thought/Intent/Action/Result Pipeline
    # ============================================================

    def _transition(self, new_state: BridgeState, thought: str, intent: str,
                    action_topic: str, action_data: dict = None):
        """
        Zentrale Transition mit Thought->Intent->Action->Result.

        1. Thought:  Perception-Input auswerten (warum?)
        2. Intent:   Zielaktion bestimmen (was?)
        3. Action:   Event auf Bus publishen (ausfuehren)
        4. Result:   Outcome loggen (was ist passiert?)
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return  # Kein Selbst-Uebergang
            self._prev_state = old_state
            self._state = new_state
            self._state_enter_time = time.time()

            # Kontext-Reset bei IDLE
            if new_state == BridgeState.IDLE:
                self._context.person_detected = False
                self._context.face_confirmed = False
                self._context.owner_detected = False

        # --- THOUGHT ---
        logger.info(f"[BRIDGE] THOUGHT: {thought}")

        # --- INTENT ---
        logger.info(f"[BRIDGE] INTENT: {intent} ({old_state.value} -> {new_state.value})")

        # --- ACTION ---
        self._bus.publish(
            event_type=action_topic,
            payload=action_data or {},
            source="action_bridge",
            priority=PRIO_ACTION,
        )
        logger.info(f"[BRIDGE] ACTION: {action_topic} -> Bus")

        # --- RESULT ---
        result = f"OK: {old_state.value} -> {new_state.value}"
        logger.info(f"[BRIDGE] RESULT: {result}")

        # Decision-Log (Ringbuffer)
        decision = {
            "timestamp": time.time(),
            "thought": thought,
            "intent": intent,
            "action": action_topic,
            "result": result,
            "old_state": old_state.value,
            "new_state": new_state.value,
        }
        self._decision_log.append(decision)
        if len(self._decision_log) > self._max_decisions:
            self._decision_log.pop(0)

    # ============================================================
    # PUBLIC API
    # ============================================================

    @property
    def state(self) -> BridgeState:
        """Aktueller FSM-State."""
        with self._lock:
            return self._state

    @property
    def context(self) -> BridgeContext:
        """Aktueller Kontext (read-only Snapshot)."""
        with self._lock:
            return self._context

    def get_status(self) -> dict:
        """Status-Dict fuer IPC/Panel."""
        with self._lock:
            return {
                "state": self._state.value,
                "prev_state": self._prev_state.value,
                "state_age_s": round(time.time() - self._state_enter_time, 1),
                "person_detected": self._context.person_detected,
                "face_confirmed": self._context.face_confirmed,
                "owner_detected": self._context.owner_detected,
                "owner_name": self._context.owner_name,
                "decisions": len(self._decision_log),
            }

    def get_decisions(self, count: int = 20) -> List[dict]:
        """Letzte N Decisions fuer Debug/Panel."""
        return self._decision_log[-count:]


# ============================================================
# SINGLETON
# ============================================================

_bridge_instance: Optional[ActionBridge] = None
_bridge_lock = threading.Lock()


def get_action_bridge() -> ActionBridge:
    """Singleton-Zugriff auf die Action Bridge."""
    global _bridge_instance
    with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = ActionBridge()
        return _bridge_instance
