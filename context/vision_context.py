#!/usr/bin/env python3
"""
M.O.L.O.C.H. Vision Context
============================

Shared runtime state for vision events.
Written by Vision Worker, read by Brain.

Philosophy:
- Runtime only (no persistence)
- Event-driven updates
- Thread-safe access
- Brain may only claim vision if events are present
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class VisionEventType(Enum):
    """Types of vision events."""
    PERSON_DETECTED = "person_detected"
    PERSON_LOST = "person_lost"
    FACE_DETECTED = "face_detected"
    PERSON_IDENTIFIED = "person_identified"
    MOTION_DETECTED = "motion_detected"
    SCENE_CHANGE = "scene_change"


@dataclass
class VisionEvent:
    """A single vision event."""
    event_type: VisionEventType
    timestamp: float
    confidence: float = 0.0
    person_count: int = 0
    person_name: Optional[str] = None
    person_id: Optional[str] = None
    source: str = "unknown"  # "sonoff", "hailo"
    bbox: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class VisionState:
    """Current vision state snapshot."""
    # Core state
    person_detected: bool = False
    person_count: int = 0
    face_detected: bool = False
    confidence: float = 0.0
    
    # Position for PTZ tracking
    target_center_x: int = 0  # X-coordinate of target center
    frame_width: int = 1920   # Frame width for normalization

    # Identification (if known person)
    person_name: Optional[str] = None
    person_id: Optional[str] = None
    is_known_person: bool = False

    # Timing
    last_update: float = 0.0
    last_person_seen: float = 0.0

    # Source info
    source: str = "none"  # Which camera/system provided this

    # Connection status
    camera_connected: bool = False
    npu_active: bool = False


class VisionContext:
    """
    M.O.L.O.C.H. Vision Context - Thread-safe shared state.

    Usage:
        # Vision Worker writes:
        context = get_vision_context()
        context.update_detection(person_detected=True, person_count=1, confidence=0.85)

        # Brain reads:
        state = context.get_state()
        if state.person_detected:
            # Can say "Ich sehe jemanden"
    """

    _instance = None
    _lock = threading.Lock()

    # State timeout - after this, consider vision stale
    STATE_TIMEOUT = 5.0  # seconds

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Current state
        self._state = VisionState()
        self._state_lock = threading.RLock()

        # Event history (last N events)
        self._events: List[VisionEvent] = []
        self._max_events = 100

        # Callbacks for state changes
        self._on_person_detected = None
        self._on_person_lost = None
        self._on_person_identified = None

    def update_detection(self,
                        person_detected: bool,
                        person_count: int = 0,
                        face_detected: bool = False,
                        confidence: float = 0.0,
                        source: str = "unknown",
                        camera_connected: bool = True,
                        npu_active: bool = True,
                        target_center_x: int = 0,
                        frame_width: int = 1920) -> None:
        """
        Update detection state (called by Vision Worker).

        Args:
            person_detected: True if person is visible
            person_count: Number of persons detected
            face_detected: True if face is clearly visible
            confidence: Detection confidence 0.0-1.0
            source: Source camera/system
            camera_connected: Is camera connected
            npu_active: Is NPU running inference
        """
        with self._state_lock:
            was_person = self._state.person_detected
            now = time.time()

            self._state.person_detected = person_detected
            self._state.person_count = person_count
            self._state.face_detected = face_detected
            self._state.confidence = confidence
            self._state.source = source
            self._state.target_center_x = target_center_x
            self._state.frame_width = frame_width
            self._state.camera_connected = camera_connected
            self._state.npu_active = npu_active
            self._state.last_update = now

            if person_detected:
                self._state.last_person_seen = now

            # Fire events
            if person_detected and not was_person:
                self._add_event(VisionEvent(
                    event_type=VisionEventType.PERSON_DETECTED,
                    timestamp=now,
                    confidence=confidence,
                    person_count=person_count,
                    source=source
                ))
                if self._on_person_detected:
                    try:
                        self._on_person_detected(self._state)
                    except Exception:
                        pass

            elif not person_detected and was_person:
                self._add_event(VisionEvent(
                    event_type=VisionEventType.PERSON_LOST,
                    timestamp=now,
                    source=source
                ))
                # Clear identification when person leaves
                self._state.person_name = None
                self._state.person_id = None
                self._state.is_known_person = False

                if self._on_person_lost:
                    try:
                        self._on_person_lost()
                    except Exception:
                        pass

    def update_identification(self,
                             person_name: str,
                             person_id: str,
                             confidence: float = 0.0,
                             source: str = "hailo") -> None:
        """
        Update person identification (called when face is recognized).

        Args:
            person_name: Recognized person's name
            person_id: Unique ID
            confidence: Recognition confidence
            source: Source of identification
        """
        with self._state_lock:
            self._state.person_name = person_name
            self._state.person_id = person_id
            self._state.is_known_person = True
            self._state.confidence = confidence
            self._state.last_update = time.time()

            self._add_event(VisionEvent(
                event_type=VisionEventType.PERSON_IDENTIFIED,
                timestamp=time.time(),
                person_name=person_name,
                person_id=person_id,
                confidence=confidence,
                source=source
            ))

            if self._on_person_identified:
                try:
                    self._on_person_identified(self._state)
                except Exception:
                    pass

    def clear(self) -> None:
        """Clear all vision state (e.g., when worker stops)."""
        with self._state_lock:
            self._state = VisionState()

    def _add_event(self, event: VisionEvent) -> None:
        """Add event to history."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)

    # === READ METHODS (for Brain) ===

    def get_state(self) -> VisionState:
        """
        Get current vision state (thread-safe copy).

        Returns:
            VisionState snapshot
        """
        with self._state_lock:
            # Check for staleness
            now = time.time()
            if self._state.last_update > 0 and now - self._state.last_update > self.STATE_TIMEOUT:
                # State is stale - consider camera disconnected
                return VisionState(
                    camera_connected=False,
                    npu_active=False,
                    last_update=self._state.last_update
                )

            # Return copy
            return VisionState(
                person_detected=self._state.person_detected,
                person_count=self._state.person_count,
                face_detected=self._state.face_detected,
                confidence=self._state.confidence,
                person_name=self._state.person_name,
                person_id=self._state.person_id,
                is_known_person=self._state.is_known_person,
                last_update=self._state.last_update,
                last_person_seen=self._state.last_person_seen,
                source=self._state.source,
                camera_connected=self._state.camera_connected,
                npu_active=self._state.npu_active
            )

    def describe(self) -> str:
        """
        Get natural language description of what is seen.

        For M.O.L.O.C.H. to use in responses.
        Only returns vision claims if state is fresh.

        Returns:
            Description string
        """
        state = self.get_state()

        # No camera
        if not state.camera_connected:
            return "Ich kann gerade nicht sehen - Kamera nicht verbunden."

        # Stale state
        if state.last_update == 0:
            return "Mein Sehen ist noch nicht initialisiert."

        now = time.time()
        if now - state.last_update > self.STATE_TIMEOUT:
            return "Meine letzte Sicht ist veraltet."

        # Fresh state
        if not state.person_detected:
            return "Ich sehe niemanden im Moment."

        # Person detected
        if state.is_known_person and state.person_name:
            if state.confidence > 0.8:
                return f"Ich sehe {state.person_name}!"
            else:
                return f"Ich glaube, ich sehe {state.person_name}."

        # Unknown person
        if state.person_count == 1:
            if state.confidence > 0.8:
                return "Ich sehe jemanden!"
            elif state.confidence > 0.5:
                return "Ich sehe eine Person, aber nicht ganz scharf."
            else:
                return "Da ist jemand, glaube ich."
        else:
            return f"Ich sehe {state.person_count} Personen!"

    def is_seeing_person(self) -> bool:
        """Quick check if a person is currently visible."""
        state = self.get_state()
        return state.person_detected and state.camera_connected

    def get_recent_events(self, seconds: float = 60.0) -> List[VisionEvent]:
        """Get recent vision events."""
        cutoff = time.time() - seconds
        return [e for e in self._events if e.timestamp > cutoff]

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary for JSON serialization."""
        state = self.get_state()
        return {
            "person_detected": state.person_detected,
            "person_count": state.person_count,
            "face_detected": state.face_detected,
            "confidence": round(state.confidence, 2),
            "person_name": state.person_name,
            "is_known_person": state.is_known_person,
            "camera_connected": state.camera_connected,
            "npu_active": state.npu_active,
            "last_update": state.last_update,
            "source": state.source
        }

    # === CALLBACKS ===

    def set_callbacks(self,
                     on_person_detected=None,
                     on_person_lost=None,
                     on_person_identified=None):
        """Set event callbacks."""
        self._on_person_detected = on_person_detected
        self._on_person_lost = on_person_lost
        self._on_person_identified = on_person_identified


# Singleton accessor
_vision_context: Optional[VisionContext] = None
_vc_lock = threading.Lock()


def get_vision_context() -> VisionContext:
    """Get or create VisionContext instance."""
    global _vision_context
    if _vision_context is None:
        with _vc_lock:
            if _vision_context is None:
                _vision_context = VisionContext()
    return _vision_context
