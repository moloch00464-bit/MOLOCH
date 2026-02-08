#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Perception State
=============================

Central perception state object that tracks:
- user_visible: Valid person detected (face + torso)
- face_visible: Face keypoints detected
- gesture_visible: Active gesture detected

States persist for 2 seconds after last detection to prevent flicker.
Used by autonomy layer and dialogue system.

Author: M.O.L.O.C.H. System
Date: 2026-02-04
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PerceptionEvent(Enum):
    """Perception state change events."""
    USER_APPEARED = "user_appeared"
    USER_DISAPPEARED = "user_disappeared"
    FACE_VISIBLE = "face_visible"
    FACE_LOST = "face_lost"
    GESTURE_STARTED = "gesture_started"
    GESTURE_ENDED = "gesture_ended"


@dataclass
class PerceptionSnapshot:
    """Immutable snapshot of perception state."""
    user_visible: bool = False
    face_visible: bool = False
    gesture_visible: bool = False
    gesture_type: str = "none"

    # Detection details
    person_count: int = 0
    confidence: float = 0.0
    face_confidence: float = 0.0
    gesture_confidence: float = 0.0

    # Keypoint counts
    face_keypoints: int = 0
    torso_keypoints: int = 0
    wrist_keypoints: int = 0

    # Timestamps
    last_user_seen: float = 0.0
    last_face_seen: float = 0.0
    last_gesture_seen: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # Source tracking
    source: str = "none"


class PerceptionState:
    """
    Central perception state with timeout-based persistence.

    States persist for TIMEOUT_SECONDS after last detection to prevent
    rapid flickering that confuses the autonomy layer.
    """

    TIMEOUT_SECONDS = 2.0  # States persist this long after last detection

    def __init__(self):
        self._lock = threading.RLock()

        # Visibility flags
        self._user_visible = False
        self._face_visible = False
        self._gesture_visible = False
        self._gesture_type = "none"

        # Detection details
        self._person_count = 0
        self._confidence = 0.0
        self._face_confidence = 0.0
        self._gesture_confidence = 0.0

        # Keypoint counts
        self._face_keypoints = 0
        self._torso_keypoints = 0
        self._wrist_keypoints = 0

        # Timestamps for timeout tracking
        self._last_user_seen = 0.0
        self._last_face_seen = 0.0
        self._last_gesture_seen = 0.0
        self._last_update = 0.0

        # Source
        self._source = "none"

        # Event callbacks
        self._on_event: Optional[Callable[[PerceptionEvent, Dict[str, Any]], None]] = None

        # Previous state for change detection
        self._prev_user_visible = False
        self._prev_face_visible = False
        self._prev_gesture_visible = False

        logger.info("PerceptionState initialized (timeout=%.1fs)", self.TIMEOUT_SECONDS)

    def update(self,
               user_detected: bool = False,
               face_detected: bool = False,
               gesture_detected: bool = False,
               gesture_type: str = "none",
               person_count: int = 0,
               confidence: float = 0.0,
               face_confidence: float = 0.0,
               gesture_confidence: float = 0.0,
               face_keypoints: int = 0,
               torso_keypoints: int = 0,
               wrist_keypoints: int = 0,
               source: str = "unknown") -> None:
        """
        Update perception state from detection results.

        Called by pose detector callback on each frame.
        """
        now = time.time()

        with self._lock:
            # Update timestamps when detected
            if user_detected:
                self._last_user_seen = now
            if face_detected:
                self._last_face_seen = now
            if gesture_detected:
                self._last_gesture_seen = now

            # Update detection details
            self._person_count = person_count
            self._confidence = confidence
            self._face_confidence = face_confidence
            self._gesture_confidence = gesture_confidence
            self._face_keypoints = face_keypoints
            self._torso_keypoints = torso_keypoints
            self._wrist_keypoints = wrist_keypoints
            self._gesture_type = gesture_type
            self._source = source
            self._last_update = now

            # Calculate visibility with timeout
            # User is visible if seen recently (within timeout)
            self._user_visible = (now - self._last_user_seen) < self.TIMEOUT_SECONDS
            self._face_visible = (now - self._last_face_seen) < self.TIMEOUT_SECONDS
            self._gesture_visible = (now - self._last_gesture_seen) < self.TIMEOUT_SECONDS

            # Detect state changes and fire events
            self._check_state_changes()

    def _check_state_changes(self) -> None:
        """Check for state changes and fire events."""
        events = []

        # User visibility changed
        if self._user_visible != self._prev_user_visible:
            if self._user_visible:
                events.append((PerceptionEvent.USER_APPEARED, {
                    "confidence": self._confidence,
                    "person_count": self._person_count
                }))
                logger.info("[PERCEPTION] User APPEARED (conf=%.2f, count=%d)",
                           self._confidence, self._person_count)
            else:
                events.append((PerceptionEvent.USER_DISAPPEARED, {}))
                logger.info("[PERCEPTION] User DISAPPEARED (timeout)")
            self._prev_user_visible = self._user_visible

        # Face visibility changed
        if self._face_visible != self._prev_face_visible:
            if self._face_visible:
                events.append((PerceptionEvent.FACE_VISIBLE, {
                    "confidence": self._face_confidence,
                    "keypoints": self._face_keypoints
                }))
                logger.info("[PERCEPTION] Face VISIBLE (conf=%.2f, kp=%d)",
                           self._face_confidence, self._face_keypoints)
            else:
                events.append((PerceptionEvent.FACE_LOST, {}))
                logger.info("[PERCEPTION] Face LOST (timeout)")
            self._prev_face_visible = self._face_visible

        # Gesture visibility changed
        if self._gesture_visible != self._prev_gesture_visible:
            if self._gesture_visible:
                events.append((PerceptionEvent.GESTURE_STARTED, {
                    "type": self._gesture_type,
                    "confidence": self._gesture_confidence
                }))
                logger.info("[PERCEPTION] Gesture STARTED: %s (conf=%.2f)",
                           self._gesture_type, self._gesture_confidence)
            else:
                events.append((PerceptionEvent.GESTURE_ENDED, {}))
                logger.info("[PERCEPTION] Gesture ENDED (timeout)")
            self._prev_gesture_visible = self._gesture_visible

        # Fire event callbacks
        if self._on_event and events:
            for event, data in events:
                try:
                    self._on_event(event, data)
                except Exception as e:
                    logger.error("Perception event callback error: %s", e)

    def get_snapshot(self) -> PerceptionSnapshot:
        """Get immutable snapshot of current perception state."""
        with self._lock:
            return PerceptionSnapshot(
                user_visible=self._user_visible,
                face_visible=self._face_visible,
                gesture_visible=self._gesture_visible,
                gesture_type=self._gesture_type if self._gesture_visible else "none",
                person_count=self._person_count,
                confidence=self._confidence,
                face_confidence=self._face_confidence,
                gesture_confidence=self._gesture_confidence,
                face_keypoints=self._face_keypoints,
                torso_keypoints=self._torso_keypoints,
                wrist_keypoints=self._wrist_keypoints,
                last_user_seen=self._last_user_seen,
                last_face_seen=self._last_face_seen,
                last_gesture_seen=self._last_gesture_seen,
                timestamp=self._last_update,
                source=self._source
            )

    @property
    def user_visible(self) -> bool:
        """Check if user is currently visible (with timeout)."""
        with self._lock:
            now = time.time()
            return (now - self._last_user_seen) < self.TIMEOUT_SECONDS

    @property
    def face_visible(self) -> bool:
        """Check if face is currently visible (with timeout)."""
        with self._lock:
            now = time.time()
            return (now - self._last_face_seen) < self.TIMEOUT_SECONDS

    @property
    def gesture_visible(self) -> bool:
        """Check if gesture is currently visible (with timeout)."""
        with self._lock:
            now = time.time()
            return (now - self._last_gesture_seen) < self.TIMEOUT_SECONDS

    @property
    def current_gesture(self) -> str:
        """Get current gesture type or 'none'."""
        with self._lock:
            if self.gesture_visible:
                return self._gesture_type
            return "none"

    def set_event_callback(self, callback: Callable[[PerceptionEvent, Dict[str, Any]], None]) -> None:
        """Set callback for perception state change events."""
        self._on_event = callback

    def describe(self) -> str:
        """
        Get natural language description of perception state.
        For dialogue system integration.
        """
        snap = self.get_snapshot()

        if not snap.user_visible:
            return "Ich sehe niemanden."

        parts = []

        # User presence
        if snap.person_count == 1:
            if snap.confidence > 0.8:
                parts.append("Ich sehe dich")
            else:
                parts.append("Ich sehe jemanden")
        else:
            parts.append(f"Ich sehe {snap.person_count} Personen")

        # Face detail
        if snap.face_visible:
            if snap.face_keypoints >= 3:
                parts.append("dein Gesicht ist gut sichtbar")
            else:
                parts.append("ich kann dein Gesicht erkennen")

        # Gesture
        if snap.gesture_visible and snap.gesture_type != "none":
            gesture_names = {
                "wave_left": "du winkst mit links",
                "wave_right": "du winkst mit rechts",
                "hand_raised_left": "deine linke Hand ist oben",
                "hand_raised_right": "deine rechte Hand ist oben",
                "hands_up": "beide Hände sind oben",
                "pointing_left": "du zeigst nach links",
                "pointing_right": "du zeigst nach rechts",
            }
            gesture_text = gesture_names.get(snap.gesture_type, f"Geste: {snap.gesture_type}")
            parts.append(gesture_text)

        return ", ".join(parts) + "."

    def __str__(self) -> str:
        snap = self.get_snapshot()
        return (f"PerceptionState(user={snap.user_visible}, face={snap.face_visible}, "
                f"gesture={snap.gesture_type if snap.gesture_visible else 'none'})")


# Singleton instance
_perception_state: Optional[PerceptionState] = None
_perception_lock = threading.Lock()


def get_perception_state() -> PerceptionState:
    """Get or create singleton PerceptionState."""
    global _perception_state
    if _perception_state is None:
        with _perception_lock:
            if _perception_state is None:
                _perception_state = PerceptionState()
    return _perception_state


# Convenience functions
def is_user_visible() -> bool:
    """Quick check if user is visible."""
    return get_perception_state().user_visible


def is_face_visible() -> bool:
    """Quick check if face is visible."""
    return get_perception_state().face_visible


def get_current_gesture() -> str:
    """Get current gesture type."""
    return get_perception_state().current_gesture
