#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Gesture Detection
==============================

Detects gestures from YOLOv8 Pose keypoints:
- Wave: Wrist moving side-to-side above shoulder
- Hand raised: Wrist above shoulder level
- Point: Arm extended in one direction
- Hands up: Both wrists above head
- Arms crossed: Wrists near opposite shoulders

COCO Keypoint indices:
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Types of detectable gestures."""
    NONE = "none"
    WAVE_LEFT = "wave_left"
    WAVE_RIGHT = "wave_right"
    HAND_RAISED_LEFT = "hand_raised_left"
    HAND_RAISED_RIGHT = "hand_raised_right"
    HANDS_UP = "hands_up"
    POINTING_LEFT = "pointing_left"
    POINTING_RIGHT = "pointing_right"
    ARMS_CROSSED = "arms_crossed"
    THUMBS_UP = "thumbs_up"  # Approximated from wrist position


@dataclass
class Gesture:
    """Detected gesture with confidence and duration."""
    type: GestureType
    confidence: float
    duration_ms: int = 0
    hand: str = "unknown"  # "left", "right", "both"

    def __str__(self):
        return f"{self.type.value} ({self.confidence:.0%}, {self.duration_ms}ms)"


@dataclass
class KeypointPosition:
    """Position of a keypoint (normalized 0-1)."""
    x: float
    y: float
    confidence: float
    visible: bool = False

    def __post_init__(self):
        self.visible = self.confidence > 0.3


class GestureDetector:
    """
    Detects gestures from pose keypoints.

    Uses temporal analysis (history of positions) to detect
    dynamic gestures like waving.
    """

    # Keypoint indices (COCO format)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12

    # Gesture detection thresholds (more lenient for easier triggering)
    HAND_RAISED_THRESHOLD = 0.02  # Wrist must be this much above shoulder (normalized) - very small
    HANDS_UP_THRESHOLD = 0.10  # Both wrists above head
    WAVE_AMPLITUDE = 0.03  # Minimum x-movement for wave detection
    WAVE_FREQUENCY_MIN = 2  # Minimum direction changes for wave
    POINTING_ARM_EXTENSION = 0.12  # Minimum arm extension for pointing

    # History settings
    HISTORY_SIZE = 15  # Frames to keep for temporal analysis
    WAVE_WINDOW_MS = 1500  # Time window for wave detection

    def __init__(self):
        """Initialize gesture detector."""
        # History of wrist positions for temporal analysis
        self._left_wrist_history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._right_wrist_history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._timestamps: deque = deque(maxlen=self.HISTORY_SIZE)

        # Current gesture state
        self._current_gesture: Optional[Gesture] = None
        self._gesture_start_time: float = 0

        # Statistics
        self.gestures_detected: int = 0

        logger.info("GestureDetector initialized")

    def detect(self, keypoints: List[KeypointPosition]) -> Optional[Gesture]:
        """
        Detect gesture from keypoints.

        Args:
            keypoints: List of 17 COCO keypoints with x, y, confidence

        Returns:
            Detected gesture or None
        """
        if len(keypoints) < 17:
            return None

        now = time.time()

        # Extract relevant keypoints
        nose = keypoints[self.NOSE]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_elbow = keypoints[self.LEFT_ELBOW]
        right_elbow = keypoints[self.RIGHT_ELBOW]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]

        # Update history
        self._timestamps.append(now)
        if left_wrist.visible:
            self._left_wrist_history.append((now, left_wrist.x, left_wrist.y))
        if right_wrist.visible:
            self._right_wrist_history.append((now, right_wrist.x, right_wrist.y))

        # Check for gestures (priority order)
        gesture = None

        # 1. Hands up (both wrists above head)
        gesture = self._detect_hands_up(nose, left_wrist, right_wrist)
        if gesture:
            return self._update_gesture(gesture)

        # 2. Wave detection (requires history)
        gesture = self._detect_wave(left_shoulder, right_shoulder,
                                     left_wrist, right_wrist, now)
        if gesture:
            return self._update_gesture(gesture)

        # 3. Hand raised
        gesture = self._detect_hand_raised(left_shoulder, right_shoulder,
                                           left_wrist, right_wrist)
        if gesture:
            return self._update_gesture(gesture)

        # 4. Pointing
        gesture = self._detect_pointing(left_shoulder, right_shoulder,
                                        left_elbow, right_elbow,
                                        left_wrist, right_wrist)
        if gesture:
            return self._update_gesture(gesture)

        # 5. Arms crossed
        gesture = self._detect_arms_crossed(left_shoulder, right_shoulder,
                                            left_wrist, right_wrist)
        if gesture:
            return self._update_gesture(gesture)

        # No gesture detected
        if self._current_gesture:
            self._current_gesture = None
        return None

    def _detect_hands_up(self, nose: KeypointPosition,
                         left_wrist: KeypointPosition,
                         right_wrist: KeypointPosition) -> Optional[Gesture]:
        """Detect both hands raised above head."""
        if not (nose.visible and left_wrist.visible and right_wrist.visible):
            return None

        # Both wrists must be above nose level
        left_above = left_wrist.y < nose.y - self.HANDS_UP_THRESHOLD
        right_above = right_wrist.y < nose.y - self.HANDS_UP_THRESHOLD

        if left_above and right_above:
            confidence = min(left_wrist.confidence, right_wrist.confidence)
            return Gesture(
                type=GestureType.HANDS_UP,
                confidence=confidence,
                hand="both"
            )
        return None

    def _detect_hand_raised(self, left_shoulder: KeypointPosition,
                            right_shoulder: KeypointPosition,
                            left_wrist: KeypointPosition,
                            right_wrist: KeypointPosition) -> Optional[Gesture]:
        """Detect single hand raised above shoulder."""

        # Check left hand
        if left_wrist.visible and left_shoulder.visible:
            if left_wrist.y < left_shoulder.y - self.HAND_RAISED_THRESHOLD:
                return Gesture(
                    type=GestureType.HAND_RAISED_LEFT,
                    confidence=left_wrist.confidence,
                    hand="left"
                )

        # Check right hand
        if right_wrist.visible and right_shoulder.visible:
            if right_wrist.y < right_shoulder.y - self.HAND_RAISED_THRESHOLD:
                return Gesture(
                    type=GestureType.HAND_RAISED_RIGHT,
                    confidence=right_wrist.confidence,
                    hand="right"
                )

        return None

    def _detect_wave(self, left_shoulder: KeypointPosition,
                     right_shoulder: KeypointPosition,
                     left_wrist: KeypointPosition,
                     right_wrist: KeypointPosition,
                     now: float) -> Optional[Gesture]:
        """Detect waving gesture from wrist movement history."""

        # Check right hand wave (more common)
        if len(self._right_wrist_history) >= 5:
            wave = self._analyze_wave_motion(
                self._right_wrist_history,
                right_shoulder,
                now
            )
            if wave:
                return Gesture(
                    type=GestureType.WAVE_RIGHT,
                    confidence=right_wrist.confidence,
                    hand="right"
                )

        # Check left hand wave
        if len(self._left_wrist_history) >= 5:
            wave = self._analyze_wave_motion(
                self._left_wrist_history,
                left_shoulder,
                now
            )
            if wave:
                return Gesture(
                    type=GestureType.WAVE_LEFT,
                    confidence=left_wrist.confidence,
                    hand="left"
                )

        return None

    def _analyze_wave_motion(self, history: deque,
                              shoulder: KeypointPosition,
                              now: float) -> bool:
        """Analyze wrist history for wave pattern."""
        if not shoulder.visible:
            return False

        # Filter to recent positions within wave window
        window_start = now - (self.WAVE_WINDOW_MS / 1000.0)
        recent = [(t, x, y) for t, x, y in history if t >= window_start]

        if len(recent) < 4:
            return False

        # Check if wrist is above shoulder (wave position)
        latest_y = recent[-1][2]
        if latest_y > shoulder.y:
            return False  # Wrist below shoulder, not waving

        # Count direction changes in x-axis (wave motion)
        x_positions = [x for _, x, _ in recent]
        direction_changes = 0
        last_direction = 0

        for i in range(1, len(x_positions)):
            diff = x_positions[i] - x_positions[i-1]
            if abs(diff) > self.WAVE_AMPLITUDE / 2:
                direction = 1 if diff > 0 else -1
                if last_direction != 0 and direction != last_direction:
                    direction_changes += 1
                last_direction = direction

        # Wave detected if enough direction changes
        return direction_changes >= self.WAVE_FREQUENCY_MIN

    def _detect_pointing(self, left_shoulder: KeypointPosition,
                         right_shoulder: KeypointPosition,
                         left_elbow: KeypointPosition,
                         right_elbow: KeypointPosition,
                         left_wrist: KeypointPosition,
                         right_wrist: KeypointPosition) -> Optional[Gesture]:
        """Detect pointing gesture (extended arm)."""

        # Check right arm pointing
        if (right_shoulder.visible and right_elbow.visible and
            right_wrist.visible):
            # Calculate arm extension (wrist distance from shoulder)
            arm_length = abs(right_wrist.x - right_shoulder.x)

            # Check if arm is relatively horizontal and extended
            vertical_diff = abs(right_wrist.y - right_shoulder.y)
            if arm_length > self.POINTING_ARM_EXTENSION and vertical_diff < 0.15:
                direction = "right" if right_wrist.x > right_shoulder.x else "left"
                return Gesture(
                    type=GestureType.POINTING_RIGHT if direction == "right" else GestureType.POINTING_LEFT,
                    confidence=right_wrist.confidence,
                    hand="right"
                )

        # Check left arm pointing
        if (left_shoulder.visible and left_elbow.visible and
            left_wrist.visible):
            arm_length = abs(left_wrist.x - left_shoulder.x)
            vertical_diff = abs(left_wrist.y - left_shoulder.y)
            if arm_length > self.POINTING_ARM_EXTENSION and vertical_diff < 0.15:
                direction = "left" if left_wrist.x < left_shoulder.x else "right"
                return Gesture(
                    type=GestureType.POINTING_LEFT if direction == "left" else GestureType.POINTING_RIGHT,
                    confidence=left_wrist.confidence,
                    hand="left"
                )

        return None

    def _detect_arms_crossed(self, left_shoulder: KeypointPosition,
                             right_shoulder: KeypointPosition,
                             left_wrist: KeypointPosition,
                             right_wrist: KeypointPosition) -> Optional[Gesture]:
        """Detect arms crossed gesture."""
        if not all([left_shoulder.visible, right_shoulder.visible,
                    left_wrist.visible, right_wrist.visible]):
            return None

        # Arms crossed: wrists near opposite shoulders
        # Left wrist near right shoulder
        left_to_right = abs(left_wrist.x - right_shoulder.x) < 0.1
        # Right wrist near left shoulder
        right_to_left = abs(right_wrist.x - left_shoulder.x) < 0.1

        # Both wrists at similar height (chest level)
        wrists_level = abs(left_wrist.y - right_wrist.y) < 0.1

        if left_to_right and right_to_left and wrists_level:
            confidence = min(left_wrist.confidence, right_wrist.confidence)
            return Gesture(
                type=GestureType.ARMS_CROSSED,
                confidence=confidence,
                hand="both"
            )

        return None

    def _update_gesture(self, gesture: Gesture) -> Gesture:
        """Update gesture state and calculate duration."""
        now = time.time()

        if (self._current_gesture and
            self._current_gesture.type == gesture.type):
            # Same gesture continues
            gesture.duration_ms = int((now - self._gesture_start_time) * 1000)
        else:
            # New gesture
            self._gesture_start_time = now
            self._current_gesture = gesture
            self.gestures_detected += 1
            logger.info(f"[GESTURE] Detected: {gesture.type.value}")

        return gesture

    def reset(self):
        """Reset gesture detection state."""
        self._left_wrist_history.clear()
        self._right_wrist_history.clear()
        self._timestamps.clear()
        self._current_gesture = None
        self._gesture_start_time = 0


# Singleton instance
_gesture_detector: Optional[GestureDetector] = None


def get_gesture_detector() -> GestureDetector:
    """Get singleton GestureDetector instance."""
    global _gesture_detector
    if _gesture_detector is None:
        _gesture_detector = GestureDetector()
    return _gesture_detector
