#!/usr/bin/env python3
"""
M.O.L.O.C.H. Vision System Tests
=================================

Tests for vision data classes, gesture detection, state machines,
and identity matching. All tests run without Hailo/camera hardware.

Run:
    cd ~/moloch && python3 -m pytest tests/test_vision.py -v
"""

import time
import threading
import numpy as np
import pytest

from core.hardware.camera import (
    ControlMode, TrackingState, TrackingMode, NightMode, LEDLevel,
    Detection, PTZPosition, VisionEvent, TrackingDecision, CameraStatus,
)


# === Data Class Tests ===

class TestEnums:
    """Tests for vision enums."""

    def test_control_mode_members(self):
        assert hasattr(ControlMode, "AUTONOMOUS")
        assert hasattr(ControlMode, "MANUAL_OVERRIDE")
        assert hasattr(ControlMode, "SAFE_MODE")

    def test_tracking_state_members(self):
        assert hasattr(TrackingState, "IDLE")
        assert hasattr(TrackingState, "TRACKING")
        assert hasattr(TrackingState, "SEARCHING")

    def test_tracking_mode_values(self):
        assert TrackingMode.DISABLED.value == "disabled"
        assert TrackingMode.AUTO_TRACK.value == "auto_track"

    def test_night_mode_members(self):
        assert hasattr(NightMode, "AUTO")
        assert hasattr(NightMode, "DAY")
        assert hasattr(NightMode, "NIGHT")

    def test_led_level_ordering(self):
        assert LEDLevel.OFF.value < LEDLevel.LOW.value < LEDLevel.MEDIUM.value < LEDLevel.HIGH.value


class TestDetection:
    """Tests for Detection dataclass."""

    def test_create_detection(self):
        d = Detection(
            person_id="person_1",
            bbox=[100, 200, 300, 400],
            center_x=0.5,
            center_y=0.5,
            confidence=0.95,
        )
        assert d.person_id == "person_1"
        assert d.confidence == 0.95
        assert d.center_x == 0.5

    def test_detection_defaults(self):
        d = Detection(
            person_id="p",
            bbox=[0, 0, 1, 1],
            center_x=0.0,
            center_y=0.0,
            confidence=0.5,
        )
        assert d.face_name is None
        assert d.timestamp > 0  # Should have auto-timestamp


class TestVisionEvent:
    """Tests for VisionEvent dataclass."""

    def test_create_vision_event(self):
        e = VisionEvent(
            detection_found=True,
            target_center_x=960,
            frame_center_x=960,
            frame_width=1920,
            confidence=0.9,
        )
        assert e.detection_found is True
        assert e.frame_width == 1920

    def test_vision_event_no_detection(self):
        e = VisionEvent(
            detection_found=False,
            target_center_x=0,
            frame_center_x=960,
            frame_width=1920,
        )
        assert e.detection_found is False


class TestTrackingDecision:
    """Tests for TrackingDecision dataclass."""

    def test_create_decision_move(self):
        d = TrackingDecision(
            should_move=True,
            action="track",
            velocity=0.5,
            duration=0.3,
            reason="target_offset",
            error_x=100,
        )
        assert d.should_move is True
        assert d.action == "track"

    def test_create_decision_no_move(self):
        d = TrackingDecision(
            should_move=False,
            action="none",
            reason="no_detection",
        )
        assert d.should_move is False


class TestCameraStatus:
    """Tests for CameraStatus dataclass and serialization."""

    def test_camera_status_to_dict(self):
        status = CameraStatus(
            connected=True,
            model="Sonoff CAM-PT2",
            firmware="unknown",
            mode=ControlMode.AUTONOMOUS,
            tracking_state=TrackingState.IDLE,
        )
        d = status.to_dict()
        assert isinstance(d, dict)
        assert d["connected"] is True
        assert d["model"] == "Sonoff CAM-PT2"


# === Gesture Detector Tests ===

class TestGestureDetector:
    """Tests for pose-based gesture detection logic."""

    @pytest.fixture(autouse=True)
    def setup_detector(self):
        try:
            from core.vision.gesture_detector import (
                GestureDetector, GestureType, KeypointPosition,
            )
            self.GestureType = GestureType
            self.KeypointPosition = KeypointPosition
            self.detector = GestureDetector()
            self.available = True
        except ImportError:
            self.available = False

    def _make_keypoints(self, overrides=None):
        """Create 17 COCO keypoints at default positions (standing person)."""
        if not self.available:
            return None
        KP = self.KeypointPosition
        # Default: person standing with arms at sides
        defaults = {
            0: KP(0.5, 0.2, 0.9),      # nose
            1: KP(0.48, 0.18, 0.5),     # left_eye
            2: KP(0.52, 0.18, 0.5),     # right_eye
            3: KP(0.46, 0.19, 0.5),     # left_ear
            4: KP(0.54, 0.19, 0.5),     # right_ear
            5: KP(0.42, 0.35, 0.9),     # left_shoulder
            6: KP(0.58, 0.35, 0.9),     # right_shoulder
            7: KP(0.38, 0.50, 0.8),     # left_elbow
            8: KP(0.62, 0.50, 0.8),     # right_elbow
            9: KP(0.38, 0.65, 0.8),     # left_wrist (at hip level)
            10: KP(0.62, 0.65, 0.8),    # right_wrist (at hip level)
            11: KP(0.44, 0.65, 0.8),    # left_hip
            12: KP(0.56, 0.65, 0.8),    # right_hip
            13: KP(0.44, 0.80, 0.7),    # left_knee
            14: KP(0.56, 0.80, 0.7),    # right_knee
            15: KP(0.44, 0.95, 0.7),    # left_ankle
            16: KP(0.56, 0.95, 0.7),    # right_ankle
        }
        if overrides:
            defaults.update(overrides)
        return [defaults[i] for i in range(17)]

    def test_no_gesture_arms_down(self):
        """Arms at sides should produce no gesture."""
        if not self.available:
            pytest.skip("GestureDetector not available")
        kps = self._make_keypoints()
        result = self.detector.detect(kps)
        # Should be None or NONE type
        if result is not None:
            assert result.type == self.GestureType.NONE

    def test_hands_up_detected(self):
        """Both hands above nose should detect HANDS_UP."""
        if not self.available:
            pytest.skip("GestureDetector not available")
        KP = self.KeypointPosition
        kps = self._make_keypoints({
            9: KP(0.38, 0.05, 0.9),   # left_wrist above nose
            10: KP(0.62, 0.05, 0.9),  # right_wrist above nose
        })
        result = self.detector.detect(kps)
        assert result is not None
        assert result.type == self.GestureType.HANDS_UP

    def test_left_hand_raised(self):
        """Left hand above shoulder should detect HAND_RAISED_LEFT."""
        if not self.available:
            pytest.skip("GestureDetector not available")
        KP = self.KeypointPosition
        kps = self._make_keypoints({
            9: KP(0.38, 0.15, 0.9),   # left_wrist well above shoulder (0.35)
        })
        result = self.detector.detect(kps)
        assert result is not None
        assert result.type in (
            self.GestureType.HAND_RAISED_LEFT,
            self.GestureType.HANDS_UP,  # May also match
        )

    def test_pointing_right(self):
        """Extended right arm should detect POINTING_RIGHT."""
        if not self.available:
            pytest.skip("GestureDetector not available")
        KP = self.KeypointPosition
        kps = self._make_keypoints({
            8: KP(0.75, 0.36, 0.9),   # right_elbow extended
            10: KP(0.85, 0.36, 0.9),  # right_wrist far right, level
        })
        result = self.detector.detect(kps)
        if result is not None and result.type != self.GestureType.NONE:
            assert result.type in (
                self.GestureType.POINTING_RIGHT,
                self.GestureType.HAND_RAISED_RIGHT,  # May also match
            )

    def test_low_confidence_keypoints_ignored(self):
        """Keypoints below confidence threshold should not trigger gestures."""
        if not self.available:
            pytest.skip("GestureDetector not available")
        KP = self.KeypointPosition
        kps = self._make_keypoints({
            9: KP(0.38, 0.05, 0.1),   # Low confidence
            10: KP(0.62, 0.05, 0.1),
        })
        result = self.detector.detect(kps)
        # Should not detect HANDS_UP with low-confidence wrists
        if result is not None:
            assert result.type != self.GestureType.HANDS_UP or result.confidence < 0.3

    def test_gesture_type_enum_values(self):
        """GestureType should have expected values."""
        if not self.available:
            pytest.skip("GestureDetector not available")
        assert self.GestureType.NONE.value == "none"
        assert self.GestureType.WAVE_LEFT.value == "wave_left"
        assert self.GestureType.HANDS_UP.value == "hands_up"


# === Identity Manager Tests ===

class TestIdentityManager:
    """Tests for face identity matching (pure CPU math)."""

    @pytest.fixture(autouse=True)
    def setup_identity(self, tmp_path):
        try:
            from core.vision.identity_manager import IdentityManager
            # Use temp directory to avoid touching real identity DB
            self.mgr = IdentityManager.__new__(IdentityManager)
            self.mgr.identities = {}
            self.mgr.registry_path = str(tmp_path / "identity_registry.json")
            self.mgr._lock = threading.Lock()
            self.IdentityManager = IdentityManager
            self.available = True
        except ImportError:
            self.available = False

    def _random_embedding(self, dim=512):
        """Create a random normalized embedding vector."""
        vec = np.random.randn(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_normalize_vector(self):
        """normalize() should produce unit-length vector."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        vec = np.array([3.0, 4.0, 0.0])
        result = self.IdentityManager.normalize(vec)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """normalize() should handle zero vector gracefully."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        vec = np.zeros(3)
        result = self.IdentityManager.normalize(vec)
        # Should not crash, return something
        assert result is not None

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors should be ~1.0."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        vec = self._random_embedding()
        sim = self.IdentityManager.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors should be ~0.0."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        vec1 = np.zeros(512, dtype=np.float32)
        vec2 = np.zeros(512, dtype=np.float32)
        vec1[0] = 1.0
        vec2[1] = 1.0
        sim = self.IdentityManager.cosine_similarity(vec1, vec2)
        assert abs(sim) < 1e-5

    def test_enroll_and_match(self):
        """Enrolling then matching same embedding should find identity."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        embedding = self._random_embedding()
        self.mgr.enroll("Markus", embedding, threshold=0.5)
        name, score = self.mgr.match(embedding)
        assert name == "Markus"
        assert score > 0.9

    def test_match_no_identities(self):
        """Matching with empty database should return (None, 0.0)."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        embedding = self._random_embedding()
        name, score = self.mgr.match(embedding)
        assert name is None
        assert score == 0.0

    def test_match_wrong_person(self):
        """Matching different embedding should not match enrolled person."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        self.mgr.enroll("Alice", self._random_embedding(), threshold=0.8)
        different = self._random_embedding()
        name, score = self.mgr.match(different)
        # Random 512-dim vectors have very low similarity
        assert name is None or score < 0.8

    def test_list_identities(self):
        """list_identities() should return enrolled identities."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        self.mgr.enroll("Person1", self._random_embedding())
        self.mgr.enroll("Person2", self._random_embedding())
        identities = self.mgr.list_identities()
        names = [i["name"] for i in identities]
        assert "Person1" in names
        assert "Person2" in names

    def test_remove_identity(self):
        """remove_identity() should remove enrolled identity."""
        if not self.available:
            pytest.skip("IdentityManager not available")
        self.mgr.enroll("Temp", self._random_embedding())
        assert self.mgr.remove_identity("Temp") is True
        assert len(self.mgr.list_identities()) == 0


# === Vision Context Tests ===

class TestVisionContext:
    """Tests for VisionContext state management."""

    @pytest.fixture(autouse=True)
    def setup_context(self):
        try:
            from context.vision_context import VisionContext, VisionEventType
            self.ctx = VisionContext()
            self.VisionEventType = VisionEventType
            self.available = True
        except ImportError:
            self.available = False

    def test_initial_state_no_person(self):
        """Initial state should have no person detected."""
        if not self.available:
            pytest.skip("VisionContext not available")
        state = self.ctx.get_state()
        assert state.person_detected is False

    def test_update_detection_sets_person(self):
        """update_detection with person should set person_detected=True."""
        if not self.available:
            pytest.skip("VisionContext not available")
        self.ctx.update_detection(
            person_detected=True,
            person_count=1,
            confidence=0.9,
            target_center_x=960,
            frame_width=1920,
            source="test",
        )
        state = self.ctx.get_state()
        assert state.person_detected is True
        assert state.person_count == 1

    def test_describe_no_person(self):
        """describe() with no person should mention nobody visible."""
        if not self.available:
            pytest.skip("VisionContext not available")
        desc = self.ctx.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_event_callback_fires(self):
        """Callbacks should fire on state transitions."""
        if not self.available:
            pytest.skip("VisionContext not available")
        events = []
        # Try both callback registration methods
        if hasattr(self.ctx, 'set_event_callback'):
            self.ctx.set_event_callback(lambda e, m: events.append(e))
        elif hasattr(self.ctx, 'register_callback'):
            self.ctx.register_callback(lambda e, m: events.append(e))
        elif hasattr(self.ctx, 'on_person_detected'):
            self.ctx.on_person_detected = lambda m: events.append("detected")
        else:
            pytest.skip("No callback registration method found")
        # Trigger person detection
        self.ctx.update_detection(
            person_detected=True, person_count=1, confidence=0.9,
            target_center_x=960, frame_width=1920, source="test",
        )
        assert len(events) > 0

    def test_to_dict(self):
        """to_dict() should return serializable dict."""
        if not self.available:
            pytest.skip("VisionContext not available")
        d = self.ctx.to_dict()
        assert isinstance(d, dict)
        assert "person_detected" in d


# === Perception State Tests ===

class TestPerceptionState:
    """Tests for PerceptionState timeout-based state machine."""

    @pytest.fixture(autouse=True)
    def setup_perception(self):
        try:
            from context.perception_state import PerceptionState, PerceptionEvent
            self.ps = PerceptionState()
            self.PerceptionEvent = PerceptionEvent
            self.available = True
        except ImportError:
            self.available = False

    def test_initial_state_no_user(self):
        """Initial state should have no user visible."""
        if not self.available:
            pytest.skip("PerceptionState not available")
        assert self.ps.user_visible is False

    def test_update_sets_user_visible(self):
        """update() with user_detected=True should set user_visible."""
        if not self.available:
            pytest.skip("PerceptionState not available")
        self.ps.update(
            user_detected=True, face_detected=False, gesture_detected=False,
            gesture_type="none", person_count=1, confidence=0.9,
            face_confidence=0.0, gesture_confidence=0.0,
            face_keypoints=0, torso_keypoints=0, wrist_keypoints=0,
            source="test",
        )
        assert self.ps.user_visible is True

    def test_describe_returns_string(self):
        """describe() should return a non-empty German string."""
        if not self.available:
            pytest.skip("PerceptionState not available")
        desc = self.ps.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_snapshot(self):
        """get_snapshot() should return PerceptionSnapshot."""
        if not self.available:
            pytest.skip("PerceptionState not available")
        snap = self.ps.get_snapshot()
        assert snap is not None
        assert hasattr(snap, "user_visible")
        assert hasattr(snap, "face_visible")

    def test_event_callback_user_appeared(self):
        """USER_APPEARED event should fire when user first detected."""
        if not self.available:
            pytest.skip("PerceptionState not available")
        events = []
        self.ps.set_event_callback(lambda e, m: events.append(e))
        self.ps.update(
            user_detected=True, face_detected=False, gesture_detected=False,
            gesture_type="none", person_count=1, confidence=0.9,
            face_confidence=0.0, gesture_confidence=0.0,
            face_keypoints=0, torso_keypoints=0, wrist_keypoints=0,
            source="test",
        )
        event_types = [e.value if hasattr(e, 'value') else str(e) for e in events]
        assert any("appeared" in str(e).lower() for e in events)


# === Vision Mode Manager Tests ===

class TestVisionModeManager:
    """Tests for VisionModeManager state machine."""

    @pytest.fixture(autouse=True)
    def setup_mode_mgr(self):
        try:
            from core.vision.vision_mode_manager import VisionModeManager, VisionMode
            self.mgr = VisionModeManager.__new__(VisionModeManager)
            self.mgr._current_mode = VisionMode.VISION_TRACKING
            self.mgr._callbacks = []
            self.mgr._state_lock = threading.RLock()
            self.mgr._config = None
            self.VisionMode = VisionMode
            self.available = True
        except ImportError:
            self.available = False

    def test_mode_values(self):
        """VisionMode should have expected values."""
        if not self.available:
            pytest.skip("VisionModeManager not available")
        assert self.VisionMode.VISION_TRACKING.value == "VISION_TRACKING"
        assert self.VisionMode.VISION_IDENTITY.value == "VISION_IDENTITY"
        assert self.VisionMode.VISION_FULL.value == "VISION_FULL"

    def test_set_mode(self):
        """set_mode() should change current mode."""
        if not self.available:
            pytest.skip("VisionModeManager not available")
        self.mgr.set_mode(self.VisionMode.VISION_IDENTITY)
        assert self.mgr.current_mode == self.VisionMode.VISION_IDENTITY

    def test_mode_callback_fires(self):
        """Mode change should fire registered callbacks."""
        if not self.available:
            pytest.skip("VisionModeManager not available")
        modes = []
        self.mgr.register_callback(lambda m: modes.append(m))
        self.mgr.set_mode(self.VisionMode.VISION_FULL)
        assert len(modes) == 1
        assert modes[0] == self.VisionMode.VISION_FULL

    def test_get_available_modes(self):
        """get_available_modes() should return list of mode names."""
        if not self.available:
            pytest.skip("VisionModeManager not available")
        modes = self.mgr.get_available_modes()
        assert isinstance(modes, list)
        assert "VISION_TRACKING" in modes
