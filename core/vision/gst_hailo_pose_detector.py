#!/usr/bin/env python3
"""
M.O.L.O.C.H. GStreamer Hailo Pose Detector
==========================================

Person detection using YOLOv8 Pose model on Hailo-10H NPU.
Uses keypoints to differentiate real persons from hands/partial bodies.

Keypoint Validation:
- Real person: Must have head (nose/eyes) AND torso (shoulders) keypoints
- Hand only: Typically only has wrist keypoints visible

COCO 17 Keypoints:
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Author: M.O.L.O.C.H. System
Date: 2026-02-04
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

import numpy as np
import threading
import time
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Callable, Tuple
from enum import Enum

# Gesture detection
from core.vision.gesture_detector import (
    get_gesture_detector, GestureDetector, Gesture, GestureType,
    KeypointPosition
)

logger = logging.getLogger(__name__)

# Debug flags (shared with gst_hailo_detector)
DEBUG_FRAME_LOGGING = True
DEBUG_FRAME_INTERVAL = 30
DEBUG_WATCHDOG_TIMEOUT = 5.0  # Trigger restart after 5s of no frames

# Hailo model directory (secondary SSD)
HAILO_MODELS_DIR = "/mnt/moloch-data/hailo/models"

# Initialize GStreamer
Gst.init(None)

# Import Hailo resource manager
try:
    from core.hardware.hailo_manager import (
        get_hailo_manager, HailoConsumer, force_hailo_reset, is_hailo_device_free
    )
    HAILO_MANAGER_AVAILABLE = True
except ImportError:
    HAILO_MANAGER_AVAILABLE = False
    force_hailo_reset = lambda: False
    is_hailo_device_free = lambda: True
    logger.warning("HailoManager not available - running without resource management")

# Import Hailo for detection parsing
try:
    import hailo
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo module not available")


class KeypointIndex(Enum):
    """COCO Keypoint indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Keypoint groups for validation
FACE_KEYPOINTS = [KeypointIndex.NOSE, KeypointIndex.LEFT_EYE, KeypointIndex.RIGHT_EYE]
HEAD_KEYPOINTS = [KeypointIndex.NOSE, KeypointIndex.LEFT_EYE, KeypointIndex.RIGHT_EYE,
                  KeypointIndex.LEFT_EAR, KeypointIndex.RIGHT_EAR]
TORSO_KEYPOINTS = [KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER]
UPPER_BODY_KEYPOINTS = HEAD_KEYPOINTS + TORSO_KEYPOINTS
HAND_KEYPOINTS = [KeypointIndex.LEFT_WRIST, KeypointIndex.RIGHT_WRIST]

# Face validation: Need at least these for "real face" (lowered for better detection)
MIN_FACE_KEYPOINTS = 1  # At least 1 of (nose, left_eye, right_eye) visible
MIN_HEAD_KEYPOINTS = 2  # At least 2 of all head keypoints
MIN_TORSO_KEYPOINTS = 1  # At least 1 shoulder visible


@dataclass
class Keypoint:
    """Single keypoint with position and confidence."""
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    confidence: float
    visible: bool = False

    def __post_init__(self):
        # Lower threshold (0.15) for better hand/wrist detection
        self.visible = self.confidence > 0.15


@dataclass
class PersonPose:
    """Person detection with pose keypoints."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    keypoints: List[Keypoint]

    # Validation flags
    has_face: bool = False      # Face keypoints visible (nose + eyes)
    has_head: bool = False      # Any head keypoints visible
    has_torso: bool = False     # Shoulder keypoints visible
    is_valid_person: bool = False
    validation_reason: str = ""

    # Face analysis
    face_confidence: float = 0.0  # Average confidence of face keypoints
    face_center: Optional[Tuple[float, float]] = None  # Normalized center of face

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate if this is a real person based on keypoints."""
        if not self.keypoints or len(self.keypoints) < 17:
            self.validation_reason = "no_keypoints"
            return

        # === FACE VALIDATION (nose + eyes) ===
        face_kps = [self.keypoints[i.value] for i in FACE_KEYPOINTS]
        face_visible = sum(1 for kp in face_kps if kp.visible)
        self.has_face = face_visible >= MIN_FACE_KEYPOINTS

        # Calculate face confidence (average of visible face keypoints)
        visible_face_kps = [kp for kp in face_kps if kp.visible]
        if visible_face_kps:
            self.face_confidence = sum(kp.confidence for kp in visible_face_kps) / len(visible_face_kps)
            # Calculate face center
            self.face_center = (
                sum(kp.x for kp in visible_face_kps) / len(visible_face_kps),
                sum(kp.y for kp in visible_face_kps) / len(visible_face_kps)
            )

        # === HEAD VALIDATION (includes ears) ===
        head_visible = sum(1 for i in HEAD_KEYPOINTS if self.keypoints[i.value].visible)
        self.has_head = head_visible >= MIN_HEAD_KEYPOINTS

        # === TORSO VALIDATION (shoulders) ===
        # Require BOTH shoulders visible for valid person (stricter filtering)
        torso_visible = sum(1 for i in TORSO_KEYPOINTS if self.keypoints[i.value].visible)
        self.has_torso = torso_visible >= MIN_TORSO_KEYPOINTS

        # === HAND-ONLY CHECK ===
        wrist_visible = sum(1 for i in HAND_KEYPOINTS if self.keypoints[i.value].visible)
        upper_body_visible = sum(1 for i in UPPER_BODY_KEYPOINTS if self.keypoints[i.value].visible)

        # === FINAL VALIDATION ===
        # Valid person needs: face (nose+eyes) AND torso (shoulders)
        # This ensures we're not tracking just a hand or partial body
        self.is_valid_person = self.has_face and self.has_torso

        # Set validation reason with detailed keypoint counts
        if wrist_visible > 0 and upper_body_visible == 0:
            self.validation_reason = "hand_only"
        elif not self.has_face:
            self.validation_reason = f"no_face({face_visible}/{MIN_FACE_KEYPOINTS})"
        elif not self.has_torso:
            self.validation_reason = f"no_torso({torso_visible}/{MIN_TORSO_KEYPOINTS})"
        elif not self.has_head:
            self.validation_reason = f"no_head({head_visible}/{MIN_HEAD_KEYPOINTS})"
        else:
            self.validation_reason = f"valid(face={self.face_confidence:.0%},torso={torso_visible})"

    def get_head_center(self) -> Optional[Tuple[float, float]]:
        """Get center of head (average of visible head keypoints)."""
        visible_head = [self.keypoints[i.value] for i in HEAD_KEYPOINTS
                       if self.keypoints[i.value].visible]
        if not visible_head:
            return None
        x = sum(kp.x for kp in visible_head) / len(visible_head)
        y = sum(kp.y for kp in visible_head) / len(visible_head)
        return (x, y)

    def get_face_center(self) -> Optional[Tuple[float, float]]:
        """Get center of face (nose + eyes only)."""
        return self.face_center

    def get_eye_positions(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get left and right eye positions."""
        left_eye = self.keypoints[KeypointIndex.LEFT_EYE.value] if self.keypoints else None
        right_eye = self.keypoints[KeypointIndex.RIGHT_EYE.value] if self.keypoints else None

        left_pos = (left_eye.x, left_eye.y) if left_eye and left_eye.visible else None
        right_pos = (right_eye.x, right_eye.y) if right_eye and right_eye.visible else None

        return (left_pos, right_pos)

    def get_face_keypoints_detail(self) -> Dict:
        """Get detailed face keypoint information for mimics analysis."""
        if not self.keypoints:
            return {}

        return {
            "nose": {
                "x": self.keypoints[KeypointIndex.NOSE.value].x,
                "y": self.keypoints[KeypointIndex.NOSE.value].y,
                "conf": self.keypoints[KeypointIndex.NOSE.value].confidence,
                "visible": self.keypoints[KeypointIndex.NOSE.value].visible
            },
            "left_eye": {
                "x": self.keypoints[KeypointIndex.LEFT_EYE.value].x,
                "y": self.keypoints[KeypointIndex.LEFT_EYE.value].y,
                "conf": self.keypoints[KeypointIndex.LEFT_EYE.value].confidence,
                "visible": self.keypoints[KeypointIndex.LEFT_EYE.value].visible
            },
            "right_eye": {
                "x": self.keypoints[KeypointIndex.RIGHT_EYE.value].x,
                "y": self.keypoints[KeypointIndex.RIGHT_EYE.value].y,
                "conf": self.keypoints[KeypointIndex.RIGHT_EYE.value].confidence,
                "visible": self.keypoints[KeypointIndex.RIGHT_EYE.value].visible
            },
            "left_ear": {
                "x": self.keypoints[KeypointIndex.LEFT_EAR.value].x,
                "y": self.keypoints[KeypointIndex.LEFT_EAR.value].y,
                "conf": self.keypoints[KeypointIndex.LEFT_EAR.value].confidence,
                "visible": self.keypoints[KeypointIndex.LEFT_EAR.value].visible
            },
            "right_ear": {
                "x": self.keypoints[KeypointIndex.RIGHT_EAR.value].x,
                "y": self.keypoints[KeypointIndex.RIGHT_EAR.value].y,
                "conf": self.keypoints[KeypointIndex.RIGHT_EAR.value].confidence,
                "visible": self.keypoints[KeypointIndex.RIGHT_EAR.value].visible
            }
        }

    def get_keypoint_counts(self) -> Dict:
        """Get counts of visible keypoints per category."""
        if not self.keypoints:
            return {"face": 0, "head": 0, "torso": 0, "wrist": 0, "total": 0}
        return {
            "face": sum(1 for i in FACE_KEYPOINTS if self.keypoints[i.value].visible),
            "head": sum(1 for i in HEAD_KEYPOINTS if self.keypoints[i.value].visible),
            "torso": sum(1 for i in TORSO_KEYPOINTS if self.keypoints[i.value].visible),
            "wrist": sum(1 for i in HAND_KEYPOINTS if self.keypoints[i.value].visible),
            "total": sum(1 for kp in self.keypoints if kp.visible)
        }

    def to_dict(self) -> Dict:
        """Convert to dict for tracking."""
        kp_counts = self.get_keypoint_counts()
        return {
            "class": "person",
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "center_x": (self.bbox[0] + self.bbox[2]) / 2,
            "center_y": (self.bbox[1] + self.bbox[3]) / 2,
            # Face validation
            "has_face": self.has_face,
            "face_confidence": self.face_confidence,
            "face_center": self.face_center,
            # Body validation
            "has_head": self.has_head,
            "has_torso": self.has_torso,
            # Final result
            "is_valid_person": self.is_valid_person,
            "validation_reason": self.validation_reason,
            # Keypoint counts for debugging
            "keypoint_counts": kp_counts,
            # Keypoints for advanced tracking
            "keypoints": [{"x": kp.x, "y": kp.y, "conf": kp.confidence}
                         for kp in self.keypoints] if self.keypoints else [],
            "face_keypoints": self.get_face_keypoints_detail()
        }


@dataclass
class PoseDetectionResult:
    """Frame pose detection result."""
    poses: List[PersonPose] = field(default_factory=list)
    valid_persons: List[PersonPose] = field(default_factory=list)  # Filtered valid persons
    frame_width: int = 640
    frame_height: int = 640
    fps: float = 0
    timestamp: float = field(default_factory=time.time)
    error: str = ""
    frame: Optional[Any] = None  # BGR numpy array for display
    model_name: str = "yolov8s_pose_h10"

    # Gesture detection
    gesture: Optional[Gesture] = None

    # Stats
    total_detections: int = 0
    faces_detected: int = 0
    hands_rejected: int = 0

    def __post_init__(self):
        """Calculate stats after init."""
        self.total_detections = len(self.poses)
        self.faces_detected = sum(1 for p in self.poses if p.has_face)
        self.hands_rejected = sum(1 for p in self.poses if p.validation_reason == "hand_only")


class GstHailoPoseDetector:
    """
    GStreamer-based pose detector using Hailo-10H.

    Uses YOLOv8 Pose model for keypoint-based person validation.
    Only detections with visible head AND torso are considered valid persons.
    """

    # Model paths - use smaller pose model for speed (secondary SSD)
    HEF_PATH = f"{HAILO_MODELS_DIR}/yolov8s_pose_h10.hef"
    # Use compiled postprocess from hailo-apps repo (works correctly)
    POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolov8pose_postprocess.so"

    # Detection settings (lower threshold for better detection)
    CONFIDENCE_THRESHOLD = 0.25  # Lower for better detection
    NMS_THRESHOLD = 0.45
    KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

    def __init__(self, rtsp_url: str = None):
        """Initialize pose detector."""
        self.rtsp_url = rtsp_url or "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"

        self._pipeline = None
        self._bus = None
        self._running = False
        self._main_loop = None
        self._loop_thread = None

        # MUTEX: Protects start/stop operations - prevents concurrent access
        self._operation_lock = threading.RLock()
        self._restarting = False  # Prevent restart loops

        # Results
        self._latest_result: PoseDetectionResult = PoseDetectionResult()
        self._result_lock = threading.Lock()

        # Gesture detection
        self._gesture_detector: GestureDetector = get_gesture_detector()
        self._current_gesture: Optional[Gesture] = None
        self._gesture_callback: Optional[Callable[[Gesture], None]] = None

        # Callbacks
        self._on_detection: Optional[Callable[[PoseDetectionResult], None]] = None

        # Stats
        self._frame_count = 0
        self._start_time = 0
        self._last_fps_time = 0
        self._fps_frame_count = 0
        self._last_frame_time = 0  # Watchdog
        self._watchdog_warned = False

        # Error recovery
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3

        logger.info(f"GstHailoPoseDetector initialized (model: {self.HEF_PATH})")

    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline for pose detection - 30 FPS low latency."""
        # ULTRA LOW LATENCY:
        # - rtspsrc latency=0 (no buffering)
        # - All queues: leaky=downstream, max-size-buffers=1
        # - appsink: drop=true, sync=false, max-buffers=1
        pipeline = f"""
            rtspsrc location={self.rtsp_url} latency=0 buffer-mode=auto !
            rtph264depay !
            h264parse !
            avdec_h264 !
            videoconvert !
            videoscale !
            video/x-raw,width=640,height=640,format=RGB !
            queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 !
            hailonet hef-path={self.HEF_PATH} batch-size=1 !
            queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 !
            hailofilter so-path={self.POSTPROCESS_SO} function-name=filter_letterbox qos=false !
            queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 !
            videoconvert !
            appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false
        """
        return " ".join(pipeline.split())

    def start(self, skip_npu_check: bool = False) -> bool:
        """Start the pose detection pipeline.

        Args:
            skip_npu_check: If True, skip the NPU availability check.
                           Use when restarting after release to avoid race condition.
        """
        # MUTEX: Only one start/stop operation at a time
        with self._operation_lock:
            if self._running:
                logger.warning("Pose detector already running")
                return True

            # === CLEANUP: Make sure old pipeline is fully stopped ===
            if self._pipeline is not None:
                logger.warning("[DEBUG] Old pipeline still exists, cleaning up...")
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                    self._pipeline = None
                except:
                    pass
            if self._main_loop is not None:
                try:
                    if self._main_loop.is_running():
                        self._main_loop.quit()
                    self._main_loop = None
                except:
                    pass

            # === DEBUG: Log startup parameters ===
            logger.info("=" * 60)
            logger.info(f"[DEBUG] GstHailoPoseDetector START (skip_npu_check={skip_npu_check})")
            logger.info(f"[DEBUG] RTSP URL: {self.rtsp_url}")
            logger.info(f"[DEBUG] HEF Path: {self.HEF_PATH}")
            logger.info(f"[DEBUG] PostProcess SO: {self.POSTPROCESS_SO}")

            # === DEBUG: Verify files exist ===
            if os.path.exists(self.HEF_PATH):
                hef_size = os.path.getsize(self.HEF_PATH) / (1024 * 1024)
                logger.info(f"[DEBUG] HEF exists: {hef_size:.1f} MB")
            else:
                logger.error(f"[DEBUG] HEF NOT FOUND: {self.HEF_PATH}")
                return False

            if os.path.exists(self.POSTPROCESS_SO):
                logger.info(f"[DEBUG] PostProcess SO exists: OK")
            else:
                logger.error(f"[DEBUG] PostProcess SO NOT FOUND: {self.POSTPROCESS_SO}")
                return False

            try:
                # === HAILO MANAGER: Acquire NPU for vision ===
                # skip_npu_check=True means manager already acquired it (restart scenario)
                if skip_npu_check:
                    logger.info("[DEBUG] Skipping NPU acquisition (already acquired by manager)")
                elif HAILO_MANAGER_AVAILABLE:
                    manager = get_hailo_manager()
                    if not manager.acquire_for_vision(timeout=5.0):
                        logger.error("[DEBUG] Failed to acquire Hailo NPU via manager")
                        self._latest_result = PoseDetectionResult(error="Hailo NPU busy (manager)")
                        return False
                    logger.info("[DEBUG] Hailo NPU acquired via manager")
                else:
                    # Fallback to old check if manager not available
                    from core.vision.gst_hailo_detector import check_hailo_device_free
                    if not check_hailo_device_free():
                        logger.warning("[DEBUG] Hailo NPU not available")
                        self._latest_result = PoseDetectionResult(error="Hailo NPU busy")
                        return False
                    logger.info("[DEBUG] Hailo NPU: available (legacy check)")

                pipeline_str = self._build_pipeline()
                logger.info(f"[DEBUG] Creating Hailo POSE pipeline...")

                self._pipeline = Gst.parse_launch(pipeline_str)
                if not self._pipeline:
                    logger.error("Failed to create pipeline")
                    return False

                # Get appsink and connect callback
                appsink = self._pipeline.get_by_name("appsink")
                if appsink:
                    handler_id = appsink.connect("new-sample", self._on_new_sample)
                    logger.info(f"[DEBUG] Appsink connected: handler_id={handler_id}")
                    logger.info(f"[DEBUG] Detection callback registered: {self._on_detection is not None}")
                else:
                    logger.error("Appsink not found")
                    return False

                # Setup bus
                self._bus = self._pipeline.get_bus()
                self._bus.add_signal_watch()
                self._bus.connect("message", self._on_bus_message)

                # Start pipeline with state transition logging
                logger.info("[DEBUG] Pipeline state: NULL -> PLAYING")
                ret = self._pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.error("[DEBUG] Pipeline state change FAILED")
                    return False
                elif ret == Gst.StateChangeReturn.ASYNC:
                    logger.info("[DEBUG] Pipeline state change: ASYNC (waiting for stream)")
                elif ret == Gst.StateChangeReturn.SUCCESS:
                    logger.info("[DEBUG] Pipeline state change: SUCCESS")
                else:
                    logger.info(f"[DEBUG] Pipeline state change: {ret}")

                self._running = True
                self._consecutive_errors = 0  # Reset error counter on success
                self._start_time = time.time()
                self._last_fps_time = self._start_time
                self._last_frame_time = self._start_time  # Initialize watchdog

                # Start main loop in thread
                self._main_loop = GLib.MainLoop()
                self._loop_thread = threading.Thread(target=self._run_main_loop, daemon=True)
                self._loop_thread.start()

                logger.info("[DEBUG] GstHailoPoseDetector started successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to start pose detector: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False

    def _run_main_loop(self):
        """Run GLib main loop."""
        try:
            # Add watchdog timer
            GLib.timeout_add(1000, self._watchdog_check)
            self._main_loop.run()
        except Exception as e:
            logger.error(f"Main loop error: {e}")

    def _watchdog_check(self) -> bool:
        """Check if frames are being received - trigger restart if stalled."""
        if not self._running:
            return False

        if self._last_frame_time > 0:
            elapsed = time.time() - self._last_frame_time
            if elapsed > DEBUG_WATCHDOG_TIMEOUT:
                if not self._watchdog_warned:
                    logger.warning(f"[WATCHDOG] No frame for {elapsed:.1f}s! (count={self._frame_count})")
                    self._watchdog_warned = True

                # Trigger restart after watchdog timeout
                if elapsed > DEBUG_WATCHDOG_TIMEOUT + 1.0 and not self._restarting:
                    logger.error(f"[WATCHDOG] RESTARTING - no frames for {elapsed:.1f}s")
                    self._consecutive_errors += 1
                    if self._consecutive_errors <= self._max_consecutive_errors:
                        restart_thread = threading.Thread(target=self._delayed_restart, daemon=True)
                        restart_thread.start()
                        return False  # Stop watchdog timer
                    else:
                        logger.error(f"[WATCHDOG] Too many restarts ({self._consecutive_errors}), giving up")

        return self._running

    def _parse_pose_detections(self, buffer) -> List[PersonPose]:
        """Parse pose detections from Hailo buffer."""
        poses = []

        if not HAILO_AVAILABLE:
            return poses

        try:
            roi = hailo.get_roi_from_buffer(buffer)
            if not roi:
                return poses

            # Get pose detections (HailoUserMeta with keypoints)
            hailo_dets = hailo.get_hailo_detections(roi)

            for det in hailo_dets:
                label = det.get_label()
                conf = det.get_confidence()
                bbox = det.get_bbox()

                if label != "person" or conf < self.CONFIDENCE_THRESHOLD:
                    continue

                # Get bbox in pixels
                x1 = bbox.xmin() * 640
                y1 = bbox.ymin() * 640
                x2 = (bbox.xmin() + bbox.width()) * 640
                y2 = (bbox.ymin() + bbox.height()) * 640

                # Get keypoints - coordinates are RELATIVE to bbox, need transformation
                keypoints = []
                try:
                    # Get landmarks from detection
                    landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
                    if self._frame_count < 5:
                        logger.info(f"[DEBUG] Detection has {len(landmarks) if landmarks else 0} landmark objects")
                    if landmarks:
                        lm = landmarks[0]
                        points = lm.get_points()
                        if self._frame_count < 5:
                            logger.info(f"[DEBUG] Landmark has {len(points)} points")

                        # Get bbox dimensions for coordinate transformation
                        bbox_w = bbox.width()
                        bbox_h = bbox.height()
                        bbox_x = bbox.xmin()
                        bbox_y = bbox.ymin()

                        for i, pt in enumerate(points):
                            # Transform relative coords to absolute (0-1 normalized)
                            # Keypoint coords are relative to bbox
                            abs_x = pt.x() * bbox_w + bbox_x
                            abs_y = pt.y() * bbox_h + bbox_y

                            # Get confidence - try multiple methods
                            kp_conf = 0.5  # default
                            if hasattr(pt, 'confidence'):
                                kp_conf = pt.confidence()
                            elif hasattr(pt, 'score'):
                                kp_conf = pt.score()

                            # If confidence is 0, check if point is within valid range
                            if kp_conf == 0 and 0 <= abs_x <= 1 and 0 <= abs_y <= 1:
                                kp_conf = 0.5  # Assume visible if in valid range

                            if self._frame_count < 5 and i < 5:
                                logger.info(f"[DEBUG] pt[{i}]: raw=({pt.x():.3f},{pt.y():.3f}) abs=({abs_x:.3f},{abs_y:.3f}) conf={kp_conf:.3f}")

                            kp = Keypoint(
                                x=abs_x,
                                y=abs_y,
                                confidence=kp_conf
                            )
                            keypoints.append(kp)
                    else:
                        if self._frame_count < 5:
                            logger.warning(f"[DEBUG] No landmarks in detection!")
                except Exception as e:
                    logger.warning(f"Keypoint parse error: {e}")

                # Pad keypoints to 17 if needed
                while len(keypoints) < 17:
                    keypoints.append(Keypoint(x=0, y=0, confidence=0))

                pose = PersonPose(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    keypoints=keypoints
                )
                poses.append(pose)

        except Exception as e:
            if self._frame_count == 0:
                logger.error(f"Pose detection parse error: {e}")
                import traceback
                logger.error(traceback.format_exc())

        return poses

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """Handle new frame from appsink."""
        try:
            # === CRITICAL: Log EVERY 10 frames to confirm signal fires ===
            if self._frame_count % 10 == 0:
                logger.info(f"[HAILO] FRAME RECEIVED #{self._frame_count}")

            sample = appsink.emit("pull-sample")
            if sample is None:
                logger.warning("[APPSINK] pull-sample returned None!")
                return Gst.FlowReturn.OK

            buffer = sample.get_buffer()
            if buffer is None:
                logger.warning("[APPSINK] buffer is None!")
                return Gst.FlowReturn.OK

            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")

            # Log buffer info for first frames
            if self._frame_count < 3:
                buf_size = buffer.get_size()
                logger.info(f"[APPSINK] buffer size={buf_size}, {width}x{height}")

            # Parse pose detections
            poses = self._parse_pose_detections(buffer)

            # Filter to valid persons only
            valid_persons = [p for p in poses if p.is_valid_person]

            # Log validation results periodically (every 30 frames = ~2 seconds)
            if self._frame_count % 30 == 0 and poses:
                logger.info(f"[POSE] Frame {self._frame_count}: {len(poses)} detections, {len(valid_persons)} valid")
                for i, p in enumerate(poses):
                    # Count keypoints per category
                    face_kps = sum(1 for j in FACE_KEYPOINTS if p.keypoints[j.value].visible)
                    head_kps = sum(1 for j in HEAD_KEYPOINTS if p.keypoints[j.value].visible)
                    torso_kps = sum(1 for j in TORSO_KEYPOINTS if p.keypoints[j.value].visible)
                    wrist_kps = sum(1 for j in HAND_KEYPOINTS if p.keypoints[j.value].visible)
                    total_kps = sum(1 for kp in p.keypoints if kp.visible)

                    status = "VALID" if p.is_valid_person else "REJECT"
                    logger.info(f"  [{i}] {status} conf={p.confidence:.2f} "
                               f"face={face_kps}/3 head={head_kps}/5 torso={torso_kps}/2 "
                               f"wrist={wrist_kps} total={total_kps}/17 => {p.validation_reason}")

            # Extract frame for display
            frame_bgr = None
            try:
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    frame_rgb = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    ).copy()
                    buffer.unmap(map_info)

                    import cv2
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                if self._frame_count == 0:
                    logger.debug(f"Frame extraction error: {e}")

            # Update FPS and watchdog
            self._frame_count += 1
            self._fps_frame_count += 1
            now = time.time()
            self._last_frame_time = now  # Update watchdog
            self._watchdog_warned = False

            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                fps = self._fps_frame_count / elapsed
                # === CRITICAL: Log FPS every second ===
                logger.info(f"LIVE FPS: {fps:.1f} (frame {self._frame_count})")
                self._fps_frame_count = 0
                self._last_fps_time = now
            else:
                fps = self._fps_frame_count / elapsed if elapsed > 0 else 0

            # === GESTURE DETECTION ===
            gesture = None
            if valid_persons and self._gesture_detector:
                # Use first valid person's keypoints for gesture detection
                person = valid_persons[0]
                # Convert to KeypointPosition format for gesture detector
                kp_positions = [
                    KeypointPosition(
                        x=kp.x,
                        y=kp.y,
                        confidence=kp.confidence
                    ) for kp in person.keypoints
                ]
                gesture = self._gesture_detector.detect(kp_positions)
                if gesture:
                    self._current_gesture = gesture
                    # Log new gestures
                    if gesture.duration_ms == 0:
                        logger.info(f"[GESTURE] {gesture.type.value} detected (conf={gesture.confidence:.2f})")
                    # Callback for gesture
                    if self._gesture_callback:
                        try:
                            self._gesture_callback(gesture)
                        except Exception as e:
                            logger.error(f"Gesture callback error: {e}")

            # Create result
            result = PoseDetectionResult(
                poses=poses,
                valid_persons=valid_persons,
                frame_width=width,
                frame_height=height,
                fps=fps,
                timestamp=now,
                frame=frame_bgr
            )
            # Attach gesture to result
            result.gesture = gesture

            # Store result
            with self._result_lock:
                self._latest_result = result

            # Callback
            if self._on_detection:
                try:
                    self._on_detection(result)
                except Exception as e:
                    logger.error(f"Detection callback error: {e}")

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"Sample callback error: {e}")
            return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages with debug logging."""
        t = message.type
        src = message.src.get_name() if message.src else "unknown"

        if t == Gst.MessageType.EOS:
            logger.info(f"[DEBUG] POSE GStreamer EOS from {src}")
            self.stop()

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            err_msg = err.message if err else "unknown"
            debug_str = debug or ""
            logger.error(f"[DEBUG] POSE GStreamer ERROR from {src}: {err_msg}")
            logger.error(f"[DEBUG] Error debug: {debug_str}")

            # CRITICAL: Detect Hailo errors
            is_device_conflict = (
                "HAILO_OUT_OF_PHYSICAL_DEVICES" in err_msg or
                "HAILO_OUT_OF_PHYSICAL_DEVICES" in debug_str or
                "status=74" in err_msg or
                "status=74" in debug_str
            )
            is_comm_closed = (
                "HAILO_COMMUNICATION_CLOSED" in err_msg or
                "HAILO_COMMUNICATION_CLOSED" in debug_str or
                "status=62" in err_msg or
                "status=62" in debug_str
            )
            is_hailo_error = is_device_conflict or is_comm_closed

            if is_hailo_error:
                self._consecutive_errors += 1
                logger.error(f"[HAILO_ERROR] Detected Hailo error (count={self._consecutive_errors})")

                # For device conflict (status=74), do force reset first
                if is_device_conflict:
                    logger.error("[HAILO_ERROR] HAILO_OUT_OF_PHYSICAL_DEVICES - forcing device reset")
                    force_hailo_reset()  # This will kill any blocking processes

                if self._consecutive_errors <= self._max_consecutive_errors:
                    # Schedule restart in separate thread to avoid deadlock
                    logger.warning("[HAILO_ERROR] Scheduling automatic restart...")
                    restart_thread = threading.Thread(target=self._delayed_restart, daemon=True)
                    restart_thread.start()
                else:
                    logger.error(f"[HAILO_ERROR] Too many consecutive errors ({self._consecutive_errors}), NOT restarting")
            else:
                # Non-Hailo error - just stop
                self.stop()

        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"[DEBUG] POSE GStreamer WARNING from {src}: {err.message}")
            logger.warning(f"[DEBUG] Warning debug: {debug}")

        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                old, new, pending = message.parse_state_changed()
                logger.info(f"[DEBUG] POSE Pipeline state: {old.value_nick} -> {new.value_nick} "
                           f"(pending: {pending.value_nick})")

        elif t == Gst.MessageType.STREAM_START:
            logger.info(f"[DEBUG] POSE Stream started from {src}")

    def _delayed_restart(self):
        """Perform delayed restart (called from separate thread)."""
        # Wait a moment for current pipeline to fully release
        time.sleep(1.0)
        self.restart()

    def stop(self, release_hailo: bool = True):
        """Stop the pose detection pipeline.

        Args:
            release_hailo: If True, release Hailo resource via manager.
                          Set False when manager handles the release.
        """
        # MUTEX: Only one start/stop operation at a time
        with self._operation_lock:
            if not self._running:
                logger.info("[DEBUG] Pose detector stop() called but not running")
                return

            logger.info("[DEBUG] Stopping pose detector...")
            self._running = False

            # Stop bus watching first
            if self._bus:
                self._bus.remove_signal_watch()
                self._bus = None

            # Stop pipeline - CRITICAL: Must reach NULL state before releasing Hailo
            if self._pipeline:
                logger.info("[DEBUG] Setting pipeline state to NULL")
                ret = self._pipeline.set_state(Gst.State.NULL)
                # Wait for state change to complete
                if ret == Gst.StateChangeReturn.ASYNC:
                    logger.info("[DEBUG] Waiting for NULL state...")
                    self._pipeline.get_state(Gst.CLOCK_TIME_NONE)
                self._pipeline = None
                logger.info("[DEBUG] Pipeline set to NULL")

            # Stop main loop
            if self._main_loop:
                if self._main_loop.is_running():
                    logger.info("[DEBUG] Quitting main loop")
                    self._main_loop.quit()
                self._main_loop = None

            # Join thread (without holding lock to prevent deadlock)
            loop_thread = self._loop_thread
            self._loop_thread = None

        # Join thread OUTSIDE lock
        if loop_thread:
            logger.info("[DEBUG] Joining loop thread")
            loop_thread.join(timeout=2.0)

        # Log stats BEFORE resetting counters
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"GstHailoPoseDetector stopped ({self._frame_count} frames, {fps:.1f} FPS)")

        # Reset frame counters for clean restart
        self._frame_count = 0
        self._fps_frame_count = 0
        self._last_frame_time = 0
        self._watchdog_warned = False

        # === HAILO MANAGER: Release NPU ===
        if release_hailo and HAILO_MANAGER_AVAILABLE:
            manager = get_hailo_manager()
            if manager.current_consumer == HailoConsumer.VISION:
                # Don't auto-restart vision when explicitly stopping
                manager._state.consumer = HailoConsumer.NONE
                manager._state.released_at = time.time()
                manager._state.release_count += 1
                logger.info("[DEBUG] Hailo NPU released via manager (no auto-restart)")

    def restart(self) -> bool:
        """Safely restart the detector after an error.

        This method performs a full teardown and reinitialize cycle.
        Use when HAILO_COMMUNICATION_CLOSED or similar errors occur.

        Returns:
            True if restart successful, False otherwise
        """
        if self._restarting:
            logger.warning("[RESTART] Already restarting, skip")
            return False

        self._restarting = True
        logger.warning("[RESTART] ===== RESTARTING HAILO DETECTOR =====")

        try:
            # Full stop (release Hailo)
            self.stop(release_hailo=True)

            # Wait for Hailo device to fully release
            time.sleep(0.5)

            # Restart
            success = self.start(skip_npu_check=False)

            if success:
                logger.info("[RESTART] Detector restarted successfully")
            else:
                logger.error("[RESTART] Detector restart FAILED")
                self._consecutive_errors += 1

            return success

        except Exception as e:
            logger.error(f"[RESTART] Error during restart: {e}")
            return False
        finally:
            self._restarting = False

    def get_latest_result(self) -> PoseDetectionResult:
        """Get latest pose detection result."""
        with self._result_lock:
            return self._latest_result

    def set_detection_callback(self, callback: Callable[[PoseDetectionResult], None]):
        """Set callback for detection results."""
        self._on_detection = callback

    def set_gesture_callback(self, callback: Callable[[Gesture], None]):
        """Set callback for gesture detection events."""
        self._gesture_callback = callback

    def get_current_gesture(self) -> Optional[Gesture]:
        """Get the current detected gesture."""
        return self._current_gesture

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        with self._result_lock:
            return self._latest_result.fps


# Singleton
_pose_detector: Optional[GstHailoPoseDetector] = None
_pose_detector_lock = threading.Lock()


def get_gst_pose_detector(rtsp_url: str = None) -> GstHailoPoseDetector:
    """Get or create GstHailoPoseDetector singleton."""
    global _pose_detector
    if _pose_detector is None:
        with _pose_detector_lock:
            if _pose_detector is None:
                _pose_detector = GstHailoPoseDetector(rtsp_url)
    return _pose_detector


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    print("=== GStreamer Hailo POSE Detector Test ===")
    print(f"Model: yolov8s_pose_h10.hef")

    detector = GstHailoPoseDetector()

    def on_detect(result: PoseDetectionResult):
        if result.poses:
            print(f"\n[{result.fps:.1f} FPS] {result.total_detections} det, {result.faces_detected} faces, {len(result.valid_persons)} valid:")
            for i, p in enumerate(result.poses):
                status = "VALID" if p.is_valid_person else f"REJECTED"
                face_info = f"face={p.face_confidence:.0%}" if p.has_face else "no_face"
                print(f"  {i}: {status} conf={p.confidence:.2f} {face_info} reason={p.validation_reason}")

    detector.set_detection_callback(on_detect)

    if not detector.start():
        print("Failed to start!")
        sys.exit(1)

    print("Running for 20 seconds...")
    try:
        time.sleep(20)
    except KeyboardInterrupt:
        pass

    detector.stop()
    print("Done!")
