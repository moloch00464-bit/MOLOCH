#!/usr/bin/env python3
"""
M.O.L.O.C.H. Autonomous Tracker
================================

Dedicated 15 Hz tracking thread with ContinuousMove velocity control.
Implements proportional tracking and search behavior.

Features:
- 15 Hz tracking loop (66ms cycle)
- ContinuousMove with proportional velocity (no Stop() during tracking)
- Search mode: slow pan sweep when target lost
- Largest bounding box selection
- Configurable deadzone and gain

Author: M.O.L.O.C.H. System
Date: 2026-02-04
"""

import time
import math
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

# Import perception state for user visibility check
try:
    from context.perception_state import get_perception_state, is_user_visible
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    logger.warning("perception_state not available - tracker will use raw detections")


class TrackerState(Enum):
    """Tracker state machine."""
    IDLE = "idle"           # Tracking disabled
    TRACKING = "tracking"   # Following target with ContinuousMove
    SEARCHING = "searching" # Lost target, sweeping
    LOCKED = "locked"       # Target centered in deadzone
    DWELL = "dwell"         # Target acquired, waiting before movement
    FROZEN = "frozen"       # Target perfectly centered, no movement needed


class TargetType(Enum):
    """Type of tracking target - adaptive selection."""
    NONE = "none"           # No valid target
    FACE = "face"           # Tracking face (preferred)
    BODY = "body"           # Tracking full body (fallback)


@dataclass
class TrackingConfig:
    """Tracking parameters."""
    # === LOCK/FROZEN State Parameters ===
    lock_threshold_pixels: int = 8      # Enter LOCK when error < this
    unlock_threshold_pixels: int = 15   # Leave LOCK when error > this (hysteresis)
    frozen_threshold_pixels: int = 5    # Enter FROZEN when error < this (very stable)

    # === Dwell Timer (wait before moving) ===
    dwell_time_sec: float = 1.5         # Wait this long after target acquired before moving

    # Proportional gains for velocity control
    pan_gain: float = 0.004             # Reduced for stability
    tilt_gain: float = 0.004

    # Maximum velocity (ONVIF normalized 0-1) - reduced for stability
    max_velocity: float = 0.35          # Reduced from 0.4

    # Minimum velocity threshold - below this, clamp to zero
    min_velocity: float = 0.08

    # === Virtual PTZ Position Tracking ===
    virtual_pan_min: float = -1.0       # Soft limit left
    virtual_pan_max: float = 1.0        # Soft limit right
    virtual_tilt_min: float = -0.5      # Soft limit down
    virtual_tilt_max: float = 0.5       # Soft limit up
    virtual_position_decay: float = 0.02  # Position integration factor per cycle

    # Search mode parameters
    search_speed: float = 0.10          # Reduced from 0.12
    search_direction_interval: float = 4.0  # Seconds before switching direction
    search_reset_to_center: bool = True # Reset to center when starting search

    # Target lost timeout before starting search
    target_lost_timeout: float = 2.0

    # Frame dimensions (detection space)
    frame_width: int = 640
    frame_height: int = 640

    # === Detection filtering (STRICT - reject hands, partial bodies) ===
    min_bbox_height_ratio: float = 0.40   # Increased from 0.25 - reject small detections
    max_bbox_center_y_ratio: float = 0.75 # Reject detections in bottom 25% of frame
    min_bbox_area_ratio: float = 0.08     # Increased from 0.05
    min_confidence: float = 0.50          # Increased from 0.45
    min_aspect_ratio: float = 0.35        # Min width/height (reject narrow objects)

    # === Target persistence (stable tracking) ===
    confidence_hysteresis: float = 0.15   # New target needs +15% confidence to switch
    stability_frames: int = 7             # Increased from 5 - more frames before switching
    center_priority_weight: float = 0.4   # Increased bonus for centered detections

    # === ADAPTIVE TARGET STRATEGY ===
    # Face tracking (preferred)
    face_min_confidence: float = 0.55     # Min face confidence to track face
    face_min_stability: int = 4           # Frames before switching to face
    face_max_bbox_height: float = 0.65    # Max bbox height for face mode (not full body)

    # Body tracking (fallback)
    body_min_confidence: float = 0.45     # Min confidence for body tracking
    body_min_bbox_height: float = 0.30    # Min height for valid body
    body_min_stability: int = 5           # Frames before switching to body

    # Target switching
    switch_cooldown_sec: float = 1.0      # Min time between target type switches
    prefer_current_target: bool = True    # Stick with current target if still valid


@dataclass
class DetectionData:
    """Detection data for tracking."""
    detected: bool = False
    bbox: List[float] = field(default_factory=lambda: [0, 0, 0, 0])  # x1, y1, x2, y2
    center_x: float = 0.5  # Normalized 0-1
    center_y: float = 0.5  # Normalized 0-1
    confidence: float = 0.0
    target_id: int = 0     # Stable target identifier
    timestamp: float = field(default_factory=time.time)
    # Pose-based tracking
    is_pose_detection: bool = False   # True if from pose detector
    has_face: bool = False            # Face keypoints visible
    has_torso: bool = False           # Torso keypoints visible
    head_center_x: float = 0.5        # Head keypoint center (for tracking)
    head_center_y: float = 0.5        # Head keypoint center (for tracking)
    validation_reason: str = ""
    # Adaptive target type
    target_type: str = "none"         # "face", "body", or "none"


class AutonomousTracker:
    """
    Autonomous person tracking with 15 Hz control loop.

    Uses ContinuousMove for smooth proportional velocity control.
    Does NOT call Stop() during tracking - only sends continuous velocity updates.
    """

    LOOP_RATE_HZ = 5  # 200ms cycle time (reduced to avoid camera rate-limiting)

    def __init__(self, camera_controller=None, config: TrackingConfig = None):
        """
        Initialize tracker.

        Args:
            camera_controller: SonoffCameraController instance
            config: Tracking configuration
        """
        self.camera = camera_controller
        self.config = config or TrackingConfig()

        # State
        self.state = TrackerState.IDLE
        self.tracking_active = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Detection data (updated by vision system)
        self.latest_detection = DetectionData()
        self.last_detection_time = 0.0

        # Target persistence state
        self.current_target_id = 0
        self.current_target_bbox = [0, 0, 0, 0]
        self.current_target_confidence = 0.0
        self.candidate_target_id = 0
        self.candidate_stability_count = 0
        self._next_target_id = 1

        # === ADAPTIVE TARGET STATE ===
        self.current_target_type = TargetType.NONE
        self.candidate_target_type = TargetType.NONE
        self.target_type_stability = 0
        self.last_target_switch_time = 0.0

        # Search mode state
        self.search_direction = 1  # 1 = right, -1 = left
        self.last_direction_switch = 0.0

        # === Virtual PTZ Position Tracking ===
        self.virtual_pan = 0.0      # Estimated pan position (-1 to 1)
        self.virtual_tilt = 0.0     # Estimated tilt position (-1 to 1)
        self.last_velocity_pan = 0.0
        self.last_velocity_tilt = 0.0
        self.last_move_time = 0.0

        # === Dwell Timer State ===
        self.dwell_start_time = 0.0
        self.dwell_target_acquired = False

        # Statistics
        self.stats = {
            "cycles": 0,
            "tracking_moves": 0,
            "search_moves": 0,
            "state_changes": 0,
            "detections_filtered": 0,
            "target_switches": 0
        }

        # Callbacks
        self.on_state_change: Optional[Callable[[TrackerState], None]] = None

        logger.info(f"AutonomousTracker initialized (rate={self.LOOP_RATE_HZ}Hz)")

    def set_camera(self, camera_controller):
        """Set camera controller."""
        self.camera = camera_controller
        # Debug: Log object IDs to verify same instance
        logger.info(f"Camera controller connected to AutonomousTracker")
        logger.info(f"  Controller id: {id(camera_controller)}")
        if camera_controller:
            logger.info(f"  ptz_service id: {id(camera_controller.ptz_service) if camera_controller.ptz_service else 'None'}")
            logger.info(f"  profile_token: {camera_controller.profile_token}")
            logger.info(f"  is_connected: {camera_controller.is_connected}")

    def start(self) -> bool:
        """Start the tracking thread."""
        if self._running:
            logger.warning("Tracker already running")
            return True

        if not self.camera:
            logger.error("No camera controller - cannot start tracker")
            return False

        # === VERIFY CONTROLLER INSTANCE ===
        logger.info("=" * 60)
        logger.info("=== TRACKER START: CONTROLLER VERIFICATION ===")
        logger.info(f"self.camera:          {self.camera}")
        logger.info(f"self.camera id:       {id(self.camera)}")
        logger.info(f"is_connected:         {self.camera.is_connected}")
        logger.info(f"ptz_service:          {self.camera.ptz_service}")
        logger.info(f"ptz_service id:       {id(self.camera.ptz_service) if self.camera.ptz_service else 'None'}")
        logger.info(f"profile_token:        {self.camera.profile_token}")
        logger.info(f"mode:                 {self.camera.mode}")
        logger.info("=" * 60)

        # === FORCE DIAGNOSTIC PTZ MOVEMENT on start ===
        logger.info("=" * 60)
        logger.info("=== PTZ DIAGNOSTIC MOVEMENT TEST ===")
        logger.info(f"Controller object id: {id(self.camera)}")
        logger.info(f"ptz_service object id: {id(self.camera.ptz_service) if self.camera.ptz_service else 'None'}")
        logger.info(f"profile_token: {self.camera.profile_token}")

        # Log ONVIF request details
        if self.camera.ptz_service:
            try:
                logger.info(f"PTZ Service type: {type(self.camera.ptz_service)}")
            except:
                pass

        # Execute diagnostic movement: pan RIGHT at 0.3 for 2 seconds
        logger.info("-" * 40)
        logger.info("DIAGNOSTIC: Sending continuous_move(pan=0.3, tilt=0.0) for 2 seconds...")
        logger.info("EXPECTED: Camera should physically pan RIGHT")

        test_start = time.time()
        test_result = self.camera.continuous_move(0.3, 0.0, timeout_sec=2.0, verbose=True)
        test_elapsed = time.time() - test_start

        logger.info(f"DIAGNOSTIC: continuous_move returned: {test_result}")
        logger.info(f"DIAGNOSTIC: Execution time: {test_elapsed:.2f}s")

        if test_result:
            logger.info("DIAGNOSTIC: Command accepted - camera SHOULD be moving!")
        else:
            logger.error("DIAGNOSTIC: Command FAILED - NO MOVEMENT!")

        # After movement, explicitly send Stop and log
        logger.info("-" * 40)
        logger.info("DIAGNOSTIC: Sending Stop() command...")
        try:
            if self.camera.ptz_service:
                stop_result = self.camera.ptz_service.Stop({
                    'ProfileToken': self.camera.profile_token,
                    'PanTilt': True,
                    'Zoom': True
                })
                logger.info(f"DIAGNOSTIC: Stop() result: {stop_result}")
            else:
                self.camera.stop()
                logger.info("DIAGNOSTIC: stop() called via controller")
        except Exception as e:
            logger.error(f"DIAGNOSTIC: Stop() error: {e}")

        logger.info("=" * 60)
        logger.info("PTZ DIAGNOSTIC COMPLETE - Did camera physically move?")
        logger.info("=" * 60)

        self._running = True
        self.tracking_active = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

        logger.info("AutonomousTracker started")
        return True

    def stop(self):
        """Stop the tracking thread."""
        self._running = False
        self.tracking_active = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Stop camera movement
        if self.camera:
            self.camera.stop()

        self._set_state(TrackerState.IDLE)
        logger.info(f"AutonomousTracker stopped (cycles={self.stats['cycles']})")

    def update_detection(self, detections: List[Dict], frame_width: int = 640, frame_height: int = 640):
        """
        Update with new detection data from vision system.

        Implements stable target selection:
        - Filters out hands, partial bodies, low-confidence detections
        - Maintains current target if still valid
        - Requires stability before switching to new target

        Args:
            detections: List of detection dicts with bbox, confidence, class
            frame_width: Detection frame width
            frame_height: Detection frame height
        """
        with self._lock:
            self.config.frame_width = frame_width
            self.config.frame_height = frame_height
            frame_area = frame_width * frame_height

            if not detections:
                self.latest_detection = DetectionData(detected=False)
                self.candidate_stability_count = 0
                return

            # Filter for person detections
            person_dets = [d for d in detections if d.get("class", "") == "person"]
            if not person_dets:
                person_dets = detections  # Use any if no person class

            # === STAGE 1: Filter out invalid detections ===
            valid_dets = []
            for d in person_dets:
                bbox = d.get("bbox", [0, 0, 0, 0])
                conf = d.get("confidence", 0)

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # Filter 1: Minimum confidence
                if conf < self.config.min_confidence:
                    self.stats["detections_filtered"] += 1
                    continue

                # Filter 2: Minimum height (reject hands, partial bodies)
                height_ratio = height / frame_height
                if height_ratio < self.config.min_bbox_height_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                # Filter 3: Minimum area
                area_ratio = area / frame_area
                if area_ratio < self.config.min_bbox_area_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                # Filter 4: Aspect ratio (reject narrow objects like hands)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < self.config.min_aspect_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                # Filter 5: Reject detections in bottom portion of frame (likely feet/floor)
                center_y_ratio = ((y1 + y2) / 2) / frame_height
                if center_y_ratio > self.config.max_bbox_center_y_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                valid_dets.append(d)

            if not valid_dets:
                self.latest_detection = DetectionData(detected=False)
                self.candidate_stability_count = 0
                return

            # === STAGE 2: Score detections ===
            def score_detection(d):
                bbox = d.get("bbox", [0, 0, 0, 0])
                conf = d.get("confidence", 0)
                x1, y1, x2, y2 = bbox

                # Base score: area (larger = closer)
                area = (x2 - x1) * (y2 - y1)
                area_score = area / frame_area

                # Center bonus: prioritize detections near frame center
                center_x = (x1 + x2) / 2 / frame_width
                center_y = (y1 + y2) / 2 / frame_height
                dist_from_center = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
                center_score = 1.0 - min(1.0, dist_from_center * 2)

                # Combined score
                return area_score + (center_score * self.config.center_priority_weight) + (conf * 0.2)

            # Sort by score
            scored_dets = sorted(valid_dets, key=score_detection, reverse=True)
            best_candidate = scored_dets[0]
            best_bbox = best_candidate.get("bbox", [0, 0, 0, 0])
            best_conf = best_candidate.get("confidence", 0)

            # === STAGE 3: Target persistence with hysteresis ===
            x1, y1, x2, y2 = best_bbox
            center_x = (x1 + x2) / 2 / frame_width
            center_y = (y1 + y2) / 2 / frame_height

            # Check if current target is still in the detections
            current_target_still_valid = False
            current_target_detection = None
            if self.current_target_id > 0 and self.current_target_confidence > 0:
                # Look for a detection overlapping with current target
                for d in valid_dets:
                    bbox = d.get("bbox", [0, 0, 0, 0])
                    if self._bbox_iou(bbox, self.current_target_bbox) > 0.3:
                        current_target_still_valid = True
                        current_target_detection = d
                        # Update current target position
                        self.current_target_bbox = bbox
                        self.current_target_confidence = d.get("confidence", 0)
                        break

            # Find highest-confidence detection that is NOT the current target
            # (Use raw confidence for switch decision, not scored selection)
            best_alternative = None
            best_alternative_conf = 0.0
            for d in valid_dets:
                conf = d.get("confidence", 0)
                bbox = d.get("bbox", [0, 0, 0, 0])
                # Different from current target?
                if self._bbox_iou(bbox, self.current_target_bbox) < 0.3:
                    if conf > best_alternative_conf:
                        best_alternative_conf = conf
                        best_alternative = d

            # Decide whether to switch targets
            should_switch = False
            switch_to_bbox = best_bbox
            switch_to_conf = best_conf

            if not current_target_still_valid:
                # Current target lost - switch immediately to best scored
                should_switch = True
                self.candidate_stability_count = 0
            elif best_alternative and best_alternative_conf > self.current_target_confidence + self.config.confidence_hysteresis:
                # Higher-confidence alternative found - require stability
                self.candidate_stability_count += 1
                logger.debug(f"Stability count: {self.candidate_stability_count}/{self.config.stability_frames} "
                           f"(alt_conf={best_alternative_conf:.2f} vs cur={self.current_target_confidence:.2f})")
                if self.candidate_stability_count >= self.config.stability_frames:
                    should_switch = True
                    switch_to_bbox = best_alternative.get("bbox", [0, 0, 0, 0])
                    switch_to_conf = best_alternative_conf
                    logger.info(f"TARGET SWITCH: id={self.current_target_id} -> new (conf {self.current_target_confidence:.2f} -> {switch_to_conf:.2f})")
            else:
                self.candidate_stability_count = 0

            if should_switch:
                self.current_target_id = self._next_target_id
                self._next_target_id += 1
                self.current_target_bbox = switch_to_bbox
                self.current_target_confidence = switch_to_conf
                self.stats["target_switches"] += 1

            # Use current target position for tracking
            if self.current_target_id > 0:
                tx1, ty1, tx2, ty2 = self.current_target_bbox
                center_x = (tx1 + tx2) / 2 / frame_width
                center_y = (ty1 + ty2) / 2 / frame_height

            self.latest_detection = DetectionData(
                detected=True,
                bbox=self.current_target_bbox,
                center_x=center_x,
                center_y=center_y,
                confidence=self.current_target_confidence,
                target_id=self.current_target_id,
                timestamp=time.time()
            )
            self.last_detection_time = time.time()

    def _bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bboxes."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update_pose_detection(self, poses: List[Dict], frame_width: int = 640, frame_height: int = 640):
        """
        ADAPTIVE pose detection - decides between FACE and BODY tracking.

        Priority: FACE > BODY > NONE
        - Track face if stable face keypoints detected
        - Fallback to body center if no stable face
        - Ignore fragments (hands, partial bodies)

        Args:
            poses: List of pose dicts from GstHailoPoseDetector
            frame_width: Detection frame width
            frame_height: Detection frame height
        """
        with self._lock:
            self.config.frame_width = frame_width
            self.config.frame_height = frame_height
            now = time.time()

            if not poses:
                self.latest_detection = DetectionData(detected=False, is_pose_detection=True)
                self.candidate_stability_count = 0
                self.target_type_stability = 0
                return

            # === STAGE 1: Categorize all detections ===
            face_candidates = []   # Poses with good face
            body_candidates = []   # Poses with body but weak/no face

            for p in poses:
                bbox = p.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue

                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                height_ratio = height / frame_height
                area_ratio = (width * height) / (frame_width * frame_height)
                aspect_ratio = width / height if height > 0 else 0

                # === REJECT FRAGMENTS ===
                # Too small
                if area_ratio < self.config.min_bbox_area_ratio:
                    self.stats["detections_filtered"] += 1
                    continue
                # Too narrow (likely hand/arm)
                if aspect_ratio < self.config.min_aspect_ratio:
                    self.stats["detections_filtered"] += 1
                    continue
                # Bottom edge (likely feet)
                center_y = (bbox[1] + bbox[3]) / 2 / frame_height
                if center_y > self.config.max_bbox_center_y_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                # === CATEGORIZE BY FACE QUALITY ===
                has_face = p.get("has_face", False)
                face_conf = p.get("face_confidence", 0)
                face_center = p.get("face_center")
                has_torso = p.get("has_torso", False)

                if has_face and face_conf >= self.config.face_min_confidence and face_center:
                    # Good face - FACE candidate
                    if height_ratio <= self.config.face_max_bbox_height:
                        face_candidates.append(p)
                    else:
                        # Face visible but bbox too large - track as body
                        body_candidates.append(p)
                elif has_torso and height_ratio >= self.config.body_min_bbox_height:
                    # Has body but weak/no face - BODY candidate
                    body_candidates.append(p)
                else:
                    self.stats["detections_filtered"] += 1

            # === STAGE 2: Select target type and best candidate ===
            selected_pose = None
            selected_type = TargetType.NONE
            track_x, track_y = 0.5, 0.5

            # Score function for ranking
            def score_pose(p, for_face: bool):
                face_conf = p.get("face_confidence", 0)
                det_conf = p.get("confidence", 0)
                fc = p.get("face_center", (0.5, 0.5))
                dist = math.sqrt((fc[0] - 0.5)**2 + (fc[1] - 0.5)**2)
                center_bonus = 1.0 - min(1.0, dist * 2)

                if for_face:
                    return face_conf * 0.5 + center_bonus * 0.3 + det_conf * 0.2
                else:
                    bbox = p.get("bbox", [0, 0, 0, 0])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area_score = area / (frame_width * frame_height)
                    return area_score * 0.4 + center_bonus * 0.3 + det_conf * 0.3

            # === ADAPTIVE SELECTION ===
            # Priority: Stick with current type if still valid, else switch
            if self.config.prefer_current_target and self.current_target_type != TargetType.NONE:
                cooldown_ok = (now - self.last_target_switch_time) > self.config.switch_cooldown_sec

                if self.current_target_type == TargetType.FACE and face_candidates:
                    # Continue face tracking
                    face_candidates.sort(key=lambda p: score_pose(p, True), reverse=True)
                    selected_pose = face_candidates[0]
                    selected_type = TargetType.FACE
                elif self.current_target_type == TargetType.BODY and body_candidates:
                    # Continue body tracking
                    body_candidates.sort(key=lambda p: score_pose(p, False), reverse=True)
                    selected_pose = body_candidates[0]
                    selected_type = TargetType.BODY
                elif cooldown_ok:
                    # Current type lost - try to switch
                    pass  # Fall through to new selection

            # New selection if no current target maintained
            if selected_pose is None:
                if face_candidates:
                    # FACE available - check stability
                    self.candidate_target_type = TargetType.FACE
                    self.target_type_stability += 1

                    if self.target_type_stability >= self.config.face_min_stability:
                        face_candidates.sort(key=lambda p: score_pose(p, True), reverse=True)
                        selected_pose = face_candidates[0]
                        selected_type = TargetType.FACE
                        if self.current_target_type != TargetType.FACE:
                            logger.info(f"[ADAPTIVE] Switching to FACE tracking (stability={self.target_type_stability})")
                            self.last_target_switch_time = now

                elif body_candidates:
                    # BODY available - check stability
                    self.candidate_target_type = TargetType.BODY
                    self.target_type_stability += 1

                    if self.target_type_stability >= self.config.body_min_stability:
                        body_candidates.sort(key=lambda p: score_pose(p, False), reverse=True)
                        selected_pose = body_candidates[0]
                        selected_type = TargetType.BODY
                        if self.current_target_type != TargetType.BODY:
                            logger.info(f"[ADAPTIVE] Switching to BODY tracking (stability={self.target_type_stability})")
                            self.last_target_switch_time = now
                else:
                    # Nothing valid
                    self.target_type_stability = 0

            # === STAGE 3: Create detection data ===
            if selected_pose is None:
                self.latest_detection = DetectionData(detected=False, is_pose_detection=True)
                return

            # Update current target type
            self.current_target_type = selected_type

            # Get tracking point based on type
            bbox = selected_pose.get("bbox", [0, 0, 0, 0])
            if selected_type == TargetType.FACE:
                # Track FACE center
                face_center = selected_pose.get("face_center", (0.5, 0.5))
                track_x, track_y = face_center
            else:
                # Track BODY center (upper body)
                # Use torso center (between shoulders and hips)
                bbox_center_x = (bbox[0] + bbox[2]) / 2 / frame_width
                bbox_center_y = (bbox[1] + bbox[3]) / 2 / frame_height
                # Offset slightly up for upper body
                track_x = bbox_center_x
                track_y = bbox_center_y * 0.85  # Bias towards upper body

            # Update target ID
            if self.current_target_id == 0:
                self.current_target_id = self._next_target_id
                self._next_target_id += 1

            self.current_target_bbox = bbox
            self.current_target_confidence = selected_pose.get("face_confidence", 0) if selected_type == TargetType.FACE else selected_pose.get("confidence", 0)

            # Create detection data with ADAPTIVE tracking point
            self.latest_detection = DetectionData(
                detected=True,
                bbox=bbox,
                center_x=track_x,
                center_y=track_y,
                confidence=self.current_target_confidence,
                target_id=self.current_target_id,
                timestamp=time.time(),
                is_pose_detection=True,
                has_face=selected_pose.get("has_face", False),
                has_torso=selected_pose.get("has_torso", False),
                head_center_x=track_x,
                head_center_y=track_y,
                validation_reason=selected_pose.get("validation_reason", ""),
                target_type=selected_type.value
            )
            self.last_detection_time = time.time()

            # Log adaptive tracking info every 30 cycles
            if self.stats["cycles"] % 30 == 0:
                logger.info(f"[ADAPTIVE] {selected_type.value.upper()} at ({track_x:.2f},{track_y:.2f}) "
                           f"conf={self.current_target_confidence:.2f} "
                           f"faces={len(face_candidates)} bodies={len(body_candidates)}")

    def _tracking_loop(self):
        """Main 15 Hz tracking loop."""
        cycle_time = 1.0 / self.LOOP_RATE_HZ
        logger.info(f"Tracking loop STARTED (rate={self.LOOP_RATE_HZ}Hz, camera={id(self.camera) if self.camera else 'None'})")

        while self._running:
            loop_start = time.time()

            try:
                if self.tracking_active:
                    self._process_tracking_cycle()
                    self.stats["cycles"] += 1

                    # Log status every 3 seconds (45 cycles at 15Hz)
                    if self.stats["cycles"] % 45 == 0:
                        logger.info(f"Tracker loop: cycles={self.stats['cycles']} state={self.state.value} "
                                  f"search_moves={self.stats['search_moves']} tracking_moves={self.stats['tracking_moves']}")

            except Exception as e:
                logger.error(f"Tracking cycle error: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = cycle_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_tracking_cycle(self):
        """Process one tracking cycle."""
        with self._lock:
            detection = self.latest_detection

        now = time.time()
        time_since_detection = now - self.last_detection_time

        # DEBUG: Log state every 15 cycles (1 second)
        debug_log = (self.stats["cycles"] % 15 == 0)
        if debug_log:
            logger.info(f"[CYCLE] detected={detection.detected} "
                       f"time_since={time_since_detection:.2f}s "
                       f"conf={detection.confidence:.2f} "
                       f"state={self.state.value}")

        if detection.detected and time_since_detection < 0.5:
            # We have a recent detection - TRACK
            self._do_tracking(detection)
        else:
            # No recent detection - check if we should SEARCH
            if time_since_detection > self.config.target_lost_timeout:
                if debug_log:
                    logger.info(f"[CYCLE] No detection for {time_since_detection:.1f}s > {self.config.target_lost_timeout}s -> SEARCH")
                self._do_search()
            else:
                # Brief loss - maintain last velocity or slow down
                if debug_log:
                    logger.info(f"[CYCLE] Brief loss ({time_since_detection:.2f}s) -> COAST")
                self._do_coast()

    def _do_tracking(self, detection: DetectionData):
        """Execute tracking with dwell timer, proportional velocity, and LOCK/FROZEN states."""
        now = time.time()

        # Calculate error from frame center (pixels)
        center_x_px = detection.center_x * self.config.frame_width
        center_y_px = detection.center_y * self.config.frame_height

        frame_center_x = self.config.frame_width / 2
        frame_center_y = self.config.frame_height / 2

        error_x = center_x_px - frame_center_x  # Positive = target is RIGHT of center
        error_y = center_y_px - frame_center_y  # Positive = target is BELOW center
        error_magnitude = math.sqrt(error_x**2 + error_y**2)

        # DEBUG: Log every 15 cycles
        debug_log = (self.stats["cycles"] % 15 == 0)
        if debug_log:
            logger.info(f"[TRACK] error=({error_x:+.0f},{error_y:+.0f})px mag={error_magnitude:.0f}px "
                       f"state={self.state.value} vPTZ=({self.virtual_pan:+.2f},{self.virtual_tilt:+.2f})")

        # === DWELL STATE: Wait before starting movement ===
        if not self.dwell_target_acquired:
            # First detection - start dwell timer
            self.dwell_target_acquired = True
            self.dwell_start_time = now
            self._set_state(TrackerState.DWELL)
            logger.info(f"[DWELL] Target acquired - waiting {self.config.dwell_time_sec}s before tracking")
            return

        if self.state == TrackerState.DWELL:
            dwell_elapsed = now - self.dwell_start_time
            if dwell_elapsed < self.config.dwell_time_sec:
                # Still waiting - don't move
                if debug_log:
                    logger.info(f"[DWELL] Waiting... {dwell_elapsed:.1f}s / {self.config.dwell_time_sec}s")
                return
            else:
                # Dwell complete - transition to tracking
                logger.info(f"[DWELL] Complete - starting tracking")
                self._set_state(TrackerState.TRACKING)

        # === FROZEN STATE: Target perfectly centered ===
        if error_magnitude < self.config.frozen_threshold_pixels:
            if self.state != TrackerState.FROZEN:
                self._set_state(TrackerState.FROZEN)
                self._send_continuous_move(0.0, 0.0)  # Stop
                logger.info(f"[FROZEN] Target perfectly centered (error={error_magnitude:.0f}px)")
            return

        # === LOCKED STATE WITH HYSTERESIS ===
        if self.state == TrackerState.LOCKED or self.state == TrackerState.FROZEN:
            if error_magnitude > self.config.unlock_threshold_pixels:
                self._set_state(TrackerState.TRACKING)
                if debug_log:
                    logger.info(f"[TRACK] UNLOCK: error {error_magnitude:.0f}px > {self.config.unlock_threshold_pixels}px")
            else:
                # Stay locked - no movement
                if debug_log:
                    logger.info(f"[TRACK] LOCKED: error {error_magnitude:.0f}px (no move)")
                return
        else:
            # Check if we should enter LOCK
            if error_magnitude < self.config.lock_threshold_pixels:
                self._set_state(TrackerState.LOCKED)
                self._send_continuous_move(0.0, 0.0)
                if debug_log:
                    logger.info(f"[TRACK] LOCK: error {error_magnitude:.0f}px < {self.config.lock_threshold_pixels}px")
                return

        # === TRACKING MODE ===
        self._set_state(TrackerState.TRACKING)

        # Calculate proportional velocities
        vel_pan = -error_x * self.config.pan_gain
        vel_tilt = -error_y * self.config.tilt_gain

        # === SOFT LIMITS: Check virtual position ===
        # Reduce velocity as we approach limits
        if self.virtual_pan >= self.config.virtual_pan_max * 0.8 and vel_pan > 0:
            vel_pan *= 0.5  # Slow down near right limit
            if debug_log:
                logger.info(f"[TRACK] Near right limit - reducing pan velocity")
        elif self.virtual_pan <= self.config.virtual_pan_min * 0.8 and vel_pan < 0:
            vel_pan *= 0.5  # Slow down near left limit
            if debug_log:
                logger.info(f"[TRACK] Near left limit - reducing pan velocity")

        if self.virtual_tilt >= self.config.virtual_tilt_max * 0.8 and vel_tilt > 0:
            vel_tilt *= 0.5
        elif self.virtual_tilt <= self.config.virtual_tilt_min * 0.8 and vel_tilt < 0:
            vel_tilt *= 0.5

        # Hard stop at limits
        if self.virtual_pan >= self.config.virtual_pan_max and vel_pan > 0:
            vel_pan = 0.0
        elif self.virtual_pan <= self.config.virtual_pan_min and vel_pan < 0:
            vel_pan = 0.0

        if self.virtual_tilt >= self.config.virtual_tilt_max and vel_tilt > 0:
            vel_tilt = 0.0
        elif self.virtual_tilt <= self.config.virtual_tilt_min and vel_tilt < 0:
            vel_tilt = 0.0

        # Clamp to max velocity
        vel_pan = max(-self.config.max_velocity, min(self.config.max_velocity, vel_pan))
        vel_tilt = max(-self.config.max_velocity, min(self.config.max_velocity, vel_tilt))

        # Clamp to ZERO if below minimum threshold (avoid oscillation)
        if abs(vel_pan) < self.config.min_velocity:
            vel_pan = 0.0
        if abs(vel_tilt) < self.config.min_velocity:
            vel_tilt = 0.0

        # If both velocities are zero, don't send command
        if vel_pan == 0.0 and vel_tilt == 0.0:
            if debug_log:
                logger.info(f"[TRACK] vel=0 (below threshold or at limit)")
            return

        if debug_log:
            logger.info(f"[TRACK] vel=({vel_pan:+.3f},{vel_tilt:+.3f}) -> ContinuousMove")

        # Send ContinuousMove command
        result = self._send_continuous_move(vel_pan, vel_tilt)
        if result:
            self.stats["tracking_moves"] += 1
            # Update virtual position estimate
            self._update_virtual_position(vel_pan, vel_tilt)

        if self.stats["tracking_moves"] % 15 == 0:
            logger.info(f"TRACK: err=({error_x:+.0f},{error_y:+.0f})px vel=({vel_pan:+.3f},{vel_tilt:+.3f}) "
                       f"vPTZ=({self.virtual_pan:+.2f},{self.virtual_tilt:+.2f})")

    def _update_virtual_position(self, vel_pan: float, vel_tilt: float):
        """Update virtual PTZ position estimate based on velocity commands."""
        now = time.time()
        if self.last_move_time > 0:
            dt = now - self.last_move_time
            # Integrate velocity to update position estimate
            self.virtual_pan += vel_pan * dt * self.config.virtual_position_decay
            self.virtual_tilt += vel_tilt * dt * self.config.virtual_position_decay
            # Clamp to limits
            self.virtual_pan = max(self.config.virtual_pan_min, min(self.config.virtual_pan_max, self.virtual_pan))
            self.virtual_tilt = max(self.config.virtual_tilt_min, min(self.config.virtual_tilt_max, self.virtual_tilt))

        self.last_velocity_pan = vel_pan
        self.last_velocity_tilt = vel_tilt
        self.last_move_time = now

    def _do_search(self):
        """Execute search mode - slow pan sweep with reset to center."""
        # Check perception state - don't search if user is visible
        if PERCEPTION_AVAILABLE:
            try:
                if is_user_visible():
                    # User is visible in perception - don't search, just coast
                    if self.state == TrackerState.SEARCHING:
                        logger.info("[SEARCH] Aborted: user_visible=True in perception")
                    self._do_coast()
                    return
            except:
                pass

        now = time.time()

        # === RESET TO CENTER when starting search ===
        if self.state != TrackerState.SEARCHING:
            self._set_state(TrackerState.SEARCHING)
            if self.config.search_reset_to_center:
                logger.info(f"[SEARCH] Starting - resetting to center (vPTZ was {self.virtual_pan:+.2f},{self.virtual_tilt:+.2f})")
                # Reset virtual position to center
                self.virtual_pan = 0.0
                self.virtual_tilt = 0.0
                # Set search direction based on which side we were on
                self.search_direction = 1  # Start sweeping right
                self.last_direction_switch = now
            # Reset dwell state for next target acquisition
            self.dwell_target_acquired = False
            self.dwell_start_time = 0.0

        # Switch direction periodically OR when hitting soft limits
        should_switch = False
        if now - self.last_direction_switch > self.config.search_direction_interval:
            should_switch = True
        elif self.virtual_pan >= self.config.virtual_pan_max * 0.9 and self.search_direction > 0:
            should_switch = True
            logger.info(f"[SEARCH] Hit right limit - reversing")
        elif self.virtual_pan <= self.config.virtual_pan_min * 0.9 and self.search_direction < 0:
            should_switch = True
            logger.info(f"[SEARCH] Hit left limit - reversing")

        if should_switch:
            self.search_direction *= -1
            self.last_direction_switch = now
            logger.info(f"[SEARCH] Direction switch -> {'RIGHT' if self.search_direction > 0 else 'LEFT'}")

        # Pan sweep at search speed
        vel_pan = self.config.search_speed * self.search_direction
        vel_tilt = 0.0

        # Log first search move and then every 15 moves
        if self.stats["search_moves"] == 0:
            logger.info(f"[SEARCH] Starting pan sweep vel_pan={vel_pan:+.3f}")

        if self._send_continuous_move(vel_pan, vel_tilt):
            self.stats["search_moves"] += 1
            self._update_virtual_position(vel_pan, vel_tilt)

    def _do_coast(self):
        """Coast - gradually slow down when target briefly lost."""
        # Send zero velocity to smoothly stop
        self._send_continuous_move(0.0, 0.0)

    def _send_continuous_move(self, vel_pan: float, vel_tilt: float) -> bool:
        """
        Send ContinuousMove command to camera.

        Args:
            vel_pan: Pan velocity (-1 to 1)
            vel_tilt: Tilt velocity (-1 to 1)

        Returns:
            True if command sent successfully
        """
        if not self.camera:
            logger.warning("[TRACKER] ContinuousMove FAILED: no camera controller (self.camera is None)")
            return False

        # Check exclusive PTZ lock
        if hasattr(self.camera, '_exclusive_owner') and self.camera._exclusive_owner is not None:
            return False

        if not self.camera.is_connected:
            logger.warning(f"[TRACKER] ContinuousMove FAILED: camera not connected (is_connected={self.camera.is_connected})")
            return False

        # Log every 15th call (1 per second at 15Hz) to avoid spam
        total_moves = self.stats["tracking_moves"] + self.stats["search_moves"]

        # Verbose on first call to log full request
        verbose = (total_moves == 0)

        if total_moves % 15 == 0:
            logger.info(f"[TRACKER] ContinuousMove: vel=({vel_pan:+.3f}, {vel_tilt:+.3f})")
            logger.info(f"[TRACKER]   camera id={id(self.camera)}")
            logger.info(f"[TRACKER]   ptz_service id={id(self.camera.ptz_service) if self.camera.ptz_service else 'None'}")
            logger.info(f"[TRACKER]   profile_token={self.camera.profile_token}")

        # Use controller's continuous_move method with 1 second timeout
        # Timeout ensures camera keeps moving even if next command delayed
        result = self.camera.continuous_move(vel_pan, vel_tilt, timeout_sec=1.0, verbose=verbose)

        if total_moves % 15 == 0:
            logger.info(f"[TRACKER] ContinuousMove result: {result}")

        return result

    def _set_state(self, new_state: TrackerState):
        """Update state with logging."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.stats["state_changes"] += 1

            # Enhanced logging with perception state
            perception_info = ""
            if PERCEPTION_AVAILABLE:
                try:
                    ps = get_perception_state()
                    snap = ps.get_snapshot()
                    perception_info = f" | perception: user={snap.user_visible}, face={snap.face_visible}, gesture={snap.gesture_type}"
                except:
                    pass

            logger.info(f"[TRACKER STATE] {old_state.value} -> {new_state.value}{perception_info}")

            if self.on_state_change:
                self.on_state_change(new_state)

    def enable(self):
        """Enable tracking."""
        self.tracking_active = True
        logger.info("Tracking ENABLED")

    def disable(self):
        """Disable tracking and stop movement."""
        self.tracking_active = False
        if self.camera:
            self.camera.stop()
        # Reset dwell state
        self.dwell_target_acquired = False
        self.dwell_start_time = 0.0
        self._set_state(TrackerState.IDLE)
        logger.info("Tracking DISABLED")

    def reset_virtual_position(self, pan: float = 0.0, tilt: float = 0.0):
        """Reset virtual PTZ position (e.g., after manual positioning)."""
        self.virtual_pan = pan
        self.virtual_tilt = tilt
        logger.info(f"Virtual position reset to ({pan:.2f}, {tilt:.2f})")

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        with self._lock:
            detection = self.latest_detection

        return {
            "state": self.state.value,
            "tracking_active": self.tracking_active,
            "running": self._running,
            "current_target": {
                "id": self.current_target_id,
                "confidence": self.current_target_confidence,
                "bbox": self.current_target_bbox
            },
            "latest_detection": {
                "detected": detection.detected,
                "center": (detection.center_x, detection.center_y),
                "confidence": detection.confidence,
                "target_id": detection.target_id,
                "age_ms": int((time.time() - detection.timestamp) * 1000)
            },
            "virtual_position": {
                "pan": self.virtual_pan,
                "tilt": self.virtual_tilt
            },
            "dwell": {
                "target_acquired": self.dwell_target_acquired,
                "elapsed_sec": time.time() - self.dwell_start_time if self.dwell_target_acquired else 0
            },
            "stats": self.stats.copy(),
            "config": {
                "lock_threshold_px": self.config.lock_threshold_pixels,
                "frozen_threshold_px": self.config.frozen_threshold_pixels,
                "dwell_time_sec": self.config.dwell_time_sec,
                "pan_gain": self.config.pan_gain,
                "tilt_gain": self.config.tilt_gain,
                "max_velocity": self.config.max_velocity,
                "min_bbox_height": self.config.min_bbox_height_ratio
            }
        }


# Singleton instance
_tracker: Optional[AutonomousTracker] = None
_tracker_lock = threading.Lock()


def get_autonomous_tracker() -> AutonomousTracker:
    """Get or create singleton tracker instance."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = AutonomousTracker()
    return _tracker
