#!/usr/bin/env python3
"""
M.O.L.O.C.H. Autonomous Tracker v2 - AbsoluteMove
===================================================

Dedicated 5 Hz tracking thread with AbsoluteMove position control.
Implements proportional tracking and search behavior using real camera
position feedback (closed-loop control).

Upgrade from v1 (ContinuousMove):
- AbsoluteMove replaces ContinuousMove (no 90-degree-per-call limit)
- Real camera position via get_position() replaces virtual position tracking
- track_target(error_x, error_y) for proportional position-based tracking
- move_absolute() for search/patrol movements
- Full 342.8 degree pan range utilization

Features:
- 5 Hz tracking loop (200ms cycle)
- AbsoluteMove with proportional position control
- Search mode: patrol sweep when target lost
- Largest bounding box selection with scoring
- Configurable deadzone and gain
- State machine: IDLE, TRACKING, SEARCHING, LOCKED, DWELL, FROZEN

Author: M.O.L.O.C.H. System
Date: 2026-02-08
"""

import time
import math
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

# PTZ Debug Logger - schreibt in ~/moloch/logs/ptz_debug.log
import os as _os
_ptz_log_path = _os.path.expanduser("~/moloch/logs/ptz_debug.log")
_os.makedirs(_os.path.dirname(_ptz_log_path), exist_ok=True)
ptz_debug = logging.getLogger("ptz_debug")
ptz_debug.setLevel(logging.DEBUG)
_ptz_fh = logging.FileHandler(_ptz_log_path, mode="w")
_ptz_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
ptz_debug.addHandler(_ptz_fh)
ptz_debug.propagate = False

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
    TRACKING = "tracking"   # Following target with AbsoluteMove
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
    lock_threshold_pixels: int = 8
    unlock_threshold_pixels: int = 15
    frozen_threshold_pixels: int = 5

    # === Dwell Timer ===
    dwell_time_sec: float = 0.5  # schneller starten (war 1.5)

    # === AbsoluteMove Tracking Parameters ===
    # Kamera Motor-Speed: ~30 deg/s (Kalibrierung: 342deg in ~12s)
    fov_horizontal: float = 110.0
    fov_vertical: float = 65.0
    pan_gain: float = 0.45          # aggressiver (war 0.25)
    tilt_gain: float = 0.40         # aggressiver (war 0.25)
    max_step_pan: float = 12.0      # groessere Schritte (was 3.0)
    max_step_tilt: float = 8.0      # groessere Schritte (was 2.0)
    min_step_deg: float = 0.3
    tracking_speed: float = 1.0     # ONVIF max speed
    move_cooldown_ms: float = 300.0  # schnellere Moves (was 800)
    smooth_alpha: float = 0.5       # schnellere EMA-Reaktion (was 0.2)

    # Search mode parameters
    search_speed: float = 0.3
    search_direction_interval: float = 4.0
    search_reset_to_center: bool = False  # NICHT zurueck auf (0,0) - bleibe wo Tracking war
    search_patrol_positions: list = field(default_factory=lambda: [
        (0.0, 0.0),
        (-84.0, 0.0),
        (-168.0, 0.0),
        (-84.0, 30.0),
        (0.0, 30.0),
        (84.0, 30.0),
        (170.0, 0.0),
        (84.0, 0.0),
    ])

    target_lost_timeout: float = 5.0  # 5s coasting bevor Search (war 2.0)
    frame_width: int = 640
    frame_height: int = 640

    # === Detection filtering ===
    min_bbox_height_ratio: float = 0.40
    max_bbox_center_y_ratio: float = 0.75
    min_bbox_area_ratio: float = 0.08
    min_confidence: float = 0.50
    min_aspect_ratio: float = 0.35

    # === Target persistence ===
    confidence_hysteresis: float = 0.15
    stability_frames: int = 7
    center_priority_weight: float = 0.4

    # === ADAPTIVE TARGET STRATEGY ===
    face_min_confidence: float = 0.55
    face_min_stability: int = 4
    face_max_bbox_height: float = 0.65
    body_min_confidence: float = 0.45
    body_min_bbox_height: float = 0.30
    body_min_stability: int = 5
    switch_cooldown_sec: float = 1.0
    prefer_current_target: bool = True


@dataclass
class DetectionData:
    """Detection data for tracking."""
    detected: bool = False
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    center_x: float = 0.5
    center_y: float = 0.5
    confidence: float = 0.0
    target_id: int = 0
    timestamp: float = field(default_factory=time.time)
    is_pose_detection: bool = False
    has_face: bool = False
    has_torso: bool = False
    head_center_x: float = 0.5
    head_center_y: float = 0.5
    validation_reason: str = ""
    target_type: str = "none"


class AutonomousTracker:
    """
    Autonomous person tracking with 5 Hz control loop.

    Uses AbsoluteMove for position-based tracking with real camera feedback.
    Replaces ContinuousMove (which was limited to 90 degrees per call).
    """

    LOOP_RATE_HZ = 5  # 200ms cycle time

    def __init__(self, camera_controller=None, config: TrackingConfig = None):
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
        self.search_direction = 1
        self.last_direction_switch = 0.0
        self.search_patrol_index = 0
        self.search_move_time = 0.0

        # === Real Camera Position (replaces virtual position) ===
        self.last_known_pan = 0.0
        self.last_known_tilt = 0.0
        self.last_position_time = 0.0
        self.last_move_time = 0.0
        # Anti-Overshoot: letztes Ziel tracken
        self._target_pan = None
        self._target_tilt = None
        self._target_arrival_thresh = 3.0  # Grad - Kamera muss so nah am Ziel sein
        # EMA Glaettung fuer smooth tracking
        self._smooth_x = None
        self._smooth_y = None

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
            "target_switches": 0,
            "position_reads": 0
        }

        # Callbacks
        self.on_state_change: Optional[Callable[[TrackerState], None]] = None

        logger.info(f"AutonomousTracker v2 (AbsoluteMove) initialized (rate={self.LOOP_RATE_HZ}Hz)")

    def set_camera(self, camera_controller):
        """Set camera controller."""
        self.camera = camera_controller
        logger.info(f"Camera controller connected to AutonomousTracker")
        if camera_controller:
            logger.info(f"  Controller id: {id(camera_controller)}")
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
        logger.info("=== TRACKER v2 START: CONTROLLER VERIFICATION ===")
        logger.info(f"self.camera:          {self.camera}")
        logger.info(f"self.camera id:       {id(self.camera)}")
        logger.info(f"is_connected:         {self.camera.is_connected}")
        logger.info(f"mode:                 {self.camera.mode}")
        logger.info("=" * 60)

        # === DIAGNOSTIC: Read initial position ===
        logger.info("=== READING INITIAL CAMERA POSITION ===")
        try:
            pos = self.camera.get_position()
            self.last_known_pan = pos.pan
            self.last_known_tilt = pos.tilt
            self.last_position_time = time.time()
            logger.info(f"Initial position: pan={pos.pan:.1f} deg, tilt={pos.tilt:.1f} deg")
        except Exception as e:
            logger.error(f"Failed to read initial position: {e}")

        # Grace Period: 5s bevor SEARCH (Modelle brauchen Zeit fuer erste Detection)
        self.last_detection_time = time.time()
        self._running = True
        self.tracking_active = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

        logger.info("AutonomousTracker v2 started")
        return True

    def stop(self):
        """Stop the tracking thread."""
        self._running = False
        self.tracking_active = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

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
        """
        with self._lock:
            self.config.frame_width = frame_width
            self.config.frame_height = frame_height
            frame_area = frame_width * frame_height

            if not detections:
                self.latest_detection = DetectionData(detected=False)
                self.candidate_stability_count = 0
                return

            person_dets = [d for d in detections if d.get("class", "") == "person"]
            if not person_dets:
                person_dets = detections

            # === STAGE 1: Filter out invalid detections ===
            valid_dets = []
            for d in person_dets:
                bbox = d.get("bbox", [0, 0, 0, 0])
                conf = d.get("confidence", 0)
                det_class = d.get("class", "person")
                is_face = (det_class == "face")

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height

                if conf < self.config.min_confidence:
                    self.stats["detections_filtered"] += 1
                    continue

                # Face-BBoxen sind viel kleiner als Person-BBoxen -> relaxed thresholds
                min_height = 0.08 if is_face else self.config.min_bbox_height_ratio
                min_area = 0.01 if is_face else self.config.min_bbox_area_ratio

                height_ratio = height / frame_height
                if height_ratio < min_height:
                    self.stats["detections_filtered"] += 1
                    continue

                area_ratio = area / frame_area
                if area_ratio < min_area:
                    self.stats["detections_filtered"] += 1
                    continue

                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < self.config.min_aspect_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

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
                area = (x2 - x1) * (y2 - y1)
                area_score = area / frame_area
                center_x = (x1 + x2) / 2 / frame_width
                center_y = (y1 + y2) / 2 / frame_height
                dist_from_center = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
                center_score = 1.0 - min(1.0, dist_from_center * 2)
                return area_score + (center_score * self.config.center_priority_weight) + (conf * 0.2)

            scored_dets = sorted(valid_dets, key=score_detection, reverse=True)
            best_candidate = scored_dets[0]
            best_bbox = best_candidate.get("bbox", [0, 0, 0, 0])
            best_conf = best_candidate.get("confidence", 0)

            # === STAGE 3: Target persistence with hysteresis ===
            x1, y1, x2, y2 = best_bbox
            center_x = (x1 + x2) / 2 / frame_width
            center_y = (y1 + y2) / 2 / frame_height

            current_target_still_valid = False
            current_target_detection = None
            if self.current_target_id > 0 and self.current_target_confidence > 0:
                for d in valid_dets:
                    bbox = d.get("bbox", [0, 0, 0, 0])
                    if self._bbox_iou(bbox, self.current_target_bbox) > 0.3:
                        current_target_still_valid = True
                        current_target_detection = d
                        self.current_target_bbox = bbox
                        self.current_target_confidence = d.get("confidence", 0)
                        break

            best_alternative = None
            best_alternative_conf = 0.0
            for d in valid_dets:
                conf = d.get("confidence", 0)
                bbox = d.get("bbox", [0, 0, 0, 0])
                if self._bbox_iou(bbox, self.current_target_bbox) < 0.3:
                    if conf > best_alternative_conf:
                        best_alternative_conf = conf
                        best_alternative = d

            should_switch = False
            switch_to_bbox = best_bbox
            switch_to_conf = best_conf

            if not current_target_still_valid:
                should_switch = True
                self.candidate_stability_count = 0
            elif best_alternative and best_alternative_conf > self.current_target_confidence + self.config.confidence_hysteresis:
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

            face_candidates = []
            body_candidates = []

            for p in poses:
                bbox = p.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue

                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                height_ratio = height / frame_height
                area_ratio = (width * height) / (frame_width * frame_height)
                aspect_ratio = width / height if height > 0 else 0

                if area_ratio < self.config.min_bbox_area_ratio:
                    self.stats["detections_filtered"] += 1
                    continue
                if aspect_ratio < self.config.min_aspect_ratio:
                    self.stats["detections_filtered"] += 1
                    continue
                center_y = (bbox[1] + bbox[3]) / 2 / frame_height
                if center_y > self.config.max_bbox_center_y_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                has_face = p.get("has_face", False)
                face_conf = p.get("face_confidence", 0)
                face_center = p.get("face_center")
                has_torso = p.get("has_torso", False)

                if has_face and face_conf >= self.config.face_min_confidence and face_center:
                    if height_ratio <= self.config.face_max_bbox_height:
                        face_candidates.append(p)
                    else:
                        body_candidates.append(p)
                elif has_torso and height_ratio >= self.config.body_min_bbox_height:
                    body_candidates.append(p)
                else:
                    self.stats["detections_filtered"] += 1

            selected_pose = None
            selected_type = TargetType.NONE
            track_x, track_y = 0.5, 0.5

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

            if self.config.prefer_current_target and self.current_target_type != TargetType.NONE:
                cooldown_ok = (now - self.last_target_switch_time) > self.config.switch_cooldown_sec
                if self.current_target_type == TargetType.FACE and face_candidates:
                    face_candidates.sort(key=lambda p: score_pose(p, True), reverse=True)
                    selected_pose = face_candidates[0]
                    selected_type = TargetType.FACE
                elif self.current_target_type == TargetType.BODY and body_candidates:
                    body_candidates.sort(key=lambda p: score_pose(p, False), reverse=True)
                    selected_pose = body_candidates[0]
                    selected_type = TargetType.BODY
                elif cooldown_ok:
                    pass

            if selected_pose is None:
                if face_candidates:
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
                    self.target_type_stability = 0

            if selected_pose is None:
                self.latest_detection = DetectionData(detected=False, is_pose_detection=True)
                return

            self.current_target_type = selected_type

            bbox = selected_pose.get("bbox", [0, 0, 0, 0])
            if selected_type == TargetType.FACE:
                face_center = selected_pose.get("face_center", (0.5, 0.5))
                track_x, track_y = face_center
            else:
                bbox_center_x = (bbox[0] + bbox[2]) / 2 / frame_width
                bbox_center_y = (bbox[1] + bbox[3]) / 2 / frame_height
                track_x = bbox_center_x
                track_y = bbox_center_y * 0.85

            if self.current_target_id == 0:
                self.current_target_id = self._next_target_id
                self._next_target_id += 1

            self.current_target_bbox = bbox
            self.current_target_confidence = selected_pose.get("face_confidence", 0) if selected_type == TargetType.FACE else selected_pose.get("confidence", 0)

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

            if self.stats["cycles"] % 30 == 0:
                logger.info(f"[ADAPTIVE] {selected_type.value.upper()} at ({track_x:.2f},{track_y:.2f}) "
                           f"conf={self.current_target_confidence:.2f} "
                           f"faces={len(face_candidates)} bodies={len(body_candidates)}")

    # =========================================================================
    # Tracking Loop
    # =========================================================================

    def _tracking_loop(self):
        """Main 5 Hz tracking loop."""
        cycle_time = 1.0 / self.LOOP_RATE_HZ
        logger.info(f"Tracking loop STARTED (rate={self.LOOP_RATE_HZ}Hz)")

        while self._running:
            loop_start = time.time()

            try:
                if self.tracking_active:
                    self._process_tracking_cycle()
                    self.stats["cycles"] += 1

                    if self.stats["cycles"] % 15 == 0:
                        logger.info(f"Tracker loop: cycles={self.stats['cycles']} state={self.state.value} "
                                  f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg "
                                  f"search={self.stats['search_moves']} track={self.stats['tracking_moves']}")

            except Exception as e:
                logger.error(f"Tracking cycle error: {e}")
                import traceback
                logger.error(traceback.format_exc())

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

        debug_log = (self.stats["cycles"] % 15 == 0)
        if debug_log:
            logger.info(f"[CYCLE] detected={detection.detected} "
                       f"time_since={time_since_detection:.2f}s "
                       f"conf={detection.confidence:.2f} "
                       f"state={self.state.value} "
                       f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

        if detection.detected and time_since_detection < 0.5:
            self._do_tracking(detection)
        else:
            if time_since_detection > self.config.target_lost_timeout:
                if debug_log:
                    logger.info(f"[CYCLE] No detection for {time_since_detection:.1f}s -> SEARCH")
                self._do_search()
            else:
                if debug_log:
                    logger.info(f"[CYCLE] Brief loss ({time_since_detection:.2f}s) -> COAST")
                self._do_coast()

    # =========================================================================
    # Tracking (AbsoluteMove-based)
    # =========================================================================

    def _do_tracking(self, detection: DetectionData):
        """Execute tracking with dwell timer, proportional position control, and LOCK/FROZEN states."""
        now = time.time()

        # EMA Glaettung: smooth detection center (kein Ruckeln/Springen)
        alpha = self.config.smooth_alpha
        if self._smooth_x is None:
            self._smooth_x = detection.center_x
            self._smooth_y = detection.center_y
        else:
            self._smooth_x = (1 - alpha) * self._smooth_x + alpha * detection.center_x
            self._smooth_y = (1 - alpha) * self._smooth_y + alpha * detection.center_y

        # Calculate error from frame center (pixels) - mit geglaetteten Werten
        center_x_px = self._smooth_x * self.config.frame_width
        center_y_px = self._smooth_y * self.config.frame_height
        frame_center_x = self.config.frame_width / 2
        frame_center_y = self.config.frame_height / 2

        error_x = center_x_px - frame_center_x  # Positive = target RIGHT of center
        error_y = center_y_px - frame_center_y  # Positive = target BELOW center
        error_magnitude = math.sqrt(error_x**2 + error_y**2)

        # Normalized error (-0.5 to +0.5) - geglaettet
        error_x_norm = self._smooth_x - 0.5
        error_y_norm = self._smooth_y - 0.5

        # PTZ Debug: raw + smooth Position + Error bei jedem Cycle
        ptz_debug.debug(
            f"DETECT raw=({detection.center_x:.3f},{detection.center_y:.3f}) "
            f"smooth=({self._smooth_x:.3f},{self._smooth_y:.3f}) "
            f"err=({error_x_norm:+.3f},{error_y_norm:+.3f}) "
            f"cam=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg"
        )

        debug_log = (self.stats["cycles"] % 15 == 0)
        if debug_log:
            logger.info(f"[TRACK] error=({error_x:+.0f},{error_y:+.0f})px mag={error_magnitude:.0f}px "
                       f"state={self.state.value} pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

        # === DWELL STATE: Wait before starting movement ===
        if not self.dwell_target_acquired:
            self.dwell_target_acquired = True
            self.dwell_start_time = now
            self._set_state(TrackerState.DWELL)
            logger.info(f"[DWELL] Target acquired - waiting {self.config.dwell_time_sec}s before tracking")
            return

        if self.state == TrackerState.DWELL:
            dwell_elapsed = now - self.dwell_start_time
            if dwell_elapsed < self.config.dwell_time_sec:
                if debug_log:
                    logger.info(f"[DWELL] Waiting... {dwell_elapsed:.1f}s / {self.config.dwell_time_sec}s")
                return
            else:
                logger.info("[DWELL] Complete - starting tracking")
                self._set_state(TrackerState.TRACKING)

        # === FROZEN STATE: Target perfectly centered ===
        if error_magnitude < self.config.frozen_threshold_pixels:
            if self.state != TrackerState.FROZEN:
                self._set_state(TrackerState.FROZEN)
                logger.info(f"[FROZEN] Target perfectly centered (error={error_magnitude:.0f}px)")
            return

        # === LOCKED STATE WITH HYSTERESIS ===
        if self.state == TrackerState.LOCKED or self.state == TrackerState.FROZEN:
            if error_magnitude > self.config.unlock_threshold_pixels:
                self._set_state(TrackerState.TRACKING)
                if debug_log:
                    logger.info(f"[TRACK] UNLOCK: error {error_magnitude:.0f}px > {self.config.unlock_threshold_pixels}px")
            else:
                if debug_log:
                    logger.info(f"[TRACK] LOCKED: error {error_magnitude:.0f}px (no move)")
                return
        else:
            if error_magnitude < self.config.lock_threshold_pixels:
                self._set_state(TrackerState.LOCKED)
                if debug_log:
                    logger.info(f"[TRACK] LOCK: error {error_magnitude:.0f}px < {self.config.lock_threshold_pixels}px")
                return

        # === TRACKING MODE: AbsoluteMove ===
        self._set_state(TrackerState.TRACKING)

        # Cooldown check
        time_since_move = (now - self.last_move_time) * 1000  # ms
        if time_since_move < self.config.move_cooldown_ms:
            return

        # Anti-Overshoot: warte bis Kamera am letzten Ziel angekommen ist
        if self._target_pan is not None:
            # Cache-Position nutzen (kein ONVIF-Call - spart 100-200ms!)
            dist = abs(self.last_known_pan - self._target_pan) + abs(self.last_known_tilt - self._target_tilt)
            if dist > self._target_arrival_thresh:
                # Timeout: nach 5s aufgeben (Kamera hat Ziel nicht erreicht)
                if not hasattr(self, '_target_wait_start') or self._target_wait_start is None:
                    self._target_wait_start = time.time()
                elif time.time() - self._target_wait_start > 5.0:
                    ptz_debug.warning(
                        f"WAIT TIMEOUT target=({self._target_pan:+.1f},{self._target_tilt:+.1f}) "
                        f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) dist={dist:.1f} - clearing"
                    )
                    self._target_pan = None
                    self._target_tilt = None
                    self._target_wait_start = None
                else:
                    ptz_debug.debug(
                        f"WAIT target=({self._target_pan:+.1f},{self._target_tilt:+.1f}) "
                        f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) dist={dist:.1f}"
                    )
                    return
            else:
                self._target_wait_start = None

        # Calculate position delta from error (pre-check for threshold)
        pan_delta = -error_x_norm * self.config.fov_horizontal * self.config.pan_gain
        tilt_delta = -error_y_norm * self.config.fov_vertical * self.config.tilt_gain
        pan_delta = max(-self.config.max_step_pan, min(self.config.max_step_pan, pan_delta))
        tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))

        if abs(pan_delta) < self.config.min_step_deg and abs(tilt_delta) < self.config.min_step_deg:
            if debug_log:
                logger.info(f"[TRACK] delta below threshold (pan={pan_delta:+.2f}, tilt={tilt_delta:+.2f})")
            return

        # Execute AbsoluteMove tracking
        result = self._track_target(error_x_norm, error_y_norm)

        if result:
            self.stats["tracking_moves"] += 1

        if self.stats["tracking_moves"] % 15 == 0:
            logger.info(f"TRACK: err=({error_x:+.0f},{error_y:+.0f})px delta=({pan_delta:+.1f},{tilt_delta:+.1f})deg "
                       f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

    def _track_target(self, error_x_norm: float, error_y_norm: float) -> bool:
        """
        Track target using AbsoluteMove with real position feedback.

        Converts normalized frame-center error to position delta,
        reads current position, and sends AbsoluteMove to target position.

        Args:
            error_x_norm: Normalized horizontal error (-0.5 to +0.5), positive = target right
            error_y_norm: Normalized vertical error (-0.5 to +0.5), positive = target below

        Returns:
            True if move command sent successfully
        """
        if not self.camera or not self.camera.is_connected:
            return False

        # Check exclusive PTZ lock
        if hasattr(self.camera, '_exclusive_owner') and self.camera._exclusive_owner is not None:
            return False

        # Cache-Position nutzen (kein ONVIF-Call - spart 100-200ms pro Cycle!)
        # last_known_pan/tilt wird nach jedem move_absolute() auf target gesetzt

        # Calculate position delta from error
        # Positive error_x (target right) -> negative pan delta (move right = decrease pan)
        # Positive error_y (target below) -> negative tilt delta (move down = decrease tilt)
        pan_delta = -error_x_norm * self.config.fov_horizontal * self.config.pan_gain
        tilt_delta = -error_y_norm * self.config.fov_vertical * self.config.tilt_gain

        # Limit step size
        pan_delta = max(-self.config.max_step_pan, min(self.config.max_step_pan, pan_delta))
        tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))

        # Calculate target position
        target_pan = self.last_known_pan + pan_delta
        target_tilt = self.last_known_tilt + tilt_delta

        # PTZ Debug: Vollstaendige Berechnung loggen
        face_side = "LINKS" if error_x_norm < 0 else "RECHTS"
        face_vert = "OBEN" if error_y_norm < 0 else "UNTEN"
        cam_pan_dir = "LINKS(+)" if pan_delta > 0 else "RECHTS(-)"
        cam_tilt_dir = "HOCH(+)" if tilt_delta > 0 else "RUNTER(-)"
        ptz_debug.info(
            f"MOVE err_norm=({error_x_norm:+.3f},{error_y_norm:+.3f}) "
            f"Gesicht={face_side}/{face_vert} | "
            f"pan_delta={pan_delta:+.1f} ({cam_pan_dir}) tilt_delta={tilt_delta:+.1f} ({cam_tilt_dir}) | "
            f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) -> "
            f"target=({target_pan:+.1f},{target_tilt:+.1f})deg"
        )

        # SonoffCameraController.move_absolute() clamps to calibrated limits internally
        result = self.camera.move_absolute(target_pan, target_tilt, speed=self.config.tracking_speed)

        if result:
            self.last_move_time = time.time()
            self._target_pan = target_pan
            self._target_tilt = target_tilt
            # Cache sofort auf Zielposition setzen (kein ONVIF noetig)
            self.last_known_pan = target_pan
            self.last_known_tilt = target_tilt

            total_moves = self.stats["tracking_moves"] + self.stats["search_moves"]
            if total_moves % 15 == 0:
                logger.info(f"[TRACKER] AbsoluteMove: pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) "
                           f"-> target=({target_pan:+.1f},{target_tilt:+.1f})deg")

        return result

    # =========================================================================
    # Search Mode (AbsoluteMove patrol)
    # =========================================================================

    def _do_search(self):
        """Execute search mode - patrol sweep using AbsoluteMove positions."""
        # Smoothing zuruecksetzen wenn Ziel verloren
        self._smooth_x = None
        self._smooth_y = None
        if PERCEPTION_AVAILABLE:
            try:
                if is_user_visible():
                    if self.state == TrackerState.SEARCHING:
                        logger.info("[SEARCH] Aborted: user_visible=True in perception")
                    self._do_coast()
                    return
            except:
                pass

        now = time.time()

        # === START SEARCH: Reset and begin patrol ===
        if self.state != TrackerState.SEARCHING:
            self._set_state(TrackerState.SEARCHING)
            self.search_patrol_index = 0
            self.search_move_time = 0.0

            if self.config.search_reset_to_center:
                logger.info(f"[SEARCH] Starting - moving to center (was pan={self.last_known_pan:+.1f}, tilt={self.last_known_tilt:+.1f})")
                self._send_search_move(0.0, 0.0)
            else:
                logger.info(f"[SEARCH] Starting - staying at last pos ({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})")
                self.search_move_time = now  # Timer starten, aber NICHT bewegen
            # Reset dwell state for next target acquisition
            self.dwell_target_acquired = False
            self.dwell_start_time = 0.0
            return

        # === KEIN PATROL: Kamera bleibt an letzter Position stehen ===
        # Smart Tracking uebernimmt nach Release ab dieser Position

    def _send_search_move(self, pan_deg: float, tilt_deg: float) -> bool:
        """Send AbsoluteMove for search/patrol."""
        if not self.camera or not self.camera.is_connected:
            return False

        if hasattr(self.camera, '_exclusive_owner') and self.camera._exclusive_owner is not None:
            return False

        result = self.camera.move_absolute(pan_deg, tilt_deg, speed=self.config.search_speed)

        if result:
            self.search_move_time = time.time()
            self.last_move_time = time.time()
            self.stats["search_moves"] += 1

        return result

    def _do_coast(self):
        """Coast - stop movement when target briefly lost."""
        # With AbsoluteMove, camera naturally stops at the last commanded position.
        # Only send explicit stop if transitioning from a search/patrol state.
        if self.state == TrackerState.SEARCHING:
            if self.camera:
                self.camera.stop()

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, new_state: TrackerState):
        """Update state with logging."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.stats["state_changes"] += 1

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
        self.dwell_target_acquired = False
        self.dwell_start_time = 0.0
        self._set_state(TrackerState.IDLE)
        logger.info("Tracking DISABLED")

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
            "camera_position": {
                "pan_deg": self.last_known_pan,
                "tilt_deg": self.last_known_tilt,
                "position_age_ms": int((time.time() - self.last_position_time) * 1000) if self.last_position_time > 0 else -1
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
                "fov_h": self.config.fov_horizontal,
                "fov_v": self.config.fov_vertical,
                "max_step_pan": self.config.max_step_pan,
                "max_step_tilt": self.config.max_step_tilt,
                "tracking_speed": self.config.tracking_speed,
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
