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
    COAST = "coast"         # Ziel stabil seit 2s, Kamera komplett eingefroren


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
    pan_gain: float = 0.25          # sanft, kein Ueberschwinger (war 0.45 = zu aggressiv)
    tilt_gain: float = 0.20         # sanft (war 0.40 = zu aggressiv)
    max_step_pan: float = 5.0       # max 5 Grad pro Move (war 12.0 = VIEL zu viel)
    max_step_tilt: float = 3.0      # max 3 Grad pro Move (war 8.0 = zu viel)
    min_step_deg: float = 0.3
    tracking_speed: float = 0.7     # 70% ONVIF Speed, nie Vollgas (war 1.0)
    move_cooldown_ms: float = 400.0  # 400ms zwischen Moves (war 300, vorher 800)
    smooth_alpha: float = 0.20      # EMA etwas schneller fuer bessere Reaktion (war 0.15)

    # Kamera Hardware-Limits (SonoffCameraController clampt intern,
    # aber Tracker muss gecachte Position AUCH clampen!)
    pan_limit_min: float = -168.4
    pan_limit_max: float = 170.0
    tilt_limit_min: float = -78.0
    tilt_limit_max: float = 78.8

    # Search mode parameters
    search_speed: float = 0.3
    search_direction_interval: float = 4.0
    search_reset_to_center: bool = False
    search_patrol_positions: list = field(default_factory=lambda: [
        (0.0, 0.0),        # Home (Markus' Sitzplatz)
        (-60.0, 0.0),      # Leicht links
        (-120.0, 0.0),     # Weiter links
        (0.0, 20.0),       # Mitte hoch
        (60.0, 0.0),       # Leicht rechts
        (120.0, 0.0),      # Weiter rechts
    ])
    search_home_timeout: float = 30.0   # 30s ohne Fund -> Home (war 60s)

    target_lost_timeout: float = 5.0  # 5s coasting bevor Search
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

    # === SMOOTHING: Face/Person Wechsel-Daempfung ===
    source_hysteresis_frames: int = 3   # Neuer Source-Typ muss 3 Frames stabil sein
    center_ring_buffer_size: int = 10   # Mittelwert ueber letzte 10 Frame-Zentren (war 5)
    min_frames_before_move: int = 3     # Mindestens 3 Frames im Buffer bevor Kamera bewegt

    # === DEAD ZONE + COAST MODE (Tracker-Beruhigung) ===
    dead_zone_pct: float = 0.15        # 15% - mittlere 30% des Bildes = RUHIG (war 3%)
    track_start_pct: float = 0.18      # 18% - erst ab hier Tracking starten (war 5%)
    coast_stable_time: float = 1.5     # 1.5s stabil im Dead Zone -> Coast (war 2.0)
    coast_resume_pct: float = 0.12     # 12% Abweichung zum Aufwachen aus Coast (war 5%)
    min_move_speed: float = 0.15       # Minimale ONVIF-Speed bei kleinen Korrekturen


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

        # === SMOOTHING: Source-Typ Hysterese + Ring-Buffer ===
        self._current_source = "none"       # "face" oder "body" - aktiver Source-Typ
        self._candidate_source = "none"     # Anwaerter-Source
        self._source_stability = 0          # Frames stabil beim neuen Source
        self._center_ring = []              # Ring-Buffer: [(cx, cy), ...] letzte N Frames

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
        # PD-Regler: vorherigen Fehler speichern fuer Derivative (Bremse)
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._prev_error_time = 0.0

        # === COAST MODE: Kamera einfrieren wenn Ziel stabil ===
        self._stable_start_time = None    # Wann wurde Ziel zuletzt stabil (fuer Coast-Timer)

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

        # Core Integrator Referenz (fuer adaptive Tracking-Parameter)
        self._core_integrator = None
        try:
            from core.core_integrator import get_core_integrator
            self._core_integrator = get_core_integrator()
            logger.info("[TRACKER] CoreIntegrator angebunden")
        except Exception as e:
            logger.warning(f"[TRACKER] CoreIntegrator nicht verfuegbar: {e}")

        # Basis-Parameter speichern (fuer dynamische Anpassung)
        self._base_pan_gain = self.config.pan_gain
        self._base_tilt_gain = self.config.tilt_gain
        self._base_max_step_pan = self.config.max_step_pan
        self._base_max_step_tilt = self.config.max_step_tilt
        self._base_move_cooldown = self.config.move_cooldown_ms
        self._base_tracking_speed = self.config.tracking_speed
        self._base_target_lost_timeout = self.config.target_lost_timeout

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
        - Face hat IMMER Prioritaet vor Person-Box
        - Source-Typ Hysterese: Wechsel erst nach N stabilen Frames
        - Ring-Buffer Smoothing: Tracking-Zentrum ueber letzte 5 Frames gemittelt
        - Filters out hands, partial bodies, low-confidence detections
        - Maintains current target if still valid
        """
        with self._lock:
            self.config.frame_width = frame_width
            self.config.frame_height = frame_height
            frame_area = frame_width * frame_height

            if not detections:
                self.latest_detection = DetectionData(detected=False)
                self.candidate_stability_count = 0
                return

            # === Face-Prioritaet: Face und Person getrennt behandeln ===
            face_dets = [d for d in detections if d.get("class", "") == "face"]
            person_dets = [d for d in detections if d.get("class", "") == "person"]
            incoming_source = "none"

            # Face hat IMMER Vorrang
            if face_dets:
                work_dets = face_dets
                incoming_source = "face"
            elif person_dets:
                work_dets = person_dets
                incoming_source = "body"
            else:
                work_dets = detections
                incoming_source = "body"

            # === Source-Typ Hysterese ===
            if incoming_source != self._current_source:
                if incoming_source == self._candidate_source:
                    self._source_stability += 1
                else:
                    self._candidate_source = incoming_source
                    self._source_stability = 1

                # Wechsel erst nach N stabilen Frames (AUSNAHME: face -> sofort)
                if incoming_source == "face":
                    # Face hat sofort Prioritaet - kein Warten
                    self._current_source = "face"
                    self._source_stability = 0
                    self._candidate_source = "none"
                elif self._source_stability >= self.config.source_hysteresis_frames:
                    logger.info(f"[SMOOTH] Source-Wechsel: {self._current_source} -> {incoming_source} "
                               f"(stabil seit {self._source_stability} Frames)")
                    self._current_source = incoming_source
                    self._source_stability = 0
                    self._candidate_source = "none"
                else:
                    # Noch nicht stabil genug - aktuelle Quelle beibehalten, KEIN Update
                    return
            else:
                self._source_stability = 0
                self._candidate_source = "none"

            # === STAGE 1: Filter out invalid detections ===
            valid_dets = []
            for d in work_dets:
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

            # === STAGE 4: Ring-Buffer Smoothing (letzte N Frames mitteln) ===
            buf_size = self.config.center_ring_buffer_size
            self._center_ring.append((center_x, center_y))
            if len(self._center_ring) > buf_size:
                self._center_ring = self._center_ring[-buf_size:]

            # Gewichteter Mittelwert: neuere Frames zaehlen mehr
            if len(self._center_ring) > 1:
                weights = list(range(1, len(self._center_ring) + 1))
                w_sum = sum(weights)
                smooth_cx = sum(w * c[0] for w, c in zip(weights, self._center_ring)) / w_sum
                smooth_cy = sum(w * c[1] for w, c in zip(weights, self._center_ring)) / w_sum
            else:
                smooth_cx = center_x
                smooth_cy = center_y

            self.latest_detection = DetectionData(
                detected=True,
                bbox=self.current_target_bbox,
                center_x=smooth_cx,
                center_y=smooth_cy,
                confidence=self.current_target_confidence,
                target_id=self.current_target_id,
                timestamp=time.time(),
                target_type=self._current_source
            )
            self.last_detection_time = time.time()

            # CoreIntegrator: Target gefunden -> Presence/Proximity steigt
            if self._core_integrator:
                try:
                    self._core_integrator.update_inputs("tracker", {
                        "user_proximity": self.current_target_confidence,
                        "time_since_interaction": 0.0,  # Reset: Ziel gerade gesehen
                    })
                except Exception:
                    pass

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
                # Body-Tracking: Oberes Drittel der Person-Box anpeilen (Kopfhoehe)
                # statt Mitte der Box (war center_y * 0.85 = kaum Korrektur)
                bbox_center_x = (bbox[0] + bbox[2]) / 2 / frame_width
                bbox_top_y = bbox[1] / frame_height
                bbox_bottom_y = bbox[3] / frame_height
                bbox_height = bbox_bottom_y - bbox_top_y
                # Ziel: 30% von oben in der Person-Box (= Kopf/Schulterbereich)
                track_x = bbox_center_x
                track_y = bbox_top_y + bbox_height * 0.30

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
    # Core Integrator Adaption
    # =========================================================================

    def _adapt_from_integrator(self):
        """Tracking-Parameter dynamisch an CoreIntegrator State anpassen.

        Hohe Attention (> 0.8): Kamera ruhig, praezises Tracking, wenig Bewegung
        Niedrige Attention (< 0.3): Kamera unruhiger, mehr Scan-Bewegungen
        Hohe Tension: Schnellere Reaktion, aggressiveres Tracking
        Niedrige Tension: Sanftere Bewegungen, mehr Coast-Zeit
        """
        if not self._core_integrator:
            return

        try:
            effects = self._core_integrator.get_effects()
            attention = self._core_integrator.get_attention()
            tension = self._core_integrator.get_tension()
        except Exception:
            return

        camera_stability = effects.get("camera_stability", 0.5)
        micro_ptz = effects.get("micro_ptz_movement", 0.2)

        # --- Attention-basierte Anpassung ---
        # Hohe Attention -> ruhigere Kamera (kleinere Schritte, laengere Cooldowns)
        # Niedrige Attention -> unruhigere Kamera (groessere Schritte, kuerzere Cooldowns)
        stability_factor = camera_stability  # 0.0 (unruhig) - 1.0 (stabil)

        # Pan/Tilt Gain: Bei hoher Stabilitaet weniger aggressiv
        self.config.pan_gain = self._base_pan_gain * (0.5 + (1.0 - stability_factor) * 0.8)
        self.config.tilt_gain = self._base_tilt_gain * (0.5 + (1.0 - stability_factor) * 0.8)

        # Max Step: Bei hoher Stabilitaet kleinere Schritte
        self.config.max_step_pan = self._base_max_step_pan * (0.4 + (1.0 - stability_factor) * 0.8)
        self.config.max_step_tilt = self._base_max_step_tilt * (0.4 + (1.0 - stability_factor) * 0.8)

        # --- Tension-basierte Anpassung ---
        # Hohe Tension -> schnellere Reaktion
        tension_speed = 0.7 + tension * 0.6  # 0.7 (ruhig) bis 1.3 (hektisch)

        # Move Cooldown: Bei hoher Tension kuerzere Pausen
        self.config.move_cooldown_ms = self._base_move_cooldown / tension_speed

        # Tracking Speed: Bei hoher Tension schneller
        self.config.tracking_speed = min(1.0, self._base_tracking_speed * tension_speed)

        # Target Lost Timeout: Bei hoher Tension schneller in Search
        self.config.target_lost_timeout = self._base_target_lost_timeout / tension_speed

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
                    # Alle 5 Zyklen (~1x/s): Parameter vom CoreIntegrator anpassen
                    if self.stats["cycles"] % 5 == 0:
                        self._adapt_from_integrator()

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

        # === Minimum Frames: Erst bewegen wenn genug Daten im Ring-Buffer ===
        if len(self._center_ring) < self.config.min_frames_before_move:
            if debug_log:
                logger.info(f"[TRACK] Warte auf Frames: {len(self._center_ring)}/{self.config.min_frames_before_move}")
            return

        # === Error-Magnitude als Prozent vom Bild (fuer Dead Zone / Coast) ===
        error_magnitude_pct = math.sqrt(error_x_norm**2 + error_y_norm**2)

        # === COAST MODE: Kamera komplett eingefroren wenn Ziel stabil ===
        if self.state == TrackerState.COAST:
            if error_magnitude_pct > self.config.coast_resume_pct:
                # Ziel hat sich signifikant bewegt -> Tracking aufnehmen
                self._set_state(TrackerState.TRACKING)
                self._stable_start_time = None
                logger.info(f"[COAST] Aufgewacht! error={error_magnitude_pct:.3f} > {self.config.coast_resume_pct}")
            else:
                # Stabil -> nichts tun
                if debug_log:
                    ptz_debug.debug(f"COAST still error={error_magnitude_pct:.3f} < {self.config.coast_resume_pct}")
                return

        # === DEAD ZONE: < 3% vom Bildzentrum -> keine Kamerabewegung ===
        if error_magnitude_pct < self.config.dead_zone_pct:
            if self.state not in (TrackerState.FROZEN, TrackerState.COAST):
                self._set_state(TrackerState.FROZEN)
                ptz_debug.debug(f"FROZEN dead_zone error={error_magnitude_pct:.3f} < {self.config.dead_zone_pct}")

            # Coast-Timer: stabil seit wann?
            if self._stable_start_time is None:
                self._stable_start_time = now
            elif (now - self._stable_start_time) >= self.config.coast_stable_time:
                # 2+ Sekunden stabil im Dead Zone -> COAST Mode
                self._set_state(TrackerState.COAST)
                logger.info(f"[COAST] Aktiviert - Ziel stabil seit {self.config.coast_stable_time:.1f}s")
            return

        # === HYSTERESE ZONE: 3-5% -> nur bewegen wenn bereits TRACKING ===
        if error_magnitude_pct < self.config.track_start_pct:
            if self.state in (TrackerState.FROZEN, TrackerState.LOCKED, TrackerState.COAST):
                # In der Hysteresezone bleiben wir still
                if debug_log:
                    ptz_debug.debug(
                        f"HYSTERESIS error={error_magnitude_pct:.3f} "
                        f"zone=[{self.config.dead_zone_pct},{self.config.track_start_pct}]"
                    )
                return
            # Wenn bereits TRACKING -> weitermachen (weiche Abbremsung)

        # Aus Dead Zone raus -> Timer zuruecksetzen
        self._stable_start_time = None

        # === TRACKING MODE: AbsoluteMove ===
        self._set_state(TrackerState.TRACKING)

        # Cooldown check
        time_since_move = (now - self.last_move_time) * 1000  # ms
        if time_since_move < self.config.move_cooldown_ms:
            return

        # Anti-Overshoot: warte bis Kamera am letzten Ziel angekommen ist
        if self._target_pan is not None:
            dist = abs(self.last_known_pan - self._target_pan) + abs(self.last_known_tilt - self._target_tilt)
            if dist > self._target_arrival_thresh:
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

        # === PD-REGLER: Proportional + Derivative (Bremse) ===
        # P-Term: proportional zum Fehler
        pan_p = -error_x_norm * self.config.fov_horizontal * self.config.pan_gain
        tilt_p = -error_y_norm * self.config.fov_vertical * self.config.tilt_gain

        # D-Term: Bremse wenn Fehler ABNIMMT (Kamera naehert sich dem Ziel)
        d_gain = 0.4  # Daempfungsfaktor (0 = kein Bremsen, 1 = starkes Bremsen)
        dt = now - self._prev_error_time if self._prev_error_time > 0 else 0.2
        if dt > 0 and dt < 2.0:  # Nur bei plausiblem Zeitintervall
            d_error_x = (error_x_norm - self._prev_error_x) / dt
            d_error_y = (error_y_norm - self._prev_error_y) / dt
            # D-Term: Aenderungsrate * Gain -> bremst ab wenn Fehler schrumpft
            pan_d = -d_error_x * self.config.fov_horizontal * d_gain * self.config.pan_gain
            tilt_d = -d_error_y * self.config.fov_vertical * d_gain * self.config.tilt_gain
        else:
            pan_d = 0.0
            tilt_d = 0.0

        # Fehler-Historie aktualisieren
        self._prev_error_x = error_x_norm
        self._prev_error_y = error_y_norm
        self._prev_error_time = now

        # PD-Summe: P bringt uns zum Ziel, D bremst beim Ankommen
        pan_delta = pan_p + pan_d
        tilt_delta = tilt_p + tilt_d
        pan_delta = max(-self.config.max_step_pan, min(self.config.max_step_pan, pan_delta))
        tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))

        if abs(pan_delta) < self.config.min_step_deg and abs(tilt_delta) < self.config.min_step_deg:
            if debug_log:
                logger.info(f"[TRACK] delta below threshold (pan={pan_delta:+.2f}, tilt={tilt_delta:+.2f})")
            return

        # === ADAPTIVE SPEED: proportional zum Error ===
        # Kleine Korrekturen (15-20%) -> langsam, grosse (>30%) -> volle Speed
        speed_range = self.config.tracking_speed - self.config.min_move_speed
        move_speed = self.config.min_move_speed + speed_range * min(1.0, error_magnitude_pct / 0.30)

        # Execute AbsoluteMove tracking (mit bereits berechnetem PD-Delta)
        result = self._track_target(error_x_norm, error_y_norm, speed=move_speed,
                                     pd_pan_delta=pan_delta, pd_tilt_delta=tilt_delta)

        if result:
            self.stats["tracking_moves"] += 1

        if self.stats["tracking_moves"] % 15 == 0:
            logger.info(f"TRACK: err=({error_x:+.0f},{error_y:+.0f})px err_pct={error_magnitude_pct:.3f} "
                       f"speed={move_speed:.2f} delta=({pan_delta:+.1f},{tilt_delta:+.1f})deg "
                       f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

    def _track_target(self, error_x_norm: float, error_y_norm: float, speed: float = None,
                       pd_pan_delta: float = None, pd_tilt_delta: float = None) -> bool:
        """
        Track target using AbsoluteMove with real position feedback.

        Args:
            error_x_norm: Normalized horizontal error (-0.5 to +0.5), positive = target right
            error_y_norm: Normalized vertical error (-0.5 to +0.5), positive = target below
            speed: ONVIF move speed (0.0-1.0), None = config.tracking_speed
            pd_pan_delta: Vorberechnetes PD-Delta fuer Pan (optional)
            pd_tilt_delta: Vorberechnetes PD-Delta fuer Tilt (optional)

        Returns:
            True if move command sent successfully
        """
        if not self.camera or not self.camera.is_connected:
            return False

        # Check exclusive PTZ lock
        if hasattr(self.camera, '_exclusive_owner') and self.camera._exclusive_owner is not None:
            return False

        # PD-Delta nutzen wenn vorhanden, sonst reiner P-Term als Fallback
        if pd_pan_delta is not None and pd_tilt_delta is not None:
            pan_delta = pd_pan_delta
            tilt_delta = pd_tilt_delta
        else:
            pan_delta = -error_x_norm * self.config.fov_horizontal * self.config.pan_gain
            tilt_delta = -error_y_norm * self.config.fov_vertical * self.config.tilt_gain
            pan_delta = max(-self.config.max_step_pan, min(self.config.max_step_pan, pan_delta))
            tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))

        # Calculate target position + Clamping auf Hardware-Limits
        # (verhindert Position-Drift wenn Kamera intern clampt)
        target_pan = max(self.config.pan_limit_min, min(self.config.pan_limit_max, self.last_known_pan + pan_delta))
        target_tilt = max(self.config.tilt_limit_min, min(self.config.tilt_limit_max, self.last_known_tilt + tilt_delta))

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

        # Adaptive Speed: uebergeben oder Default
        move_speed = speed if speed is not None else self.config.tracking_speed

        # SonoffCameraController.move_absolute() clamps to calibrated limits internally
        result = self.camera.move_absolute(target_pan, target_tilt, speed=move_speed)

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
        """Execute search mode - patrol sweep using AbsoluteMove positions.

        Search Pattern: Home -> Links -> Rechts -> Home -> Hoch -> Runter -> Home
        Nach 60s ohne Fund: Zurueck zu Home, Presence sinkt.
        CoreIntegrator wird ueber Such-Status informiert.
        """
        # Smoothing + Coast-Timer + PD-State zuruecksetzen wenn Ziel verloren
        self._smooth_x = None
        self._smooth_y = None
        self._stable_start_time = None
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._prev_error_time = 0.0
        if PERCEPTION_AVAILABLE:
            try:
                if is_user_visible():
                    if self.state == TrackerState.SEARCHING:
                        logger.info("[SEARCH] Aborted: user_visible=True in perception")
                    self._do_coast()
                    return
            except Exception:
                pass

        now = time.time()

        # === START SEARCH: Reset and begin patrol ===
        if self.state != TrackerState.SEARCHING:
            self._set_state(TrackerState.SEARCHING)
            self.search_patrol_index = 0
            self.search_move_time = now
            self._search_start_time = now

            # Erste Position ansteuern
            logger.info(f"[SEARCH] Patrol gestartet ab ({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})")

            # Reset dwell state for next target acquisition
            self.dwell_target_acquired = False
            self.dwell_start_time = 0.0

            # CoreIntegrator: Presence sinkt beim Suchen
            if self._core_integrator:
                try:
                    self._core_integrator.update_input("tracker", "time_since_interaction", 0.5)
                except Exception:
                    pass
            return

        # === PATROL: Definierte Positionen abfahren ===
        search_duration = now - getattr(self, '_search_start_time', now)
        patrol_positions = self.config.search_patrol_positions

        # Nach 30s ohne Fund: Zurueck zu Home und aufhoeren
        if search_duration > self.config.search_home_timeout:
            if self.search_patrol_index != 0 or self.search_move_time == 0:
                logger.info(f"[SEARCH] {self.config.search_home_timeout:.0f}s ohne Fund -> Home-Position")
                self._send_search_move(0.0, 0.0)
                self.search_patrol_index = 0
                # CoreIntegrator: Presence sinkt stark
                if self._core_integrator:
                    try:
                        self._core_integrator.update_input("tracker", "time_since_interaction", 1.0)
                        self._core_integrator.update_input("tracker", "user_proximity", 0.0)
                    except Exception:
                        pass
            return

        # Patrol-Position wechseln alle 4 Sekunden
        time_at_position = now - self.search_move_time
        if time_at_position >= self.config.search_direction_interval:
            # Naechste Patrol-Position
            self.search_patrol_index = (self.search_patrol_index + 1) % len(patrol_positions)
            target_pan, target_tilt = patrol_positions[self.search_patrol_index]

            logger.info(f"[SEARCH] Patrol [{self.search_patrol_index}/{len(patrol_positions)}] "
                       f"-> ({target_pan:+.1f},{target_tilt:+.1f}) "
                       f"(Search seit {search_duration:.0f}s)")

            self._send_search_move(target_pan, target_tilt)

            # CoreIntegrator: Presence sinkt langsam beim Suchen
            if self._core_integrator:
                try:
                    decay = min(1.0, search_duration / 60.0)  # 0->1 ueber 60s
                    self._core_integrator.update_input("tracker", "time_since_interaction", decay)
                except Exception:
                    pass

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
                except Exception:
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
