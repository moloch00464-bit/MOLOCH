import os
#!/usr/bin/env python3
"""
SonoffCameraController v2 - AbsoluteMove Tracking
==================================================

Unified camera controller for Sonoff GK-200MP2-B (Pan-Tilt 2)
Uses AbsoluteMove for full 342.8 degree pan range access.

Architecture: Vision -> MPO -> Controller -> Camera

Calibrated Limits (manually verified 2026-02-04):
- Pan:  -168.4 deg (right) to +170.0 deg (left) = 338.4 deg total
- Tilt: -78.0 deg (down) to +78.8 deg (up) = 156.8 deg total

Features:
- AbsoluteMove-based tracking (replaces velocity-based ContinuousMove)
- Full 342.8 degree pan range utilization
- Position feedback via ONVIF GetStatus
- Proportional tracking with configurable gain
- 360 degree patrol scan when target lost
- Safety modes: AUTONOMOUS, MANUAL_OVERRIDE, SAFE_MODE

Author: M.O.L.O.C.H. System
"""

import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, Callable
from datetime import datetime
import threading

# ONVIF imports
try:
    from onvif import ONVIFCamera
    from zeep.helpers import serialize_object
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False
    print("WARNING: onvif-zeep not installed. Run: pip install onvif-zeep")

# PTZ calibration (lazy import to avoid circular dependency)
def _load_calibrated_limits():
    """Load calibrated PTZ limits from config file."""
    try:
        from core.hardware.ptz_calibration import get_ptz_limits
        return get_ptz_limits()
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load PTZ limits: {e}")
        return None


class ControlMode(Enum):
    """Camera control modes."""
    AUTONOMOUS = auto()      # Full AI control
    MANUAL_OVERRIDE = auto() # User has control
    SAFE_MODE = auto()       # Minimal movement, logging only


class TrackingState(Enum):
    """Person tracking states."""
    IDLE = auto()           # No target
    TRACKING = auto()       # Following target
    SEARCHING = auto()      # Lost target, scanning
    PATROL = auto()         # 360 degree patrol scan
    LOCKED = auto()         # Target centered and stable


@dataclass
class Detection:
    """Detection result from perception system."""
    person_id: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center_x: float                   # Normalized 0-1
    center_y: float                   # Normalized 0-1
    confidence: float
    face_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PTZPosition:
    """Current PTZ position."""
    pan: float = 0.0      # Degrees
    tilt: float = 0.0     # Degrees
    zoom: float = 0.0     # 0-1 normalized
    moving: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibrationData:
    """PTZ calibration data (verified 2026-02-04)."""
    pan_min: float = -168.4   # Right limit (verified)
    pan_max: float = 170.0    # Left limit (verified)
    tilt_min: float = -78.0   # Down limit (verified)
    tilt_max: float = 78.8    # Up limit (verified)
    pan_speed: float = 20.0   # Degrees per second (measured)
    tilt_speed: float = 12.0  # Degrees per second (measured)
    verified: bool = True
    timestamp: float = field(default_factory=time.time)


class SonoffCameraController:
    """
    Unified camera controller for Sonoff Pan-Tilt camera.

    Uses AbsoluteMove for tracking to access full 342.8 degree pan range.
    ContinuousMove is limited to 90 degrees per call (firmware limitation).
    """

    # Hardware limits (calibrated 2026-02-04)
    HARDWARE_PAN_MIN = -168.4   # Degrees (RECHTS, verified)
    HARDWARE_PAN_MAX = 170.0    # Degrees (LINKS, verified)
    HARDWARE_TILT_MIN = -78.0   # Degrees (down, verified)
    HARDWARE_TILT_MAX = 78.8    # Degrees (up, verified)

    # Soft limits (same as hardware after calibration)
    SOFT_PAN_MIN = -168.4   # RECHTS
    SOFT_PAN_MAX = 170.0    # LINKS
    SOFT_TILT_MIN = -78.0
    SOFT_TILT_MAX = 78.8

    # Tracking parameters
    DEADZONE = 0.05           # 5% of frame - no movement if within
    TRACKING_GAIN_PAN = 0.7   # Proportional gain for pan
    TRACKING_GAIN_TILT = 0.5  # Proportional gain for tilt
    MAX_STEP_PAN = 30.0       # Max degrees per tracking step
    MAX_STEP_TILT = 20.0      # Max degrees per tracking step
    TARGET_LOST_TIMEOUT = 10.0  # Seconds before starting patrol

    # Safety parameters
    COOLDOWN_MS = 50         # Minimum ms between moves
    POSITION_TOLERANCE = 1.0  # Degrees - considered "at position"

    # Patrol scan parameters
    PATROL_POSITIONS = [
        (0.0, 0.0),           # Center
        (-84.0, 0.0),         # Half-right
        (-168.0, 0.0),        # Full right
        (-84.0, 30.0),        # Half-right, up
        (0.0, 30.0),          # Center, up
        (84.0, 30.0),         # Half-left, up
        (174.0, 0.0),         # Full left
        (84.0, 0.0),          # Half-left
    ]

    def __init__(
        self,
        camera_ip: str = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP"),
        username: str = os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME"),
        password: str = os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME"),
        onvif_port: int = 80,
        log_level: int = logging.INFO
    ):
        """Initialize camera controller."""
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.onvif_port = onvif_port

        # Logging
        self.logger = logging.getLogger("SonoffCamera")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)

        # State
        self.mode = ControlMode.AUTONOMOUS
        self.tracking_state = TrackingState.IDLE
        self.current_position = PTZPosition()
        self.calibration = CalibrationData()

        # ONVIF
        self.camera: Optional[ONVIFCamera] = None
        self.ptz_service = None
        self.media_service = None
        self.profile_token: Optional[str] = None

        # Tracking state
        self.last_detection: Optional[Detection] = None
        self.last_detection_time: float = 0
        self.last_move_time: float = 0
        self.patrol_index: int = 0

        # Threading
        self._lock = threading.Lock()
        self._connected = False

        # Exclusive PTZ access (for Vision Lab etc.)
        self._exclusive_owner: Optional[str] = None
        self._exclusive_lock = threading.Lock()

        # Callbacks
        self.on_position_update: Optional[Callable[[PTZPosition], None]] = None
        self.on_state_change: Optional[Callable[[TrackingState], None]] = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to camera via ONVIF."""
        if not ONVIF_AVAILABLE:
            self.logger.error("ONVIF not available")
            return False

        try:
            self.logger.info(f"Connecting to camera at {self.camera_ip}...")

            self.camera = ONVIFCamera(
                self.camera_ip,
                self.onvif_port,
                self.username,
                self.password
            )

            # Get services
            self.media_service = self.camera.create_media_service()
            self.ptz_service = self.camera.create_ptz_service()

            # Get profile
            profiles = self.media_service.GetProfiles()
            if profiles:
                self.profile_token = profiles[0].token
                self.logger.info(f"Using profile: {self.profile_token}")
            else:
                self.logger.error("No media profiles found")
                return False

            # Get initial position
            self._update_position()

            # Load calibrated limits from config
            limits = _load_calibrated_limits()
            if limits and limits.calibrated:
                self.calibration.pan_min = limits.pan_min
                self.calibration.pan_max = limits.pan_max
                self.calibration.tilt_min = limits.tilt_min
                self.calibration.tilt_max = limits.tilt_max
                self.calibration.verified = True
                self.logger.info(f"Loaded CALIBRATED limits: pan=[{limits.pan_min:.1f}, {limits.pan_max:.1f}], "
                               f"tilt=[{limits.tilt_min:.1f}, {limits.tilt_max:.1f}]")
            else:
                self.logger.info("Using default PTZ limits (not calibrated)")

            self._connected = True
            self.logger.info("Camera connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera."""
        self.stop()
        self._connected = False
        self.camera = None
        self.ptz_service = None
        self.logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.camera is not None

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    def _update_position(self) -> Optional[PTZPosition]:
        """Get current position from camera."""
        if not self.ptz_service or not self.profile_token:
            return None

        try:
            status = self.ptz_service.GetStatus({'ProfileToken': self.profile_token})
            status_dict = serialize_object(status)

            pos = status_dict.get('Position', {})
            pan_tilt = pos.get('PanTilt', {})

            # ONVIF returns normalized -1 to 1
            pan_norm = float(pan_tilt.get('x', 0))
            tilt_norm = float(pan_tilt.get('y', 0))

            # Convert to degrees using calibration
            pan_deg = float(pan_tilt.get("x", 0))  # Direct degrees
            tilt_deg = float(pan_tilt.get("y", 0))  # Direct degrees

            # Check if moving
            move_status = status_dict.get('MoveStatus', {})
            pan_moving = move_status.get('PanTilt', 'IDLE') != 'IDLE'

            self.current_position = PTZPosition(
                pan=pan_deg,
                tilt=tilt_deg,
                zoom=float(pos.get('Zoom', {}).get('x', 0)),
                moving=pan_moving,
                timestamp=time.time()
            )

            if self.on_position_update:
                self.on_position_update(self.current_position)

            return self.current_position

        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            return None

    def get_position(self) -> PTZPosition:
        """Get current PTZ position."""
        self._update_position()
        return self.current_position

    def _norm_to_degrees_pan(self, norm: float) -> float:
        """Convert normalized pan (-1 to 1) to degrees."""
        # Linear interpolation
        if norm >= 0:
            return norm * self.calibration.pan_max
        else:
            return -norm * abs(self.calibration.pan_min)

    def _norm_to_degrees_tilt(self, norm: float) -> float:
        """Convert normalized tilt (-1 to 1) to degrees."""
        if norm >= 0:
            return norm * self.calibration.tilt_max
        else:
            return -norm * abs(self.calibration.tilt_min)

    def _degrees_to_norm_pan(self, deg: float) -> float:
        """Convert degrees to normalized pan (-1 to 1)."""
        if deg >= 0:
            return min(1.0, deg / self.calibration.pan_max)
        else:
            return max(-1.0, deg / abs(self.calibration.pan_min))

    def _degrees_to_norm_tilt(self, deg: float) -> float:
        """Convert degrees to normalized tilt (-1 to 1)."""
        if deg >= 0:
            return min(1.0, deg / self.calibration.tilt_max)
        else:
            return max(-1.0, deg / abs(self.calibration.tilt_min))

    # -------------------------------------------------------------------------
    # Movement Commands
    # -------------------------------------------------------------------------

    def move_absolute(
        self,
        pan_deg: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        speed: float = 0.5
    ) -> bool:
        """
        Move to absolute position using ONVIF AbsoluteMove.

        This is the primary movement method for tracking.
        Uses full 342.8 degree pan range (no 90 degree limit).

        Args:
            pan_deg: Target pan in degrees (-168.4 to 174.4)
            tilt_deg: Target tilt in degrees (-78.8 to 101.3)
            speed: Movement speed (0-1)

        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            return False

        if self.mode == ControlMode.SAFE_MODE:
            self.logger.warning("Movement blocked: SAFE_MODE active")
            return False

        # Apply calibrated limits (use calibration data if available, fallback to SOFT_* constants)
        if pan_deg is not None:
            pan_min = self.calibration.pan_min if self.calibration.verified else self.SOFT_PAN_MIN
            pan_max = self.calibration.pan_max if self.calibration.verified else self.SOFT_PAN_MAX
            pan_deg = max(pan_min, min(pan_max, pan_deg))
        else:
            pan_deg = self.current_position.pan

        if tilt_deg is not None:
            tilt_min = self.calibration.tilt_min if self.calibration.verified else self.SOFT_TILT_MIN
            tilt_max = self.calibration.tilt_max if self.calibration.verified else self.SOFT_TILT_MAX
            tilt_deg = max(tilt_min, min(tilt_max, tilt_deg))
        else:
            tilt_deg = self.current_position.tilt

        # Sonoff camera uses RAW DEGREE values directly (NOT normalized -1 to 1)
        # Clamp speed
        speed = max(0.1, min(1.0, speed))

        try:
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self.profile_token
            request.Position = {
                'PanTilt': {'x': pan_deg, 'y': tilt_deg},
                'Zoom': {'x': self.current_position.zoom}
            }
            request.Speed = {
                'PanTilt': {'x': speed, 'y': speed},
                'Zoom': {'x': speed}
            }

            self.ptz_service.AbsoluteMove(request)

            self.last_move_time = time.time()
            self.logger.debug(f"AbsoluteMove: Pan={pan_deg:.1f}°, Tilt={tilt_deg:.1f}°")

            return True

        except Exception as e:
            # Sonoff returns "Invalid position" error but still executes the move
            if "Invalid position" in str(e):
                self.last_move_time = time.time()
                self.logger.debug(f"AbsoluteMove: Pan={pan_deg:.1f}°, Tilt={tilt_deg:.1f}° (camera quirk)")
                return True
            self.logger.error(f"AbsoluteMove failed: {e}")
            return False

    def move_relative(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        speed: float = 0.5
    ) -> bool:
        """
        Move relative to current position.

        Args:
            pan_delta: Pan offset in degrees
            tilt_delta: Tilt offset in degrees
            speed: Movement speed (0-1)
        """
        self._update_position()

        target_pan = self.current_position.pan + pan_delta
        target_tilt = self.current_position.tilt + tilt_delta

        return self.move_absolute(target_pan, target_tilt, speed)

    def stop(self) -> bool:
        """Stop all movement."""
        if not self.is_connected:
            return False

        try:
            request = self.ptz_service.create_type('Stop')
            request.ProfileToken = self.profile_token
            request.PanTilt = True
            request.Zoom = True

            self.ptz_service.Stop(request)
            self.logger.debug("Movement stopped")
            return True

        except Exception as e:
            self.logger.error(f"Stop failed: {e}")
            return False

    def continuous_move(self, vel_pan: float, vel_tilt: float, vel_zoom: float = 0.0, timeout_sec: float = 1.0, verbose: bool = False) -> bool:
        """
        Send ContinuousMove command for velocity-based control.

        Used by AutonomousTracker for smooth tracking.

        Args:
            vel_pan: Pan velocity (-1 to 1), negative = right, positive = left
            vel_tilt: Tilt velocity (-1 to 1), negative = down, positive = up
            vel_zoom: Zoom velocity (-1 to 1)
            timeout_sec: Duration in seconds (converted to ISO 8601 duration PT{n}S)
            verbose: Log full request details

        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            self.logger.warning("continuous_move: not connected")
            return False

        if self.mode == ControlMode.SAFE_MODE:
            self.logger.warning("continuous_move blocked: SAFE_MODE active")
            return False

        if not self.ptz_service:
            self.logger.error("continuous_move: ptz_service is None!")
            return False

        if not self.profile_token:
            self.logger.error("continuous_move: profile_token is None!")
            return False

        try:
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile_token
            request.Velocity = {
                'PanTilt': {'x': vel_pan, 'y': vel_tilt},
                'Zoom': {'x': vel_zoom}
            }
            # Add explicit Timeout in ISO 8601 duration format
            request.Timeout = f'PT{int(timeout_sec)}S'

            if verbose:
                self.logger.info(f"=== ContinuousMove REQUEST ===")
                self.logger.info(f"  ProfileToken: {request.ProfileToken}")
                self.logger.info(f"  Velocity.PanTilt: x={vel_pan}, y={vel_tilt}")
                self.logger.info(f"  Velocity.Zoom: x={vel_zoom}")
                self.logger.info(f"  Timeout: {request.Timeout}")
                self.logger.info(f"  ptz_service id: {id(self.ptz_service)}")
                # Log serialized request
                try:
                    from zeep.helpers import serialize_object
                    self.logger.info(f"  Full request: {serialize_object(request)}")
                except:
                    pass

            result = self.ptz_service.ContinuousMove(request)

            if verbose:
                self.logger.info(f"=== ContinuousMove RESULT: {result} ===")

            self.last_move_time = time.time()
            self.logger.debug(f"ContinuousMove: vel=({vel_pan:+.3f}, {vel_tilt:+.3f}) -> OK")
            return True

        except Exception as e:
            self.logger.error(f"ContinuousMove EXCEPTION: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def test_continuous_move(self, direction: str = "left", speed: float = 0.3, duration: float = 1.0) -> bool:
        """
        Test ContinuousMove manually (for debugging).

        Args:
            direction: "left", "right", "up", "down"
            speed: Velocity magnitude (0-1)
            duration: How long to move (seconds)

        Returns:
            True if command sent successfully
        """
        vel_pan = 0.0
        vel_tilt = 0.0

        if direction == "left":
            vel_pan = speed
        elif direction == "right":
            vel_pan = -speed
        elif direction == "up":
            vel_tilt = speed
        elif direction == "down":
            vel_tilt = -speed

        self.logger.info(f"TEST ContinuousMove: {direction} speed={speed} duration={duration}s")
        return self.continuous_move(vel_pan, vel_tilt, timeout_sec=duration, verbose=True)

    def goto_home(self) -> bool:
        """Move to home position (0, 0)."""
        return self.move_absolute(0.0, 0.0, speed=0.5)

    def center(self) -> bool:
        """Alias for goto_home - move to center position."""
        return self.goto_home()

    def move_manual(self, direction: str, speed: float = 0.3) -> bool:
        """
        Manual PTZ movement in a direction.

        Args:
            direction: "left", "right", "up", "down", "stop"
            speed: Movement speed (0.0-1.0)

        Returns:
            True if command sent successfully
        """
        if direction == "stop":
            return self.stop()

        # Get current position
        pos = self.get_position()

        # Calculate relative movement (10 degrees per step)
        step = 10.0
        new_pan = pos.pan
        new_tilt = pos.tilt

        if direction == "left":
            new_pan = pos.pan + step
        elif direction == "right":
            new_pan = pos.pan - step
        elif direction == "up":
            new_tilt = pos.tilt + step
        elif direction == "down":
            new_tilt = pos.tilt - step
        else:
            self.logger.warning(f"Unknown direction: {direction}")
            return False

        # Clamp to limits
        new_pan = max(self.calibration.pan_min, min(self.calibration.pan_max, new_pan))
        new_tilt = max(self.calibration.tilt_min, min(self.calibration.tilt_max, new_tilt))

        return self.move_absolute(new_pan, new_tilt, speed=speed)

    def wait_for_position(
        self,
        target_pan: float,
        target_tilt: float,
        timeout: float = 15.0
    ) -> bool:
        """
        Wait until camera reaches target position.

        Args:
            target_pan: Target pan in degrees
            target_tilt: Target tilt in degrees
            timeout: Maximum wait time in seconds

        Returns:
            True if position reached within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            pos = self.get_position()

            pan_error = abs(pos.pan - target_pan)
            tilt_error = abs(pos.tilt - target_tilt)

            if pan_error < self.POSITION_TOLERANCE and tilt_error < self.POSITION_TOLERANCE:
                return True

            if not pos.moving:
                # Not moving but not at target - might be at limit
                return False

            time.sleep(0.1)

        self.logger.warning(f"Position timeout after {timeout}s")
        return False

    # -------------------------------------------------------------------------
    # Tracking System (AbsoluteMove-based)
    # -------------------------------------------------------------------------

    def process_detection(self, detection: Detection) -> bool:
        """
        Process a detection and move camera to track target.

        Uses AbsoluteMove for full range access (342.8 degrees).
        Calculates target position based on detection error.

        Args:
            detection: Detection result from perception system

        Returns:
            True if tracking command was issued
        """
        if self.mode != ControlMode.AUTONOMOUS:
            return False

        # Cooldown check
        now = time.time()
        if (now - self.last_move_time) * 1000 < self.COOLDOWN_MS:
            return False

        self.last_detection = detection
        self.last_detection_time = now

        # Calculate error from center (detection center is 0-1, we want error from 0.5)
        error_x = detection.center_x - 0.5  # Positive = target is right of center
        error_y = detection.center_y - 0.5  # Positive = target is below center

        # Check deadzone
        if abs(error_x) < self.DEADZONE and abs(error_y) < self.DEADZONE:
            self._set_tracking_state(TrackingState.LOCKED)
            return False

        self._set_tracking_state(TrackingState.TRACKING)

        # Calculate pan/tilt deltas
        # Positive error_x (target right) -> negative pan delta (move right = decrease pan)
        # Positive error_y (target below) -> negative tilt delta (move down = decrease tilt)

        # Field of view estimation (approximate for this camera)
        fov_h = 110.0  # Horizontal FOV in degrees
        fov_v = 65.0   # Vertical FOV in degrees

        # Convert error to degrees
        pan_delta = -error_x * fov_h * self.TRACKING_GAIN_PAN
        tilt_delta = -error_y * fov_v * self.TRACKING_GAIN_TILT

        # Limit step size
        pan_delta = max(-self.MAX_STEP_PAN, min(self.MAX_STEP_PAN, pan_delta))
        tilt_delta = max(-self.MAX_STEP_TILT, min(self.MAX_STEP_TILT, tilt_delta))

        # Get current position
        self._update_position()

        # Calculate target position
        target_pan = self.current_position.pan + pan_delta
        target_tilt = self.current_position.tilt + tilt_delta

        # Apply soft limits
        target_pan = max(self.SOFT_PAN_MIN, min(self.SOFT_PAN_MAX, target_pan))
        target_tilt = max(self.SOFT_TILT_MIN, min(self.SOFT_TILT_MAX, target_tilt))

        # Calculate speed based on error magnitude
        error_magnitude = math.sqrt(error_x**2 + error_y**2)
        speed = 1.0  # Max speed for fast tracking

        # Execute AbsoluteMove
        success = self.move_absolute(target_pan, target_tilt, speed)

        if success:
            self.logger.debug(
                f"Tracking: error=({error_x:.2f}, {error_y:.2f}) -> "
                f"target=({target_pan:.1f} deg, {target_tilt:.1f} deg)"
            )

        return success

    def check_target_lost(self) -> bool:
        """
        Check if target has been lost and start patrol if needed.

        Call this periodically when no detections are received.

        Returns:
            True if patrol was started
        """
        if self.mode != ControlMode.AUTONOMOUS:
            return False

        now = time.time()
        time_since_detection = now - self.last_detection_time

        if time_since_detection > self.TARGET_LOST_TIMEOUT:
            if self.tracking_state not in [TrackingState.PATROL, TrackingState.IDLE]:
                self._set_tracking_state(TrackingState.SEARCHING)
                self.logger.info("Target lost, starting patrol scan")
                return self.start_patrol()

        return False

    def start_patrol(self) -> bool:
        """Start 360 degree patrol scan."""
        self._set_tracking_state(TrackingState.PATROL)
        self.patrol_index = 0
        return self._patrol_next()

    def _patrol_next(self) -> bool:
        """Move to next patrol position."""
        if self.tracking_state != TrackingState.PATROL:
            return False

        if self.patrol_index >= len(self.PATROL_POSITIONS):
            self.patrol_index = 0  # Loop

        pan, tilt = self.PATROL_POSITIONS[self.patrol_index]
        self.patrol_index += 1

        self.logger.debug(f"Patrol position {self.patrol_index}/{len(self.PATROL_POSITIONS)}")
        return self.move_absolute(pan, tilt, speed=0.3)

    def patrol_step(self) -> bool:
        """
        Execute one patrol step.

        Call this when camera reaches patrol position to move to next.
        """
        if self.tracking_state != TrackingState.PATROL:
            return False

        # Check if at current patrol position
        pos = self.get_position()
        target_pan, target_tilt = self.PATROL_POSITIONS[(self.patrol_index - 1) % len(self.PATROL_POSITIONS)]

        if not pos.moving:
            # At position, move to next
            return self._patrol_next()

        return False

    def stop_tracking(self):
        """Stop tracking and return to idle."""
        self.stop()
        self._set_tracking_state(TrackingState.IDLE)
        self.last_detection = None

    def _set_tracking_state(self, state: TrackingState):
        """Update tracking state with callback."""
        if state != self.tracking_state:
            old_state = self.tracking_state
            self.tracking_state = state
            self.logger.info(f"Tracking state: {old_state.name} -> {state.name}")

            if self.on_state_change:
                self.on_state_change(state)

    # -------------------------------------------------------------------------
    # Mode Management
    # -------------------------------------------------------------------------

    def set_mode(self, mode: ControlMode):
        """Set control mode."""
        old_mode = self.mode
        self.mode = mode
        self.logger.info(f"Mode changed: {old_mode.name} -> {mode.name}")

        if mode == ControlMode.SAFE_MODE:
            self.stop()
            self._set_tracking_state(TrackingState.IDLE)

    def enable_autonomous(self):
        """Enable autonomous tracking mode."""
        self.set_mode(ControlMode.AUTONOMOUS)

    def enable_manual(self):
        """Enable manual override mode."""
        self.set_mode(ControlMode.MANUAL_OVERRIDE)

    def enable_safe(self):
        """Enable safe mode (no movement)."""
        self.set_mode(ControlMode.SAFE_MODE)

    # -------------------------------------------------------------------------
    # Exclusive PTZ Access
    # -------------------------------------------------------------------------

    def acquire_exclusive(self, owner: str) -> bool:
        """Acquire exclusive PTZ control. Returns False if already owned by someone else."""
        with self._exclusive_lock:
            if self._exclusive_owner is not None and self._exclusive_owner != owner:
                self.logger.warning(f"Exclusive PTZ already held by '{self._exclusive_owner}', rejected '{owner}'")
                return False
            self._exclusive_owner = owner
            self.set_mode(ControlMode.MANUAL_OVERRIDE)
            self.logger.info(f"Exclusive PTZ control granted to '{owner}'")
            return True

    def release_exclusive(self, owner: str):
        """Release exclusive PTZ control."""
        with self._exclusive_lock:
            if self._exclusive_owner == owner:
                self._exclusive_owner = None
                self.logger.info(f"Exclusive PTZ control released by '{owner}'")
            else:
                self.logger.warning(f"Release attempt by '{owner}' but owner is '{self._exclusive_owner}'")

    @property
    def exclusive_owner(self) -> Optional[str]:
        """Current exclusive PTZ owner, or None."""
        return self._exclusive_owner

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def calibrate(self) -> CalibrationData:
        """
        Perform full PTZ calibration.

        Moves to each limit and records positions.
        """
        self.logger.info("Starting PTZ calibration...")

        if not self.is_connected:
            self.logger.error("Not connected")
            return self.calibration

        # Move to extremes and record positions
        limits = {}

        # Right limit
        self.logger.info("Moving to right limit...")
        self.move_absolute(pan_deg=-180.0, tilt_deg=0.0, speed=0.5)
        self.wait_for_position(-180.0, 0.0, timeout=15.0)
        time.sleep(0.5)
        pos = self.get_position()
        limits['pan_min'] = pos.pan
        self.logger.info(f"Right limit: {pos.pan:.1f} deg")

        # Left limit
        self.logger.info("Moving to left limit...")
        self.move_absolute(pan_deg=180.0, tilt_deg=0.0, speed=0.5)
        self.wait_for_position(180.0, 0.0, timeout=15.0)
        time.sleep(0.5)
        pos = self.get_position()
        limits['pan_max'] = pos.pan
        self.logger.info(f"Left limit: {pos.pan:.1f} deg")

        # Down limit
        self.logger.info("Moving to down limit...")
        self.move_absolute(pan_deg=0.0, tilt_deg=-90.0, speed=0.5)
        self.wait_for_position(0.0, -90.0, timeout=10.0)
        time.sleep(0.5)
        pos = self.get_position()
        limits['tilt_min'] = pos.tilt
        self.logger.info(f"Down limit: {pos.tilt:.1f} deg")

        # Up limit
        self.logger.info("Moving to up limit...")
        self.move_absolute(pan_deg=0.0, tilt_deg=110.0, speed=0.5)
        self.wait_for_position(0.0, 110.0, timeout=10.0)
        time.sleep(0.5)
        pos = self.get_position()
        limits['tilt_max'] = pos.tilt
        self.logger.info(f"Up limit: {pos.tilt:.1f} deg")

        # Return to center
        self.goto_home()

        # Update calibration
        self.calibration = CalibrationData(
            pan_min=limits['pan_min'],
            pan_max=limits['pan_max'],
            tilt_min=limits['tilt_min'],
            tilt_max=limits['tilt_max'],
            verified=True,
            timestamp=time.time()
        )

        total_pan = limits['pan_max'] - limits['pan_min']
        total_tilt = limits['tilt_max'] - limits['tilt_min']

        self.logger.info(f"Calibration complete:")
        self.logger.info(f"  Pan range: {limits['pan_min']:.1f} to {limits['pan_max']:.1f} deg ({total_pan:.1f} deg total)")
        self.logger.info(f"  Tilt range: {limits['tilt_min']:.1f} to {limits['tilt_max']:.1f} deg ({total_tilt:.1f} deg total)")

        return self.calibration

    def get_calibration(self) -> CalibrationData:
        """Get current calibration data."""
        return self.calibration

    # -------------------------------------------------------------------------
    # LED Control (Stubbed - not supported on this camera)
    # -------------------------------------------------------------------------

    def set_led(self, state: bool) -> bool:
        """Set IR LED state (not supported)."""
        self.logger.warning("LED control not supported on this camera")
        return False

    def set_ir_mode(self, mode: str) -> bool:
        """Set IR mode: auto/on/off (not supported)."""
        self.logger.warning("IR mode control not supported on this camera")
        return False

    # -------------------------------------------------------------------------
    # Audio Control (Stubbed - backchannel not supported, mic only via RTSP)
    # -------------------------------------------------------------------------

    def play_audio(self, audio_data: bytes) -> bool:
        """Play audio via speaker (not supported - no Profile T)."""
        self.logger.warning("Audio backchannel not supported (ONVIF Profile S only)")
        return False

    def get_audio_stream_url(self) -> str:
        """Get RTSP URL for audio stream (G.711 A-law, 8kHz, mono)."""
        return f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/av_stream/ch0"

    # -------------------------------------------------------------------------
    # Status and Info
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get full camera status."""
        pos = self.get_position()

        return {
            'connected': self.is_connected,
            'mode': self.mode.name,
            'tracking_state': self.tracking_state.name,
            'position': {
                'pan': pos.pan,
                'tilt': pos.tilt,
                'zoom': pos.zoom,
                'moving': pos.moving
            },
            'calibration': {
                'pan_range': (self.calibration.pan_min, self.calibration.pan_max),
                'tilt_range': (self.calibration.tilt_min, self.calibration.tilt_max),
                'total_pan': self.calibration.pan_max - self.calibration.pan_min,
                'verified': self.calibration.verified
            },
            'last_detection': {
                'time': self.last_detection_time,
                'person_id': self.last_detection.person_id if self.last_detection else None
            } if self.last_detection else None
        }


# =============================================================================
# Singleton
# =============================================================================

_camera_controller: Optional[SonoffCameraController] = None


def get_camera_controller(
    camera_ip: str = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP"),
    username: str = os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME"),
    password: str = os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME"),
    auto_connect: bool = True
) -> SonoffCameraController:
    """Get or create SonoffCameraController singleton with auto-connect."""
    global _camera_controller
    if _camera_controller is None:
        _camera_controller = SonoffCameraController(
            camera_ip=camera_ip,
            username=username,
            password=password
        )
        if auto_connect:
            _camera_controller.connect()
    elif auto_connect and not _camera_controller.is_connected:
        _camera_controller.connect()
    return _camera_controller


# =============================================================================
# Main / Test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sonoff Camera Controller Test")
    parser.add_argument("--ip", default="192.168.178.89", help="Camera IP")
    parser.add_argument("--user", default="admin", help="Username")
    parser.add_argument("--password", default="moloch2024", help="Password")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--test", action="store_true", help="Run movement test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    controller = SonoffCameraController(
        camera_ip=args.ip,
        username=args.user,
        password=args.password,
        log_level=logging.DEBUG
    )

    if not controller.connect():
        print("Failed to connect!")
        exit(1)

    print(f"\nCamera Status:")
    status = controller.get_status()
    print(f"  Position: Pan={status['position']['pan']:.1f} deg, Tilt={status['position']['tilt']:.1f} deg")
    print(f"  Calibration: Pan {status['calibration']['pan_range'][0]:.1f} to {status['calibration']['pan_range'][1]:.1f} deg")
    print(f"  Total Pan Range: {status['calibration']['total_pan']:.1f} deg")

    if args.calibrate:
        print("\n=== Running Calibration ===")
        cal = controller.calibrate()
        print(f"\nCalibration Result:")
        print(f"  Pan: {cal.pan_min:.1f} to {cal.pan_max:.1f} deg")
        print(f"  Tilt: {cal.tilt_min:.1f} to {cal.tilt_max:.1f} deg")

    if args.test:
        print("\n=== Movement Test (AbsoluteMove) ===")

        # Test full range
        positions = [
            (0, 0, "Center"),
            (-168.0, 0, "Full Right"),
            (174.0, 0, "Full Left"),
            (0, -78.0, "Center Down"),
            (0, 100.0, "Center Up"),
            (0, 0, "Back to Center")
        ]

        for pan, tilt, name in positions:
            print(f"\nMoving to {name} ({pan:.0f}, {tilt:.0f})...")
            controller.move_absolute(pan, tilt, speed=0.5)
            controller.wait_for_position(pan, tilt, timeout=10.0)
            pos = controller.get_position()
            print(f"  Reached: Pan={pos.pan:.1f} deg, Tilt={pos.tilt:.1f} deg")
            time.sleep(1)

    print("\nTest complete!")
    controller.disconnect()
