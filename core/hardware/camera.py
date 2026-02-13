#!/usr/bin/env python3
"""
M.O.L.O.C.H. Camera Controller
===============================

Merged controller for Sonoff CAM-PT2 (GK-200MP2-B).
Combines ONVIF PTZ + eWeLink Cloud + Vision Integration.

Calibrated Limits (verified 2026-02-04):
- Pan:  -168.4 deg (right) to +170.0 deg (left) = 338.4 deg total
- Tilt: -78.0 deg (down) to +78.8 deg (up) = 156.8 deg total
- AbsoluteMove with RAW DEGREE values (not normalized!)

Architecture:
  Vision -> camera.process_vision_event() -> ONVIF AbsoluteMove
  GUI    -> camera.move_absolute/manual()  -> ONVIF AbsoluteMove
  Cloud  -> camera.set_night_mode() etc.   -> eWeLink API
"""

import os
import logging
import time
import math
import threading
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, Callable, List

# ONVIF imports
try:
    from onvif import ONVIFCamera
    from zeep.helpers import serialize_object
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False

# Cloud bridge imports
try:
    from core.hardware.camera_cloud_bridge import (
        CameraCloudBridgeSync,
        CloudConfig,
        CloudStatus
    )
    CLOUD_BRIDGE_AVAILABLE = True
except ImportError:
    CLOUD_BRIDGE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _load_calibrated_limits():
    """Load calibrated PTZ limits from config file."""
    try:
        from core.hardware.ptz_calibration import get_ptz_limits
        return get_ptz_limits()
    except Exception:
        return None


# =============================================================================
# Enums
# =============================================================================

class ControlMode(Enum):
    """Camera control modes."""
    AUTONOMOUS = auto()
    MANUAL_OVERRIDE = auto()
    SAFE_MODE = auto()


class TrackingState(Enum):
    """Person tracking states."""
    IDLE = auto()
    TRACKING = auto()
    SEARCHING = auto()
    PATROL = auto()
    LOCKED = auto()


class TrackingMode(Enum):
    """Vision tracking modes (from orchestrator layer)."""
    DISABLED = "disabled"
    MANUAL = "manual"
    AUTO_TRACK = "auto_track"
    SEARCH = "search"


class NightMode(Enum):
    """IR/Night mode settings (via eWeLink Cloud only)."""
    AUTO = auto()
    DAY = auto()
    NIGHT = auto()


class LEDLevel(Enum):
    """LED brightness levels (via eWeLink Cloud only)."""
    OFF = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# =============================================================================
# Data Classes
# =============================================================================

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
    """Current PTZ position in degrees."""
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    moving: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibrationData:
    """PTZ calibration data (verified 2026-02-04)."""
    pan_min: float = -168.4
    pan_max: float = 170.0
    tilt_min: float = -78.0
    tilt_max: float = 78.8
    pan_speed: float = 20.0
    tilt_speed: float = 12.0
    verified: bool = True
    timestamp: float = field(default_factory=time.time)


@dataclass
class VisionEvent:
    """Data from the vision module."""
    detection_found: bool
    target_center_x: int = 0
    frame_center_x: int = 960
    frame_width: int = 1920
    confidence: float = 0.0
    bbox: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrackingDecision:
    """Decision from vision processing."""
    should_move: bool = False
    action: str = "none"
    velocity: float = 0.0
    duration: float = 0.0
    reason: str = ""
    error_x: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CameraStatus:
    """Complete camera status."""
    connected: bool = False
    model: str = "Unknown"
    firmware: str = "Unknown"
    mode: str = "UNKNOWN"
    tracking_state: str = "IDLE"
    position: Optional[PTZPosition] = None
    pan_range: Tuple[float, float] = (-168.4, 170.0)
    tilt_range: Tuple[float, float] = (-78.0, 78.8)
    calibrated: bool = True
    ptz_available: bool = True
    audio_available: bool = True
    night_mode_available: bool = False
    led_control_available: bool = False
    sleep_mode_available: bool = False
    cloud_enabled: bool = False
    cloud_connected: bool = False
    cloud_status: str = "disabled"
    rtsp_url: str = ""
    last_move_time: float = 0.0
    last_detection_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'connected': self.connected,
            'model': self.model,
            'firmware': self.firmware,
            'mode': self.mode,
            'tracking_state': self.tracking_state,
            'position': {
                'pan': self.position.pan if self.position else 0.0,
                'tilt': self.position.tilt if self.position else 0.0,
                'zoom': self.position.zoom if self.position else 0.0,
                'moving': self.position.moving if self.position else False
            } if self.position else None,
            'calibration': {
                'pan_range': self.pan_range,
                'tilt_range': self.tilt_range,
                'calibrated': self.calibrated
            },
            'cloud': {
                'enabled': self.cloud_enabled,
                'connected': self.cloud_connected,
                'status': self.cloud_status
            }
        }


# =============================================================================
# Main Controller
# =============================================================================

class SonoffCameraController:
    """
    Unified camera controller for Sonoff CAM-PT2.

    Uses AbsoluteMove for tracking (full 342.8 degree pan range).
    Cloud features (LED, IR, Sleep) via eWeLink Cloud Bridge.
    """

    # Hardware limits (calibrated 2026-02-04)
    PAN_MIN = -168.4
    PAN_MAX = 170.0
    TILT_MIN = -78.0
    TILT_MAX = 78.8

    # Tracking parameters
    DEADZONE = 0.05
    TRACKING_GAIN_PAN = 0.7
    TRACKING_GAIN_TILT = 0.5
    MAX_STEP_PAN = 30.0
    MAX_STEP_TILT = 20.0
    TARGET_LOST_TIMEOUT = 10.0
    FOV_HORIZONTAL = 110.0
    FOV_VERTICAL = 65.0

    # Safety parameters
    COOLDOWN_MS = 50
    POSITION_TOLERANCE = 1.0

    # Patrol scan positions
    PATROL_POSITIONS = [
        (0.0, 0.0),
        (-84.0, 0.0),
        (-168.0, 0.0),
        (-84.0, 30.0),
        (0.0, 30.0),
        (84.0, 30.0),
        (170.0, 0.0),
        (84.0, 0.0),
    ]

    # Vision integration (absorbed from orchestrator)
    MIN_VISION_CYCLE = 0.1

    def __init__(
        self,
        camera_ip: str = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP"),
        username: str = os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME"),
        password: str = os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME"),
        onvif_port: int = 80,
        rtsp_port: int = 554,
        log_level: int = logging.INFO
    ):
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.onvif_port = onvif_port
        self.rtsp_port = rtsp_port

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
        self.device_service = None
        self.profile_token: Optional[str] = None

        # Device info
        self.device_model = "Unknown"
        self.device_firmware = "Unknown"

        # Tracking state
        self.last_detection: Optional[Detection] = None
        self.last_detection_time: float = 0
        self.last_move_time: float = 0
        self.patrol_index: int = 0

        # Vision integration (from orchestrator)
        self._tracking_mode = TrackingMode.DISABLED
        self._last_vision_time: float = 0
        self._vision_stats = {"total_events": 0, "tracking_calls": 0, "mode_changes": 0}

        # Threading
        self._lock = threading.Lock()
        self._connected = False

        # Exclusive PTZ access
        self._exclusive_owner: Optional[str] = None
        self._exclusive_lock = threading.Lock()

        # Cloud bridge
        self.cloud_bridge: Optional[CameraCloudBridgeSync] = None
        self._load_cloud_config()

        # Callbacks
        self.on_position_update: Optional[Callable[[PTZPosition], None]] = None
        self.on_state_change: Optional[Callable[[TrackingState], None]] = None

    # -------------------------------------------------------------------------
    # Cloud Bridge Configuration
    # -------------------------------------------------------------------------

    def _load_cloud_config(self):
        """Load cloud bridge configuration from file."""
        if not CLOUD_BRIDGE_AVAILABLE:
            return

        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "camera_cloud.json"
            if not config_path.exists():
                return

            with open(config_path, 'r') as f:
                config_data = json.load(f)

            if not config_data.get('cloud_enabled', False):
                return

            cc = config_data.get('cloud_config', {})

            def _cfg(key, env_var=None, default=''):
                val = cc.get(key, default)
                if env_var and (not val or val == 'CHANGE_ME'):
                    val = os.environ.get(env_var, default)
                return val

            cloud_config = CloudConfig(
                enabled=True,
                api_base_url=_cfg('api_base_url'),
                app_id=_cfg('app_id', 'EWELINK_APP_ID_1'),
                app_secret=_cfg('app_secret', 'EWELINK_APP_SECRET_1'),
                device_id=_cfg('device_id'),
                username=_cfg('username', 'EWELINK_USERNAME'),
                password=_cfg('password', 'EWELINK_PASSWORD'),
                timeout=cc.get('timeout', 3.0),
                retry_count=cc.get('retry_count', 1),
                token_refresh_margin=cc.get('token_refresh_margin', 300)
            )

            self.cloud_bridge = CameraCloudBridgeSync(cloud_config)
            self.logger.info("Cloud bridge configured")

        except Exception as e:
            self.logger.warning(f"Failed to load cloud config: {e}")
            self.cloud_bridge = None

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
            self.device_service = self.camera.create_devicemgmt_service()
            self.media_service = self.camera.create_media_service()
            self.ptz_service = self.camera.create_ptz_service()

            # Device info
            try:
                device_info = self.device_service.GetDeviceInformation()
                self.device_model = device_info.Model
                self.device_firmware = device_info.FirmwareVersion
                self.logger.info(f"Device: {device_info.Manufacturer} {self.device_model}")
            except Exception:
                pass

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

            # Load calibrated limits
            limits = _load_calibrated_limits()
            if limits and limits.calibrated:
                self.calibration.pan_min = limits.pan_min
                self.calibration.pan_max = limits.pan_max
                self.calibration.tilt_min = limits.tilt_min
                self.calibration.tilt_max = limits.tilt_max
                self.calibration.verified = True
                self.logger.info(f"Calibrated limits: pan=[{limits.pan_min:.1f}, {limits.pan_max:.1f}]")

            self._connected = True
            self.logger.info("Camera connected successfully")

            # Connect cloud bridge (non-blocking)
            if self.cloud_bridge:
                try:
                    if self.cloud_bridge.connect():
                        self.logger.info("Cloud bridge connected")
                    else:
                        self.logger.warning("Cloud bridge connection failed")
                except Exception as e:
                    self.logger.warning(f"Cloud bridge error: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera."""
        self.stop()
        if self.cloud_bridge:
            try:
                self.cloud_bridge.disconnect()
            except Exception:
                pass
        self._connected = False
        self.camera = None
        self.ptz_service = None
        self.media_service = None
        self.device_service = None
        self.logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
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

            # Sonoff returns raw degree values
            pan_deg = float(pan_tilt.get('x', 0))
            tilt_deg = float(pan_tilt.get('y', 0))

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
        Uses full 342.8 degree pan range (no 90 degree limit).
        """
        if not self.is_connected:
            return False

        if self.mode == ControlMode.SAFE_MODE:
            self.logger.warning("Movement blocked: SAFE_MODE active")
            return False

        # Apply calibrated limits
        if pan_deg is not None:
            pan_min = self.calibration.pan_min if self.calibration.verified else self.PAN_MIN
            pan_max = self.calibration.pan_max if self.calibration.verified else self.PAN_MAX
            pan_deg = max(pan_min, min(pan_max, pan_deg))
        else:
            pan_deg = self.current_position.pan

        if tilt_deg is not None:
            tilt_min = self.calibration.tilt_min if self.calibration.verified else self.TILT_MIN
            tilt_max = self.calibration.tilt_max if self.calibration.verified else self.TILT_MAX
            tilt_deg = max(tilt_min, min(tilt_max, tilt_deg))
        else:
            tilt_deg = self.current_position.tilt

        speed = max(0.1, min(1.0, speed))

        try:
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self.profile_token
            request.Position = {
                'PanTilt': {'x': pan_deg, 'y': tilt_deg},
            }
            request.Speed = {
                'PanTilt': {'x': speed, 'y': speed},
            }

            self.ptz_service.AbsoluteMove(request)
            self.last_move_time = time.time()
            self.logger.debug(f"AbsoluteMove: Pan={pan_deg:.1f}, Tilt={tilt_deg:.1f}")
            return True

        except Exception as e:
            if "Invalid position" in str(e):
                self.logger.warning(f"AbsoluteMove: Position out of range (Pan={pan_deg:.1f}, Tilt={tilt_deg:.1f})")
                return False
            self.logger.error(f"AbsoluteMove failed: {e}")
            return False

    def move_relative(self, pan_delta: float = 0.0, tilt_delta: float = 0.0, speed: float = 0.5) -> bool:
        """Move relative to current position."""
        self._update_position()
        return self.move_absolute(
            self.current_position.pan + pan_delta,
            self.current_position.tilt + tilt_delta,
            speed
        )

    def continuous_move(self, vel_pan: float, vel_tilt: float, vel_zoom: float = 0.0,
                        timeout_sec: float = 1.0, verbose: bool = False) -> bool:
        """
        ContinuousMove for velocity-based control.
        Used by AutonomousTracker for smooth tracking.
        """
        if not self.is_connected:
            return False

        if self.mode == ControlMode.SAFE_MODE:
            return False

        try:
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile_token
            request.Velocity = {
                'PanTilt': {'x': vel_pan, 'y': vel_tilt},
            }
            request.Timeout = f'PT{int(timeout_sec)}S'

            if verbose:
                self.logger.info(f"ContinuousMove: vel=({vel_pan:+.3f}, {vel_tilt:+.3f}) timeout={timeout_sec}s")

            self.ptz_service.ContinuousMove(request)
            self.last_move_time = time.time()
            self.logger.debug(f"ContinuousMove: vel=({vel_pan:+.3f}, {vel_tilt:+.3f})")
            return True

        except Exception as e:
            self.logger.error(f"ContinuousMove failed: {e}")
            return False

    def test_continuous_move(self, direction: str = "left", speed: float = 0.3, duration: float = 1.0) -> bool:
        """Test ContinuousMove manually (for debugging)."""
        vel_map = {"left": (speed, 0), "right": (-speed, 0), "up": (0, speed), "down": (0, -speed)}
        vel = vel_map.get(direction, (0, 0))
        self.logger.info(f"TEST ContinuousMove: {direction} speed={speed} duration={duration}s")
        return self.continuous_move(vel[0], vel[1], timeout_sec=duration, verbose=True)

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

    def goto_home(self) -> bool:
        """Move to home position (0, 0)."""
        return self.move_absolute(0.0, 0.0, speed=0.5)

    def center(self) -> bool:
        """Alias for goto_home."""
        return self.goto_home()

    def move_manual(self, direction: str, speed: float = 0.3) -> bool:
        """Manual PTZ movement: left/right/up/down/stop."""
        if direction == "stop":
            return self.stop()

        pos = self.get_position()
        step = 10.0
        new_pan, new_tilt = pos.pan, pos.tilt

        if direction == "left":
            new_pan += step
        elif direction == "right":
            new_pan -= step
        elif direction == "up":
            new_tilt += step
        elif direction == "down":
            new_tilt -= step
        else:
            return False

        return self.move_absolute(new_pan, new_tilt, speed=speed)

    def wait_for_position(self, target_pan: float, target_tilt: float, timeout: float = 15.0) -> bool:
        """Wait until camera reaches target position."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            pos = self.get_position()
            if (abs(pos.pan - target_pan) < self.POSITION_TOLERANCE and
                    abs(pos.tilt - target_tilt) < self.POSITION_TOLERANCE):
                return True
            if not pos.moving:
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
        Uses AbsoluteMove for full 342.8 degree range.
        """
        if self.mode != ControlMode.AUTONOMOUS:
            return False

        now = time.time()
        if (now - self.last_move_time) * 1000 < self.COOLDOWN_MS:
            return False

        self.last_detection = detection
        self.last_detection_time = now

        error_x = detection.center_x - 0.5
        error_y = detection.center_y - 0.5

        if abs(error_x) < self.DEADZONE and abs(error_y) < self.DEADZONE:
            self._set_tracking_state(TrackingState.LOCKED)
            return False

        self._set_tracking_state(TrackingState.TRACKING)

        pan_delta = -error_x * self.FOV_HORIZONTAL * self.TRACKING_GAIN_PAN
        tilt_delta = -error_y * self.FOV_VERTICAL * self.TRACKING_GAIN_TILT

        pan_delta = max(-self.MAX_STEP_PAN, min(self.MAX_STEP_PAN, pan_delta))
        tilt_delta = max(-self.MAX_STEP_TILT, min(self.MAX_STEP_TILT, tilt_delta))

        self._update_position()
        target_pan = max(self.PAN_MIN, min(self.PAN_MAX, self.current_position.pan + pan_delta))
        target_tilt = max(self.TILT_MIN, min(self.TILT_MAX, self.current_position.tilt + tilt_delta))

        success = self.move_absolute(target_pan, target_tilt, speed=1.0)
        if success:
            self.logger.debug(f"Tracking: error=({error_x:.2f}, {error_y:.2f}) -> ({target_pan:.1f}, {target_tilt:.1f})")
        return success

    def check_target_lost(self) -> bool:
        """Check if target lost, start patrol if needed."""
        if self.mode != ControlMode.AUTONOMOUS:
            return False

        if time.time() - self.last_detection_time > self.TARGET_LOST_TIMEOUT:
            if self.tracking_state not in [TrackingState.PATROL, TrackingState.IDLE]:
                self._set_tracking_state(TrackingState.SEARCHING)
                self.logger.info("Target lost, starting patrol")
                return self.start_patrol()
        return False

    def start_patrol(self) -> bool:
        """Start 360 degree patrol scan."""
        self._set_tracking_state(TrackingState.PATROL)
        self.patrol_index = 0
        return self._patrol_next()

    def _patrol_next(self) -> bool:
        if self.tracking_state != TrackingState.PATROL:
            return False
        if self.patrol_index >= len(self.PATROL_POSITIONS):
            self.patrol_index = 0
        pan, tilt = self.PATROL_POSITIONS[self.patrol_index]
        self.patrol_index += 1
        return self.move_absolute(pan, tilt, speed=0.3)

    def patrol_step(self) -> bool:
        """Execute one patrol step (call when camera reaches position)."""
        if self.tracking_state != TrackingState.PATROL:
            return False
        pos = self.get_position()
        if not pos.moving:
            return self._patrol_next()
        return False

    def stop_tracking(self):
        """Stop tracking and return to idle."""
        self.stop()
        self._set_tracking_state(TrackingState.IDLE)
        self.last_detection = None

    def _set_tracking_state(self, state: TrackingState):
        if state != self.tracking_state:
            old_state = self.tracking_state
            self.tracking_state = state
            self.logger.info(f"Tracking: {old_state.name} -> {state.name}")
            if self.on_state_change:
                self.on_state_change(state)

    # -------------------------------------------------------------------------
    # Vision Integration (absorbed from PTZOrchestrator)
    # -------------------------------------------------------------------------

    def set_tracking_mode(self, mode: TrackingMode):
        """Set vision tracking mode (translates to ControlMode)."""
        old = self._tracking_mode
        self._tracking_mode = mode
        self._vision_stats["mode_changes"] += 1

        if mode == TrackingMode.AUTO_TRACK:
            self.set_mode(ControlMode.AUTONOMOUS)
        elif mode == TrackingMode.MANUAL:
            self.set_mode(ControlMode.MANUAL_OVERRIDE)
        else:
            self.set_mode(ControlMode.SAFE_MODE)

        self.logger.info(f"Tracking mode: {old.value} -> {mode.value}")

    def process_vision_event(self, event: VisionEvent) -> TrackingDecision:
        """Process vision event and track target."""
        self._vision_stats["total_events"] += 1
        now = time.time()

        if now - self._last_vision_time < self.MIN_VISION_CYCLE:
            return TrackingDecision(should_move=False, action="skip", reason="cycle_interval")
        self._last_vision_time = now

        if self._tracking_mode == TrackingMode.DISABLED:
            return TrackingDecision(should_move=False, action="none", reason="tracking_disabled")
        if self._tracking_mode == TrackingMode.MANUAL:
            return TrackingDecision(should_move=False, action="none", reason="manual_mode")

        if not event.detection_found:
            self.check_target_lost()
            return TrackingDecision(should_move=False, action="none", reason="no_detection")

        # Convert VisionEvent to Detection
        center_x_norm = event.target_center_x / event.frame_width
        detection = Detection(
            person_id="tracked_person",
            bbox=(event.target_center_x - 50, 200, 100, 300),
            center_x=center_x_norm,
            center_y=0.5,
            confidence=event.confidence
        )

        self._vision_stats["tracking_calls"] += 1
        moved = self.process_detection(detection)

        error_x = event.target_center_x - event.frame_center_x
        return TrackingDecision(
            should_move=moved,
            action="track" if moved else "deadzone",
            velocity=1.0 if moved else 0.0,
            duration=0.25,
            reason="tracking" if moved else "in_deadzone",
            error_x=error_x
        )

    def process_detections(self, detections: List[Dict], frame_width: int = 1920) -> TrackingDecision:
        """Process raw detections from perception."""
        self._vision_stats["total_events"] += 1
        now = time.time()

        if now - self._last_vision_time < self.MIN_VISION_CYCLE:
            return TrackingDecision(should_move=False, action="skip", reason="cycle_interval")
        self._last_vision_time = now

        if self._tracking_mode != TrackingMode.AUTO_TRACK:
            return TrackingDecision(should_move=False, action="none", reason=self._tracking_mode.value)

        if not detections:
            self.check_target_lost()
            return TrackingDecision(should_move=False, action="none", reason="no_detections")

        # Find best person detection
        person_dets = [d for d in detections if d.get("class", "") == "person"]
        if not person_dets:
            person_dets = detections

        best = max(person_dets, key=lambda d: d.get("confidence", 0))
        bbox = best.get("bbox", [0, 0, 0, 0])

        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if x2 <= 1.0 and y2 <= 1.0:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
            else:
                center_x = (x1 + x2) / 2 / frame_width
                center_y = (y1 + y2) / 2 / 1080
        else:
            center_x, center_y = 0.5, 0.5

        detection = Detection(
            person_id="detected_person",
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
            center_x=center_x,
            center_y=center_y,
            confidence=best.get("confidence", 0.5)
        )

        self._vision_stats["tracking_calls"] += 1
        moved = self.process_detection(detection)

        return TrackingDecision(
            should_move=moved,
            action="track" if moved else "deadzone",
            reason="tracking" if moved else "in_deadzone"
        )

    def get_vision_stats(self) -> Dict[str, Any]:
        """Get vision processing statistics."""
        return {
            **self._vision_stats,
            "tracking_mode": self._tracking_mode.value,
            "camera_connected": self.is_connected
        }

    # -------------------------------------------------------------------------
    # Mode Management
    # -------------------------------------------------------------------------

    def set_mode(self, mode: ControlMode):
        """Set control mode."""
        old_mode = self.mode
        self.mode = mode
        self.logger.info(f"Mode: {old_mode.name} -> {mode.name}")
        if mode == ControlMode.SAFE_MODE:
            self.stop()
            self._set_tracking_state(TrackingState.IDLE)

    def enable_autonomous(self):
        self.set_mode(ControlMode.AUTONOMOUS)

    def enable_manual(self):
        self.set_mode(ControlMode.MANUAL_OVERRIDE)

    def enable_safe(self):
        self.set_mode(ControlMode.SAFE_MODE)

    # -------------------------------------------------------------------------
    # Exclusive PTZ Access
    # -------------------------------------------------------------------------

    def acquire_exclusive(self, owner: str) -> bool:
        """Acquire exclusive PTZ control."""
        with self._exclusive_lock:
            if self._exclusive_owner is not None and self._exclusive_owner != owner:
                self.logger.warning(f"Exclusive PTZ held by '{self._exclusive_owner}', rejected '{owner}'")
                return False
            self._exclusive_owner = owner
            self.set_mode(ControlMode.MANUAL_OVERRIDE)
            self.logger.info(f"Exclusive PTZ granted to '{owner}'")
            return True

    def release_exclusive(self, owner: str):
        """Release exclusive PTZ control."""
        with self._exclusive_lock:
            if self._exclusive_owner == owner:
                self._exclusive_owner = None
                self.logger.info(f"Exclusive PTZ released by '{owner}'")
            else:
                self.logger.warning(f"Release by '{owner}' but owner is '{self._exclusive_owner}'")

    @property
    def exclusive_owner(self) -> Optional[str]:
        return self._exclusive_owner

    def has_exclusive_control(self, owner: str) -> bool:
        with self._exclusive_lock:
            return self._exclusive_owner == owner

    def is_exclusive_locked(self) -> bool:
        with self._exclusive_lock:
            return self._exclusive_owner is not None

    # -------------------------------------------------------------------------
    # Cloud Controls (via eWeLink Cloud Bridge)
    # -------------------------------------------------------------------------

    def set_night_mode(self, mode: NightMode) -> bool:
        """Set IR/Night mode (via cloud bridge)."""
        if self.cloud_bridge:
            try:
                return self.cloud_bridge.set_night(mode.name.lower())
            except Exception as e:
                self.logger.warning(f"Cloud night mode failed: {e}")
        self.logger.warning("Night mode not available (no cloud bridge)")
        return False

    def set_led_level(self, level: LEDLevel) -> bool:
        """Set LED brightness (via cloud bridge)."""
        if self.cloud_bridge:
            try:
                return self.cloud_bridge.set_led(level.value)
            except Exception as e:
                self.logger.warning(f"Cloud LED control failed: {e}")
        self.logger.warning("LED control not available (no cloud bridge)")
        return False

    def set_sleep_mode(self, enabled: bool) -> bool:
        """Enable/disable sleep/privacy mode (via cloud bridge)."""
        if self.cloud_bridge:
            try:
                return self.cloud_bridge.sleep_on() if enabled else self.cloud_bridge.sleep_off()
            except Exception as e:
                self.logger.warning(f"Cloud sleep mode failed: {e}")
        self.logger.warning("Sleep mode not available (no cloud bridge)")
        return False

    def set_mic_gain(self, gain: float) -> bool:
        """Set microphone gain (via cloud bridge)."""
        if self.cloud_bridge:
            try:
                return self.cloud_bridge.set_mic_gain(gain)
            except Exception as e:
                self.logger.warning(f"Cloud mic gain failed: {e}")
        self.logger.warning("Mic gain not available (no cloud bridge)")
        return False

    # -------------------------------------------------------------------------
    # Audio / RTSP
    # -------------------------------------------------------------------------

    def get_rtsp_url(self, stream: str = "main") -> str:
        """Get RTSP stream URL (main=1080p, sub=360p)."""
        channel = "ch0" if stream == "main" else "ch1"
        return f"rtsp://{self.username}:{self.password}@{self.camera_ip}:{self.rtsp_port}/av_stream/{channel}"

    def get_audio_stream_url(self) -> str:
        """Get audio stream URL (G.711 A-law, 8kHz, mono)."""
        return self.get_rtsp_url("main")

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def calibrate(self) -> CalibrationData:
        """Perform full PTZ calibration. WARNING: Camera will move through full range!"""
        self.logger.info("Starting PTZ calibration...")

        if not self.is_connected:
            return self.calibration

        limits = {}

        for name, pan, tilt in [
            ("right", -180.0, 0.0), ("left", 180.0, 0.0),
            ("down", 0.0, -90.0), ("up", 0.0, 110.0)
        ]:
            self.logger.info(f"Moving to {name} limit...")
            self.move_absolute(pan_deg=pan, tilt_deg=tilt, speed=0.5)
            self.wait_for_position(pan, tilt, timeout=15.0)
            time.sleep(0.5)
            pos = self.get_position()
            if name == "right":
                limits['pan_min'] = pos.pan
            elif name == "left":
                limits['pan_max'] = pos.pan
            elif name == "down":
                limits['tilt_min'] = pos.tilt
            elif name == "up":
                limits['tilt_max'] = pos.tilt
            self.logger.info(f"  {name}: {pos.pan:.1f} / {pos.tilt:.1f}")

        self.goto_home()

        self.calibration = CalibrationData(
            pan_min=limits['pan_min'], pan_max=limits['pan_max'],
            tilt_min=limits['tilt_min'], tilt_max=limits['tilt_max'],
            verified=True, timestamp=time.time()
        )

        self.logger.info(f"Calibration complete: "
                        f"Pan [{limits['pan_min']:.1f}, {limits['pan_max']:.1f}], "
                        f"Tilt [{limits['tilt_min']:.1f}, {limits['tilt_max']:.1f}]")
        return self.calibration

    def get_calibration(self) -> CalibrationData:
        return self.calibration

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> CameraStatus:
        """Get complete camera status."""
        pos = self.get_position()

        cloud_enabled = cloud_connected = False
        cloud_status_str = "disabled"

        if self.cloud_bridge:
            try:
                cs = self.cloud_bridge.get_stats()
                cloud_enabled = cs.get('enabled', False)
                cloud_connected = cs.get('connected', False)
                cloud_status_str = cs.get('status', 'unknown')
            except Exception:
                pass

        features_via_cloud = cloud_enabled and cloud_connected

        return CameraStatus(
            connected=self.is_connected,
            model=self.device_model,
            firmware=self.device_firmware,
            mode=self.mode.name,
            tracking_state=self.tracking_state.name,
            position=pos,
            pan_range=(self.PAN_MIN, self.PAN_MAX),
            tilt_range=(self.TILT_MIN, self.TILT_MAX),
            calibrated=self.calibration.verified,
            night_mode_available=features_via_cloud,
            led_control_available=features_via_cloud,
            sleep_mode_available=features_via_cloud,
            cloud_enabled=cloud_enabled,
            cloud_connected=cloud_connected,
            cloud_status=cloud_status_str,
            rtsp_url=self.get_rtsp_url("main"),
            last_move_time=self.last_move_time,
            last_detection_time=self.last_detection_time
        )

    def emergency_stop(self):
        """Emergency stop - disable tracking and stop camera."""
        self.logger.warning("EMERGENCY STOP")
        self._tracking_mode = TrackingMode.DISABLED
        self.set_mode(ControlMode.SAFE_MODE)
        self.stop()


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
    """Get or create SonoffCameraController singleton."""
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


# Alias for unified_camera_controller backwards compat
UnifiedCameraController = SonoffCameraController
