import os
#!/usr/bin/env python3
"""
Unified Camera Controller for Sonoff GK-200MP2-B
=================================================

Single unified class for ALL camera control operations.
No direct camera API calls should be made outside this class.

Supported Features (via ONVIF):
- PTZ control (pan, tilt, zoom) - Full 338° range
- Position tracking and status
- Continuous and absolute movement
- Patrol scanning
- Audio stream access (G.711 A-law, 8kHz, mono)

Unsupported Features (Camera Limitations):
- IR/Night mode control - Not available via ONVIF (no Imaging Service)
- LED level control - Not available via ONVIF (no Device I/O Service)
- Sleep/Privacy mode - Not available via ONVIF
- Mic gain control - Not available via ONVIF

Note: The Sonoff GK-200MP2-B does NOT support ONVIF Imaging or Device I/O services.
These features would require proprietary HTTP API or mobile app access.

Architecture:
  Application Code
         ↓
  UnifiedCameraController (this file)
         ↓
    ONVIF Camera API

Author: M.O.L.O.C.H. System
Date: 2026-02-07
"""

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
    print("WARNING: onvif-zeep not installed. Run: pip install onvif-zeep")

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
    print("WARNING: Cloud bridge not available")


# =============================================================================
# Enums and Data Classes
# =============================================================================

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


class NightMode(Enum):
    """IR/Night mode settings (NOT SUPPORTED - stub for future)."""
    AUTO = auto()           # Automatic day/night switching
    DAY = auto()            # Force day mode (IR off)
    NIGHT = auto()          # Force night mode (IR on)


class LEDLevel(Enum):
    """LED brightness levels (NOT SUPPORTED - stub for future)."""
    OFF = 0                 # LED off
    LOW = 1                 # Low brightness
    MEDIUM = 2              # Medium brightness
    HIGH = 3                # High brightness


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
class CameraStatus:
    """Complete camera status."""
    # Connection
    connected: bool = False
    model: str = "Unknown"
    firmware: str = "Unknown"

    # Control state
    mode: str = "UNKNOWN"
    tracking_state: str = "IDLE"

    # Position
    position: Optional[PTZPosition] = None

    # Calibration
    pan_range: Tuple[float, float] = (-168.4, 170.0)
    tilt_range: Tuple[float, float] = (-78.0, 78.8)
    calibrated: bool = True

    # Features (available on this camera)
    ptz_available: bool = True
    audio_available: bool = True

    # Features (NOT available on this camera via ONVIF, may be available via cloud)
    night_mode_available: bool = False
    led_control_available: bool = False
    sleep_mode_available: bool = False
    mic_gain_available: bool = False

    # Cloud bridge
    cloud_enabled: bool = False
    cloud_connected: bool = False
    cloud_status: str = "disabled"

    # Streams
    rtsp_url: str = ""
    audio_codec: str = "G711"
    audio_sample_rate: int = 8000

    # Timing
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
                'total_pan': self.pan_range[1] - self.pan_range[0],
                'total_tilt': self.tilt_range[1] - self.tilt_range[0],
                'calibrated': self.calibrated
            },
            'features': {
                'ptz': self.ptz_available,
                'audio': self.audio_available,
                'night_mode': self.night_mode_available,
                'led_control': self.led_control_available,
                'sleep_mode': self.sleep_mode_available,
                'mic_gain': self.mic_gain_available
            },
            'cloud': {
                'enabled': self.cloud_enabled,
                'connected': self.cloud_connected,
                'status': self.cloud_status
            },
            'streams': {
                'rtsp_url': self.rtsp_url,
                'audio_codec': self.audio_codec,
                'audio_sample_rate': self.audio_sample_rate
            },
            'timing': {
                'last_move_time': self.last_move_time,
                'last_detection_time': self.last_detection_time
            }
        }


# =============================================================================
# Unified Camera Controller
# =============================================================================

class UnifiedCameraController:
    """
    Unified camera controller for Sonoff GK-200MP2-B.

    All camera operations should go through this class.
    Consolidates PTZ, audio, and status into single interface.

    IMPORTANT: This camera does NOT support ONVIF Imaging or Device I/O services.
    IR/Night mode, LED control, sleep mode, and mic gain are NOT available.
    """

    # Hardware limits (calibrated 2026-02-04)
    PAN_MIN = -168.4   # Degrees (right limit)
    PAN_MAX = 170.0    # Degrees (left limit)
    TILT_MIN = -78.0   # Degrees (down limit)
    TILT_MAX = 78.8    # Degrees (up limit)

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

    # Field of view (approximate)
    FOV_HORIZONTAL = 110.0    # Degrees
    FOV_VERTICAL = 65.0       # Degrees

    # Patrol scan positions
    PATROL_POSITIONS = [
        (0.0, 0.0),           # Center
        (-84.0, 0.0),         # Half-right
        (-168.0, 0.0),        # Full right
        (-84.0, 30.0),        # Half-right, up
        (0.0, 30.0),          # Center, up
        (84.0, 30.0),         # Half-left, up
        (170.0, 0.0),         # Full left
        (84.0, 0.0),          # Half-left
    ]

    def __init__(
        self,
        camera_ip: str = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP"),
        username: str = os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME"),
        password: str = os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME"),
        onvif_port: int = 80,
        rtsp_port: int = 554,
        log_level: int = logging.INFO
    ):
        """
        Initialize unified camera controller.

        Args:
            camera_ip: Camera IP address
            username: ONVIF username
            password: ONVIF password
            onvif_port: ONVIF service port
            rtsp_port: RTSP stream port
            log_level: Logging level
        """
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.onvif_port = onvif_port
        self.rtsp_port = rtsp_port

        # Logging
        self.logger = logging.getLogger("UnifiedCameraController")
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

    # =========================================================================
    # Cloud Bridge Configuration
    # =========================================================================

    def _load_cloud_config(self):
        """Load cloud bridge configuration from file."""
        if not CLOUD_BRIDGE_AVAILABLE:
            self.logger.debug("Cloud bridge not available")
            return

        try:
            # Determine config file path
            config_path = Path(__file__).parent.parent.parent / "config" / "camera_cloud.json"

            if not config_path.exists():
                self.logger.debug(f"Cloud config not found: {config_path}")
                return

            # Load config
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            cloud_enabled = config_data.get('cloud_enabled', False)

            if not cloud_enabled:
                self.logger.info("Cloud bridge disabled in config")
                return

            # Parse cloud config
            cloud_config_data = config_data.get('cloud_config', {})
            cloud_config = CloudConfig(
                enabled=True,
                api_base_url=cloud_config_data.get('api_base_url', ''),
                app_id=cloud_config_data.get('app_id', ''),
                app_secret=cloud_config_data.get('app_secret', ''),
                device_id=cloud_config_data.get('device_id', ''),
                username=cloud_config_data.get('username', ''),
                password=cloud_config_data.get('password', ''),
                timeout=cloud_config_data.get('timeout', 3.0),
                retry_count=cloud_config_data.get('retry_count', 1),
                token_refresh_margin=cloud_config_data.get('token_refresh_margin', 300)
            )

            # Create cloud bridge
            self.cloud_bridge = CameraCloudBridgeSync(cloud_config)
            self.logger.info("Cloud bridge configured")

        except Exception as e:
            self.logger.warning(f"Failed to load cloud config: {e}")
            self.cloud_bridge = None

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to camera via ONVIF.

        Returns:
            True if connection successful
        """
        if not ONVIF_AVAILABLE:
            self.logger.error("ONVIF library not available")
            return False

        try:
            self.logger.info(f"Connecting to camera at {self.camera_ip}...")

            # Create ONVIF camera client
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

            # Get device info
            device_info = self.device_service.GetDeviceInformation()
            self.device_model = device_info.Model
            self.device_firmware = device_info.FirmwareVersion
            self.logger.info(f"Device: {device_info.Manufacturer} {self.device_model} "
                           f"(FW {self.device_firmware})")

            # Get media profile
            profiles = self.media_service.GetProfiles()
            if profiles:
                self.profile_token = profiles[0].token
                self.logger.info(f"Using media profile: {self.profile_token}")
            else:
                self.logger.error("No media profiles found")
                return False

            # Get initial position
            self._update_position()

            self._connected = True
            self.logger.info("✓ Camera connected successfully")

            # Connect cloud bridge (non-blocking, graceful failure)
            if self.cloud_bridge:
                try:
                    self.logger.info("Connecting cloud bridge...")
                    cloud_success = self.cloud_bridge.connect()
                    if cloud_success:
                        self.logger.info("✓ Cloud bridge connected")
                    else:
                        self.logger.warning("Cloud bridge connection failed - continuing with ONVIF only")
                except Exception as e:
                    self.logger.warning(f"Cloud bridge error: {e} - continuing with ONVIF only")

            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera."""
        self.stop()

        # Disconnect cloud bridge
        if self.cloud_bridge:
            try:
                self.cloud_bridge.disconnect()
                self.logger.info("Cloud bridge disconnected")
            except Exception as e:
                self.logger.warning(f"Cloud bridge disconnect error: {e}")

        self._connected = False
        self.camera = None
        self.ptz_service = None
        self.media_service = None
        self.device_service = None
        self.logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.camera is not None

    # =========================================================================
    # PTZ Control - Position Management
    # =========================================================================

    def _update_position(self) -> Optional[PTZPosition]:
        """Get current position from camera."""
        if not self.ptz_service or not self.profile_token:
            return None

        try:
            status = self.ptz_service.GetStatus({'ProfileToken': self.profile_token})
            status_dict = serialize_object(status)

            pos = status_dict.get('Position', {})
            pan_tilt = pos.get('PanTilt', {})

            # Sonoff returns degrees directly (not normalized -1 to 1)
            pan_deg = float(pan_tilt.get('x', 0))
            tilt_deg = float(pan_tilt.get('y', 0))

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
        """
        Get current PTZ position.

        Returns:
            PTZPosition with current pan, tilt, zoom
        """
        self._update_position()
        return self.current_position

    # =========================================================================
    # PTZ Control - Movement Commands
    # =========================================================================

    def move_absolute(
        self,
        pan_deg: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        speed: float = 0.5
    ) -> bool:
        """
        Move to absolute position.

        Args:
            pan_deg: Target pan in degrees (-168.4 to 170.0)
            tilt_deg: Target tilt in degrees (-78.0 to 78.8)
            speed: Movement speed (0.1-1.0)

        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            self.logger.warning("Not connected")
            return False

        if self.mode == ControlMode.SAFE_MODE:
            self.logger.warning("Movement blocked: SAFE_MODE active")
            return False

        # Apply limits
        if pan_deg is not None:
            pan_deg = max(self.PAN_MIN, min(self.PAN_MAX, pan_deg))
        else:
            pan_deg = self.current_position.pan

        if tilt_deg is not None:
            tilt_deg = max(self.TILT_MIN, min(self.TILT_MAX, tilt_deg))
        else:
            tilt_deg = self.current_position.tilt

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
            # Sonoff camera quirk: returns "Invalid position" but still executes
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
            speed: Movement speed (0.1-1.0)

        Returns:
            True if command sent successfully
        """
        self._update_position()
        target_pan = self.current_position.pan + pan_delta
        target_tilt = self.current_position.tilt + tilt_delta
        return self.move_absolute(target_pan, target_tilt, speed)

    def continuous_move(
        self,
        vel_pan: float,
        vel_tilt: float,
        vel_zoom: float = 0.0,
        timeout_sec: float = 1.0
    ) -> bool:
        """
        Continuous velocity-based movement.

        Args:
            vel_pan: Pan velocity (-1 to 1), negative=right, positive=left
            vel_tilt: Tilt velocity (-1 to 1), negative=down, positive=up
            vel_zoom: Zoom velocity (-1 to 1)
            timeout_sec: Duration in seconds

        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            self.logger.warning("Not connected")
            return False

        if self.mode == ControlMode.SAFE_MODE:
            self.logger.warning("Movement blocked: SAFE_MODE active")
            return False

        try:
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile_token
            request.Velocity = {
                'PanTilt': {'x': vel_pan, 'y': vel_tilt},
                'Zoom': {'x': vel_zoom}
            }
            request.Timeout = f'PT{int(timeout_sec)}S'

            self.ptz_service.ContinuousMove(request)
            self.last_move_time = time.time()
            self.logger.debug(f"ContinuousMove: vel=({vel_pan:+.3f}, {vel_tilt:+.3f})")
            return True

        except Exception as e:
            self.logger.error(f"ContinuousMove failed: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop all movement.

        Returns:
            True if command sent successfully
        """
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

    # =========================================================================
    # PTZ Control - Tracking System
    # =========================================================================

    def process_detection(self, detection: Detection) -> bool:
        """
        Process detection and track target.

        Args:
            detection: Detection from vision system

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

        # Calculate error from center
        error_x = detection.center_x - 0.5  # Positive = target right of center
        error_y = detection.center_y - 0.5  # Positive = target below center

        # Check deadzone
        if abs(error_x) < self.DEADZONE and abs(error_y) < self.DEADZONE:
            self._set_tracking_state(TrackingState.LOCKED)
            return False

        self._set_tracking_state(TrackingState.TRACKING)

        # Convert error to degrees
        pan_delta = -error_x * self.FOV_HORIZONTAL * self.TRACKING_GAIN_PAN
        tilt_delta = -error_y * self.FOV_VERTICAL * self.TRACKING_GAIN_TILT

        # Limit step size
        pan_delta = max(-self.MAX_STEP_PAN, min(self.MAX_STEP_PAN, pan_delta))
        tilt_delta = max(-self.MAX_STEP_TILT, min(self.MAX_STEP_TILT, tilt_delta))

        # Get current position
        self._update_position()

        # Calculate target
        target_pan = self.current_position.pan + pan_delta
        target_tilt = self.current_position.tilt + tilt_delta

        # Execute move
        success = self.move_absolute(target_pan, target_tilt, speed=1.0)

        if success:
            self.logger.debug(
                f"Tracking: error=({error_x:.2f}, {error_y:.2f}) -> "
                f"target=({target_pan:.1f}°, {target_tilt:.1f}°)"
            )

        return success

    def check_target_lost(self) -> bool:
        """
        Check if target lost and start patrol if needed.

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
                self.logger.info("Target lost, starting patrol")
                return self.start_patrol()

        return False

    def start_patrol(self) -> bool:
        """Start patrol scan."""
        self._set_tracking_state(TrackingState.PATROL)
        self.patrol_index = 0
        return self._patrol_next()

    def _patrol_next(self) -> bool:
        """Move to next patrol position."""
        if self.tracking_state != TrackingState.PATROL:
            return False

        if self.patrol_index >= len(self.PATROL_POSITIONS):
            self.patrol_index = 0

        pan, tilt = self.PATROL_POSITIONS[self.patrol_index]
        self.patrol_index += 1

        self.logger.debug(f"Patrol {self.patrol_index}/{len(self.PATROL_POSITIONS)}")
        return self.move_absolute(pan, tilt, speed=0.3)

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
            self.logger.info(f"Tracking: {old_state.name} -> {state.name}")

            if self.on_state_change:
                self.on_state_change(state)

    # =========================================================================
    # Control Modes
    # =========================================================================

    def set_mode(self, mode: ControlMode):
        """Set control mode."""
        old_mode = self.mode
        self.mode = mode
        self.logger.info(f"Mode: {old_mode.name} -> {mode.name}")

        if mode == ControlMode.SAFE_MODE:
            self.stop()
            self._set_tracking_state(TrackingState.IDLE)

    def enable_autonomous(self):
        """Enable autonomous tracking."""
        self.set_mode(ControlMode.AUTONOMOUS)

    def enable_manual(self):
        """Enable manual control."""
        self.set_mode(ControlMode.MANUAL_OVERRIDE)

    def enable_safe_mode(self):
        """Enable safe mode (no movement)."""
        self.set_mode(ControlMode.SAFE_MODE)

    # =========================================================================
    # Exclusive Access Control
    # =========================================================================

    def acquire_exclusive(self, owner: str) -> bool:
        """
        Acquire exclusive PTZ control.

        Args:
            owner: Identifier of the requesting owner

        Returns:
            True if granted, False if already owned
        """
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
        """Get current exclusive owner."""
        return self._exclusive_owner

    def has_exclusive_control(self, owner: str) -> bool:
        """
        Check if owner has exclusive control.

        Args:
            owner: Identifier to check

        Returns:
            True if owner has exclusive control
        """
        with self._exclusive_lock:
            return self._exclusive_owner == owner

    def is_exclusive_locked(self) -> bool:
        """
        Check if camera is under exclusive control.

        Returns:
            True if any owner has exclusive control
        """
        with self._exclusive_lock:
            return self._exclusive_owner is not None

    # =========================================================================
    # Audio Control
    # =========================================================================

    def get_rtsp_url(self, stream: str = "main") -> str:
        """
        Get RTSP stream URL.

        Args:
            stream: "main" (1920x1080) or "sub" (640x360)

        Returns:
            RTSP URL string
        """
        channel = "ch0" if stream == "main" else "ch1"
        return f"rtsp://{self.username}:{self.password}@{self.camera_ip}:{self.rtsp_port}/av_stream/{channel}"

    def get_audio_stream_url(self) -> str:
        """
        Get RTSP URL for audio stream.

        Returns:
            RTSP URL (same as main stream, includes audio)
        """
        return self.get_rtsp_url("main")

    # =========================================================================
    # IR/Night Mode Control (NOT SUPPORTED)
    # =========================================================================

    def set_night_mode(self, mode: NightMode) -> bool:
        """
        Set IR/Night mode.

        Tries cloud bridge if available, otherwise returns False.
        NOT SUPPORTED via ONVIF (no Imaging Service).

        Args:
            mode: NightMode enum value

        Returns:
            True if successful (via cloud), False otherwise
        """
        # Try cloud bridge
        if self.cloud_bridge:
            try:
                mode_str = mode.name.lower()  # AUTO, DAY, NIGHT -> auto, day, night
                return self.cloud_bridge.set_night(mode_str)
            except Exception as e:
                self.logger.warning(f"Cloud bridge night mode failed: {e}")

        # Not available
        self.logger.warning("Night mode control NOT SUPPORTED (no ONVIF Imaging Service, no cloud)")
        self.logger.info(f"Requested mode: {mode.name} - requires cloud API")
        return False

    def get_night_mode(self) -> Optional[NightMode]:
        """
        Get current IR/Night mode.

        NOT SUPPORTED: Sonoff GK-200MP2-B does not support ONVIF Imaging Service.

        Returns:
            None (not supported)
        """
        self.logger.warning("Night mode query NOT SUPPORTED (no ONVIF Imaging Service)")
        return None

    # =========================================================================
    # LED Control (NOT SUPPORTED)
    # =========================================================================

    def set_led_level(self, level: LEDLevel) -> bool:
        """
        Set LED brightness level.

        Tries cloud bridge if available, otherwise returns False.
        NOT SUPPORTED via ONVIF (no Device I/O Service).

        Args:
            level: LEDLevel enum value (OFF, LOW, MEDIUM, HIGH)

        Returns:
            True if successful (via cloud), False otherwise
        """
        # Try cloud bridge
        if self.cloud_bridge:
            try:
                level_int = level.value  # OFF=0, LOW=1, MEDIUM=2, HIGH=3
                return self.cloud_bridge.set_led(level_int)
            except Exception as e:
                self.logger.warning(f"Cloud bridge LED control failed: {e}")

        # Not available
        self.logger.warning("LED control NOT SUPPORTED (no ONVIF Device I/O Service, no cloud)")
        self.logger.info(f"Requested level: {level.name} - requires cloud API")
        return False

    def get_led_level(self) -> Optional[LEDLevel]:
        """
        Get current LED brightness level.

        NOT SUPPORTED: Sonoff GK-200MP2-B does not support ONVIF Device I/O Service.

        Returns:
            None (not supported)
        """
        self.logger.warning("LED query NOT SUPPORTED (no ONVIF Device I/O Service)")
        return None

    # =========================================================================
    # Sleep/Privacy Mode (NOT SUPPORTED)
    # =========================================================================

    def set_sleep_mode(self, enabled: bool) -> bool:
        """
        Enable/disable sleep/privacy mode.

        Tries cloud bridge if available, otherwise returns False.
        NOT SUPPORTED via ONVIF (no Privacy API).

        Args:
            enabled: True to enable sleep mode, False to disable

        Returns:
            True if successful (via cloud), False otherwise
        """
        # Try cloud bridge
        if self.cloud_bridge:
            try:
                if enabled:
                    return self.cloud_bridge.sleep_on()
                else:
                    return self.cloud_bridge.sleep_off()
            except Exception as e:
                self.logger.warning(f"Cloud bridge sleep mode failed: {e}")

        # Not available
        self.logger.warning("Sleep/Privacy mode NOT SUPPORTED (no ONVIF API, no cloud)")
        self.logger.info(f"Requested: {'ENABLE' if enabled else 'DISABLE'} - requires cloud API")
        return False

    def get_sleep_mode(self) -> Optional[bool]:
        """
        Get current sleep/privacy mode status.

        NOT SUPPORTED: Sonoff GK-200MP2-B does not expose privacy mode via ONVIF.

        Returns:
            None (not supported)
        """
        self.logger.warning("Sleep/Privacy mode query NOT SUPPORTED (no ONVIF API)")
        return None

    # =========================================================================
    # Microphone Gain Control (NOT SUPPORTED)
    # =========================================================================

    def set_mic_gain(self, gain: float) -> bool:
        """
        Set microphone gain level.

        Tries cloud bridge if available, otherwise returns False.
        NOT SUPPORTED via ONVIF (no Audio Input Config).

        Args:
            gain: Gain value (0.0-1.0)

        Returns:
            True if successful (via cloud), False otherwise
        """
        # Try cloud bridge
        if self.cloud_bridge:
            try:
                return self.cloud_bridge.set_mic_gain(gain)
            except Exception as e:
                self.logger.warning(f"Cloud bridge mic gain failed: {e}")

        # Not available
        self.logger.warning("Mic gain control NOT SUPPORTED (no ONVIF Audio Config, no cloud)")
        self.logger.info(f"Requested gain: {gain:.2f} - requires cloud API")
        return False

    def get_mic_gain(self) -> Optional[float]:
        """
        Get current microphone gain level.

        NOT SUPPORTED: Sonoff GK-200MP2-B does not expose audio input config via ONVIF.

        Returns:
            None (not supported)
        """
        self.logger.warning("Mic gain query NOT SUPPORTED (no ONVIF Audio Config)")
        return None

    # =========================================================================
    # Status Query
    # =========================================================================

    def get_status(self) -> CameraStatus:
        """
        Get complete camera status.

        Returns:
            CameraStatus object with all available information
        """
        # Update position
        pos = self.get_position()

        # Check cloud bridge status
        cloud_enabled = False
        cloud_connected = False
        cloud_status_str = "disabled"

        if self.cloud_bridge:
            try:
                cloud_stats = self.cloud_bridge.get_stats()
                cloud_enabled = cloud_stats.get('enabled', False)
                cloud_connected = cloud_stats.get('connected', False)
                cloud_status_str = cloud_stats.get('status', 'unknown')
            except:
                pass

        # Features available via cloud
        features_via_cloud = cloud_enabled and cloud_connected

        # Build status
        status = CameraStatus(
            connected=self.is_connected,
            model=self.device_model,
            firmware=self.device_firmware,
            mode=self.mode.name,
            tracking_state=self.tracking_state.name,
            position=pos,
            pan_range=(self.PAN_MIN, self.PAN_MAX),
            tilt_range=(self.TILT_MIN, self.TILT_MAX),
            calibrated=True,
            ptz_available=True,
            audio_available=True,
            # Features available via cloud bridge
            night_mode_available=features_via_cloud,
            led_control_available=features_via_cloud,
            sleep_mode_available=features_via_cloud,
            mic_gain_available=features_via_cloud,
            # Cloud status
            cloud_enabled=cloud_enabled,
            cloud_connected=cloud_connected,
            cloud_status=cloud_status_str,
            # Streams
            rtsp_url=self.get_rtsp_url("main"),
            audio_codec="G711",
            audio_sample_rate=8000,
            last_move_time=self.last_move_time,
            last_detection_time=self.last_detection_time
        )

        return status


# =============================================================================
# Singleton
# =============================================================================

_camera_controller: Optional[UnifiedCameraController] = None


def get_camera_controller(
    camera_ip: str = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP"),
    username: str = os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME"),
    password: str = os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME"),
    auto_connect: bool = True
) -> UnifiedCameraController:
    """
    Get or create UnifiedCameraController singleton.

    Args:
        camera_ip: Camera IP address
        username: ONVIF username
        password: ONVIF password
        auto_connect: Connect automatically if not connected

    Returns:
        UnifiedCameraController instance
    """
    global _camera_controller
    if _camera_controller is None:
        _camera_controller = UnifiedCameraController(
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
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Unified Camera Controller - Quick Test")
    print("=" * 80)

    # Create controller
    controller = UnifiedCameraController()

    # Connect
    print("\n[1] Connecting...")
    if controller.connect():
        print("✓ Connected")
    else:
        print("✗ Connection failed")
        exit(1)

    # Get status
    print("\n[2] Camera Status:")
    status = controller.get_status()
    print(f"  Model: {status.model}")
    print(f"  Firmware: {status.firmware}")
    print(f"  Position: Pan={status.position.pan:.1f}°, Tilt={status.position.tilt:.1f}°")
    print(f"  Pan Range: {status.pan_range[0]:.1f}° to {status.pan_range[1]:.1f}°")
    print(f"  Tilt Range: {status.tilt_range[0]:.1f}° to {status.tilt_range[1]:.1f}°")
    print(f"  RTSP URL: {status.rtsp_url}")

    print("\n[3] Feature Support:")
    print(f"  PTZ Control: {'✓' if status.ptz_available else '✗'}")
    print(f"  Audio Stream: {'✓' if status.audio_available else '✗'}")
    print(f"  Night Mode: {'✓' if status.night_mode_available else '✗'}")
    print(f"  LED Control: {'✓' if status.led_control_available else '✗'}")
    print(f"  Sleep Mode: {'✓' if status.sleep_mode_available else '✗'}")
    print(f"  Mic Gain: {'✓' if status.mic_gain_available else '✗'}")

    # Test movement
    print("\n[4] Testing PTZ movement...")
    print("  Moving to center...")
    controller.goto_home()
    time.sleep(2)

    print("  Moving right...")
    controller.move_relative(-30.0, 0.0)
    time.sleep(2)

    print("  Moving left...")
    controller.move_relative(60.0, 0.0)
    time.sleep(2)

    print("  Returning to center...")
    controller.goto_home()

    # Disconnect
    print("\n[5] Disconnecting...")
    controller.disconnect()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
