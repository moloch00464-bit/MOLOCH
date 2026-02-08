import os
#\!/usr/bin/env python3
import time, logging, threading
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ControlMode(Enum):
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    CENTERING = "centering"

@dataclass
class PTZAction:
    timestamp: float
    action: str
    pan: float = 0.0
    tilt: float = 0.0
    duration: float = 0.0
    success: bool = False
    error: str = ""
    mode: str = ""
    reason: str = ""
    def to_log(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        status = "OK" if self.success else f"FAIL:{self.error}"
        return f"[PTZ {ts}] {self.action} mode={self.mode} vel={self.pan:+.2f} dur={self.duration:.2f}s reason={self.reason} {status}"

class PTZController:
    MAX_VELOCITY = 0.4
    MIN_VELOCITY = 0.05
    MAX_DURATION = 0.5
    MIN_DURATION = 0.1
    COOLDOWN_MS = 300
    
    def __init__(self, host=os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP"), port=80, user=os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME"), password=os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME")):
        self.host, self.port, self.user, self.password = host, port, user, password
        self._camera = self._ptz = self._token = None
        self._lock = threading.Lock()
        self._is_moving = False
        self.connected = False
        self._history = []
        self._control_mode = ControlMode.MANUAL
        self._manual_override = True
        self._autonomous_enabled = False
        self._last_move_time = 0
        self._centering_active = False
        self._stop_centering = threading.Event()
        logger.info(f"PTZController init (host={host})")
    
    def connect(self) -> bool:
        with self._lock:
            if self.connected: return True
            try:
                from onvif import ONVIFCamera
                self._camera = ONVIFCamera(self.host, self.port, self.user, self.password)
                self._ptz = self._camera.create_ptz_service()
                media = self._camera.create_media_service()
                self._token = media.GetProfiles()[0].token
                self.connected = True
                logger.info(f"PTZ connected (token={self._token})")
                return True
            except Exception as e:
                logger.error(f"PTZ connect failed: {e}")
                return False
    
    def set_manual_mode(self):
        self._manual_override, self._autonomous_enabled = True, False
        self._control_mode = ControlMode.MANUAL
        self.stop()
        logger.info("[PTZ MODE] MANUAL")
    
    def set_autonomous_mode(self):
        self._manual_override, self._autonomous_enabled = False, True
        self._control_mode = ControlMode.AUTONOMOUS
        logger.info("[PTZ MODE] AUTONOMOUS")
    
    def is_manual_mode(self): return self._manual_override
    def is_autonomous_mode(self): return self._autonomous_enabled
    @property
    def control_mode(self): return self._control_mode
    
    def _clamp_velocity(self, v): return max(-self.MAX_VELOCITY, min(self.MAX_VELOCITY, v))
    def _clamp_duration(self, d): return max(self.MIN_DURATION, min(self.MAX_DURATION, d))
    def _check_cooldown(self): return (time.time() - self._last_move_time) * 1000 >= self.COOLDOWN_MS
    
    def move(self, pan, tilt=0, duration=None, reason="", source="unknown"):
        if duration is None: duration = self.MAX_DURATION
        pan, tilt = self._clamp_velocity(pan), self._clamp_velocity(tilt)
        duration = self._clamp_duration(duration)
        action = PTZAction(time.time(), "move", pan, tilt, duration, mode=self._control_mode.value, reason=reason)
        if source == "manual" and not self._manual_override:
            action.error = "manual_disabled"; logger.warning(action.to_log()); return action
        if source == "mpo" and not self._autonomous_enabled:
            action.error = "mpo_disabled"; logger.warning(action.to_log()); return action
        with self._lock:
            if not self.connected: action.error = "not_connected"; return action
            if self._is_moving: action.error = "busy"; return action
            if not self._check_cooldown(): action.error = "cooldown"; return action
            try:
                self._is_moving = True
                req = self._ptz.create_type("ContinuousMove")
                req.ProfileToken = self._token
                req.Velocity = {"PanTilt": {"x": pan, "y": tilt}}
                self._ptz.ContinuousMove(req)
                time.sleep(duration)
                self._stop_internal()
                action.success = True
                self._last_move_time = time.time()
                self._history.append(action)
            except Exception as e: action.error = str(e)
            finally: self._is_moving = False
        logger.info(action.to_log())
        return action
    
    def move_manual(self, direction, duration=0.3):
        dirs = {"left": (-self.MAX_VELOCITY, 0), "right": (self.MAX_VELOCITY, 0), "up": (0, self.MAX_VELOCITY), "down": (0, -self.MAX_VELOCITY)}
        if direction not in dirs: return PTZAction(time.time(), "move", error="invalid")
        return self.move(*dirs[direction], duration, reason=f"manual_{direction}", source="manual")
    
    def move_autonomous(self, pan, tilt=0, duration=0.3, reason="tracking"):
        return self.move(pan, tilt, duration, reason=reason, source="mpo")
    
    def stop(self):
        action = PTZAction(time.time(), "stop", mode=self._control_mode.value)
        if self._centering_active: self._stop_centering.set(); self._centering_active = False
        with self._lock:
            if not self.connected: action.error = "not_connected"; return action
            try: self._stop_internal(); action.success = True; self._is_moving = False
            except Exception as e: action.error = str(e)
        logger.info(action.to_log())
        return action
    
    def center(self):
        action = PTZAction(time.time(), "center", mode="centering", reason="home")
        self.stop()
        with self._lock:
            if not self.connected: action.error = "not_connected"; return action
            try:
                req = self._ptz.create_type("GotoHomePosition")
                req.ProfileToken = self._token
                self._ptz.GotoHomePosition(req)
                action.success = True
            except Exception as e: action.error = str(e); action.reason = "home_failed"
        logger.info(action.to_log())
        return action
    
    def _stop_internal(self):
        req = self._ptz.create_type("Stop")
        req.ProfileToken = self._token
        req.PanTilt = req.Zoom = True
        self._ptz.Stop(req)
    
    def is_moving(self):
        with self._lock: return self._is_moving
    def is_centering(self): return self._centering_active
    def get_history(self, n=10): return self._history[-n:]
    def emergency_stop(self):
        logger.warning("[PTZ] EMERGENCY STOP")
        self._manual_override, self._autonomous_enabled = True, False
        self._control_mode = ControlMode.MANUAL
        self.stop()

_ctrl = None
def get_ptz_controller():
    global _ctrl
    if _ctrl is None: _ctrl = PTZController()
    return _ctrl
