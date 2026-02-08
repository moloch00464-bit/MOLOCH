"""Shim: All code moved to camera.py"""
from core.hardware.camera import (
    SonoffCameraController as UnifiedCameraController,
    ControlMode,
    TrackingState,
    NightMode,
    LEDLevel,
    Detection,
    PTZPosition,
    CameraStatus,
    get_camera_controller,
)
