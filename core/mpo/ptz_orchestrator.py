"""Shim: Orchestrator logic absorbed into camera.py"""
from core.hardware.camera import (
    TrackingMode,
    VisionEvent,
    TrackingDecision,
    SonoffCameraController,
    get_camera_controller,
)

import logging

logger = logging.getLogger(__name__)


class PTZOrchestrator:
    """
    Thin wrapper around SonoffCameraController for backwards compat.
    All logic now lives in camera.py.
    """

    def __init__(self, camera_controller=None):
        self.camera: SonoffCameraController = camera_controller

    def set_ptz(self, controller):
        self.set_camera(controller)

    def set_camera(self, camera_controller):
        self.camera = camera_controller
        logger.info("Camera controller connected to Orchestrator")

    def set_mode(self, mode: TrackingMode):
        if self.camera:
            self.camera.set_tracking_mode(mode)

    def process_vision_event(self, event: VisionEvent) -> TrackingDecision:
        if not self.camera:
            return TrackingDecision(should_move=False, action="none", reason="no_camera")
        return self.camera.process_vision_event(event)

    def process_detections(self, detections, frame_width: int = 1920) -> TrackingDecision:
        if not self.camera:
            return TrackingDecision(should_move=False, action="none", reason="no_camera")
        return self.camera.process_detections(detections, frame_width)

    def get_stats(self):
        if self.camera:
            return self.camera.get_vision_stats()
        return {"tracking_mode": "disabled", "camera_connected": False}

    def emergency_stop(self):
        if self.camera:
            self.camera.emergency_stop()


_orchestrator = None

def get_ptz_orchestrator() -> PTZOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PTZOrchestrator()
    return _orchestrator
