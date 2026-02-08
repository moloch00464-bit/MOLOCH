#!/usr/bin/env python3
"""
M.O.L.O.C.H. PTZ Orchestrator - Zentrale Entscheidungslogik
Delegiert alle Kamera-Operationen an SonoffCameraController.

FIXED 2026-02-04: API-Mismatch behoben - verwendet jetzt korrekt Detection-Objekte
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TrackingMode(Enum):
    DISABLED = "disabled"
    MANUAL = "manual"
    AUTO_TRACK = "auto_track"
    SEARCH = "search"


@dataclass
class VisionEvent:
    """Daten vom Vision-Modul."""
    detection_found: bool
    target_center_x: int = 0
    frame_center_x: int = 960
    frame_width: int = 1920
    confidence: float = 0.0
    bbox: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrackingDecision:
    """Entscheidung des Orchestrators."""
    should_move: bool = False
    action: str = "none"
    velocity: float = 0.0
    duration: float = 0.0
    reason: str = ""
    error_x: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_log(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return f"[{ts}] decision={self.action} error_x={self.error_x:+d} vel={self.velocity:.2f} reason={self.reason}"


class PTZOrchestrator:
    """
    Zentrale Entscheidungsinstanz - Delegiert an SonoffCameraController.
    
    Architecture:
        Vision -> PTZOrchestrator -> SonoffCameraController -> Camera
        GUI -> PTZOrchestrator (mode change only)
    """
    
    MIN_CYCLE_INTERVAL = 0.1
    
    def __init__(self, camera_controller=None):
        self.camera = camera_controller
        self.mode = TrackingMode.DISABLED
        self._last_decision_time = 0
        self._last_decision = None
        self._stats = {"total_events": 0, "tracking_calls": 0, "mode_changes": 0}
        logger.info("PTZOrchestrator initialized")
    
    def set_ptz(self, controller):
        """Set camera controller (legacy compat - use set_camera)."""
        self.set_camera(controller)
    
    def set_camera(self, camera_controller):
        """Set the SonoffCameraController."""
        self.camera = camera_controller
        logger.info("Camera controller connected to Orchestrator")
    
    def set_mode(self, mode: TrackingMode):
        """Set tracking mode - updates camera controller mode."""
        from core.hardware.sonoff_camera_controller import ControlMode
        
        old_mode = self.mode
        self.mode = mode
        self._stats["mode_changes"] += 1
        
        if self.camera:
            if mode == TrackingMode.AUTO_TRACK:
                self.camera.set_mode(ControlMode.AUTONOMOUS)
            elif mode == TrackingMode.MANUAL:
                self.camera.set_mode(ControlMode.MANUAL_OVERRIDE)
            else:
                self.camera.set_mode(ControlMode.SAFE_MODE)
        
        logger.info(f"Tracking mode: {old_mode.value} -> {mode.value}")
    
    def process_vision_event(self, event: VisionEvent) -> TrackingDecision:
        """Process vision event and delegate tracking to camera controller."""
        from core.hardware.sonoff_camera_controller import Detection

        self._stats["total_events"] += 1
        now = time.time()

        # Rate limiting
        if now - self._last_decision_time < self.MIN_CYCLE_INTERVAL:
            return TrackingDecision(should_move=False, action="skip", reason="cycle_interval")
        self._last_decision_time = now

        # Mode check
        if self.mode == TrackingMode.DISABLED:
            return TrackingDecision(should_move=False, action="none", reason="tracking_disabled")
        if self.mode == TrackingMode.MANUAL:
            return TrackingDecision(should_move=False, action="none", reason="manual_mode")

        # Delegate to camera controller
        if not self.camera:
            return TrackingDecision(should_move=False, action="none", reason="no_camera")

        if not event.detection_found:
            # No detection - check if target lost
            self.camera.check_target_lost()
            return TrackingDecision(should_move=False, action="none", reason="no_detection")

        # Convert VisionEvent to Detection object for SonoffCameraController
        center_x_norm = event.target_center_x / event.frame_width
        center_y_norm = 0.5  # Assume vertical center if not provided

        # Create Detection object
        detection = Detection(
            person_id="tracked_person",
            bbox=(event.target_center_x - 50, 200, 100, 300),  # Synthetic bbox
            center_x=center_x_norm,
            center_y=center_y_norm,
            confidence=event.confidence
        )

        # Call camera controller's tracking with correct API
        self._stats["tracking_calls"] += 1
        moved = self.camera.process_detection(detection)

        # Calculate error for logging
        error_x = event.target_center_x - event.frame_center_x

        decision = TrackingDecision(
            should_move=moved,
            action="track" if moved else "deadzone",
            velocity=1.0 if moved else 0.0,
            duration=0.25,
            reason="tracking" if moved else "in_deadzone",
            error_x=error_x
        )

        logger.info(decision.to_log())
        self._last_decision = decision
        return decision
    
    def process_detections(self, detections: List[Dict], frame_width: int = 1920) -> TrackingDecision:
        """Process raw detections directly from perception."""
        from core.hardware.sonoff_camera_controller import Detection

        self._stats["total_events"] += 1
        now = time.time()

        if now - self._last_decision_time < self.MIN_CYCLE_INTERVAL:
            return TrackingDecision(should_move=False, action="skip", reason="cycle_interval")
        self._last_decision_time = now

        if self.mode != TrackingMode.AUTO_TRACK or not self.camera:
            return TrackingDecision(should_move=False, action="none",
                reason="disabled" if not self.camera else self.mode.value)

        if not detections:
            self.camera.check_target_lost()
            return TrackingDecision(should_move=False, action="none", reason="no_detections")

        # Find best person detection
        person_dets = [d for d in detections if d.get("class", "") == "person"]
        if not person_dets:
            person_dets = detections  # Use any detection if no person class

        # Get highest confidence detection
        best = max(person_dets, key=lambda d: d.get("confidence", 0))
        bbox = best.get("bbox", [0, 0, 0, 0])

        # Calculate center from bbox (normalized or pixel coordinates)
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Check if normalized (0-1) or pixel coordinates
            if x2 <= 1.0 and y2 <= 1.0:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
            else:
                center_x = (x1 + x2) / 2 / frame_width
                center_y = (y1 + y2) / 2 / 1080  # Assume 1080p
        else:
            center_x = 0.5
            center_y = 0.5

        # Create Detection object
        detection = Detection(
            person_id="detected_person",
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])),
            center_x=center_x,
            center_y=center_y,
            confidence=best.get("confidence", 0.5)
        )

        self._stats["tracking_calls"] += 1
        moved = self.camera.process_detection(detection)

        decision = TrackingDecision(
            should_move=moved,
            action="track" if moved else "deadzone",
            reason="tracking" if moved else "in_deadzone"
        )

        self._last_decision = decision
        return decision
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        camera_connected = False
        if self.camera:
            camera_connected = getattr(self.camera, 'is_connected', False)
            if callable(camera_connected):
                camera_connected = camera_connected
            elif hasattr(self.camera, '_connected'):
                camera_connected = self.camera._connected
        return {
            **self._stats,
            "mode": self.mode.value,
            "camera_connected": camera_connected
        }
    
    def emergency_stop(self):
        """Emergency stop - disable tracking and stop camera."""
        logger.warning("EMERGENCY STOP from Orchestrator")
        self.mode = TrackingMode.DISABLED
        if self.camera:
            self.camera.emergency_stop()


_orchestrator = None

def get_ptz_orchestrator() -> PTZOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PTZOrchestrator()
    return _orchestrator
