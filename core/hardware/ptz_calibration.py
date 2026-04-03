#!/usr/bin/env python3
"""
PTZ Calibration System
======================

Assisted calibration for accurate PTZ limit mapping.
User manually moves camera to physical limits, system records actual positions.

Usage:
    1. Start calibration mode
    2. Move camera to each limit (left, right, up, down)
    3. Confirm each position
    4. Limits saved to config/ptz_limits.json

Author: M.O.L.O.C.H. System
Date: 2026-02-04
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# Config file path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "ptz_limits.json"


class CalibrationStep(Enum):
    """Calibration steps."""
    IDLE = "idle"
    PAN_LEFT = "pan_left"      # Move to full left
    PAN_RIGHT = "pan_right"    # Move to full right
    TILT_UP = "tilt_up"        # Move to full up
    TILT_DOWN = "tilt_down"    # Move to full down
    COMPLETE = "complete"


@dataclass
class PTZLimits:
    """Calibrated PTZ limits."""
    pan_min: float = -168.4    # Right limit (negative)
    pan_max: float = 170.0     # Left limit (positive)
    tilt_min: float = -78.0    # Down limit (negative)
    tilt_max: float = 78.8     # Up limit (positive)
    calibrated: bool = False
    calibration_date: Optional[str] = None

    @property
    def pan_range(self) -> float:
        """Total pan range in degrees."""
        return self.pan_max - self.pan_min

    @property
    def tilt_range(self) -> float:
        """Total tilt range in degrees."""
        return self.tilt_max - self.tilt_min

    def clamp_pan(self, value: float) -> float:
        """Clamp pan value to calibrated limits."""
        return max(self.pan_min, min(self.pan_max, value))

    def clamp_tilt(self, value: float) -> float:
        """Clamp tilt value to calibrated limits."""
        return max(self.tilt_min, min(self.tilt_max, value))


class PTZCalibration:
    """
    Assisted PTZ calibration system.

    Guides user through moving camera to physical limits
    and records actual ONVIF position values.
    """

    STEP_INSTRUCTIONS = {
        CalibrationStep.PAN_LEFT: "Bewege Kamera nach LINKS bis zum Anschlag",
        CalibrationStep.PAN_RIGHT: "Bewege Kamera nach RECHTS bis zum Anschlag",
        CalibrationStep.TILT_UP: "Bewege Kamera nach OBEN bis zum Anschlag",
        CalibrationStep.TILT_DOWN: "Bewege Kamera nach UNTEN bis zum Anschlag",
    }

    def __init__(self, camera_controller=None):
        """
        Initialize calibration system.

        Args:
            camera_controller: SonoffCameraController instance for reading position
        """
        self.camera = camera_controller
        self.limits = PTZLimits()
        self.current_step = CalibrationStep.IDLE

        # Recorded values during calibration
        self._recorded = {
            "pan_left": None,
            "pan_right": None,
            "tilt_up": None,
            "tilt_down": None,
        }

        # Callbacks
        self.on_step_change: Optional[Callable[[CalibrationStep, str], None]] = None
        self.on_complete: Optional[Callable[[PTZLimits], None]] = None

        # Load existing limits
        self.load_limits()

    def set_camera(self, camera_controller):
        """Set camera controller for position reading."""
        self.camera = camera_controller

    def load_limits(self) -> PTZLimits:
        """Load limits from config file."""
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r') as f:
                    data = json.load(f)

                limits_data = data.get("limits", {})
                self.limits = PTZLimits(
                    pan_min=limits_data.get("pan_min", -168.4),
                    pan_max=limits_data.get("pan_max", 170.0),
                    tilt_min=limits_data.get("tilt_min", -78.0),
                    tilt_max=limits_data.get("tilt_max", 78.8),
                    calibrated=data.get("calibrated", False),
                    calibration_date=data.get("calibration_date")
                )
                logger.info(f"Loaded PTZ limits: pan=[{self.limits.pan_min}, {self.limits.pan_max}], "
                           f"tilt=[{self.limits.tilt_min}, {self.limits.tilt_max}], "
                           f"calibrated={self.limits.calibrated}")
            else:
                logger.warning(f"PTZ limits config not found: {CONFIG_PATH}")
                self.limits = PTZLimits()
        except Exception as e:
            logger.error(f"Failed to load PTZ limits: {e}")
            self.limits = PTZLimits()

        return self.limits

    def save_limits(self) -> bool:
        """Save current limits to config file."""
        try:
            data = {
                "calibrated": self.limits.calibrated,
                "calibration_date": self.limits.calibration_date,
                "limits": {
                    "pan_min": self.limits.pan_min,
                    "pan_max": self.limits.pan_max,
                    "tilt_min": self.limits.tilt_min,
                    "tilt_max": self.limits.tilt_max,
                },
                "notes": f"Calibrated via assisted mode on {self.limits.calibration_date}"
            }

            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved PTZ limits to {CONFIG_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to save PTZ limits: {e}")
            return False

    def get_current_position(self) -> Optional[Dict[str, float]]:
        """
        Read current position from camera via ONVIF GetStatus.

        Returns:
            Dict with 'pan' and 'tilt' in degrees, or None on error
        """
        if not self.camera:
            logger.error("No camera controller set")
            return None

        if not self.camera.is_connected:
            logger.error("Camera not connected")
            return None

        try:
            # Get status via ONVIF
            status = self.camera.ptz_service.GetStatus({
                'ProfileToken': self.camera.profile_token
            })

            # Extract position
            from zeep.helpers import serialize_object
            status_dict = serialize_object(status)

            pos = status_dict.get('Position', {})
            pan_tilt = pos.get('PanTilt', {})

            pan = float(pan_tilt.get('x', 0))
            tilt = float(pan_tilt.get('y', 0))

            logger.info(f"Current position: pan={pan:.2f}, tilt={tilt:.2f}")
            return {"pan": pan, "tilt": tilt}

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    def start_calibration(self) -> bool:
        """Start the calibration process."""
        if not self.camera or not self.camera.is_connected:
            logger.error("Cannot start calibration: camera not connected")
            return False

        # Reset recorded values
        self._recorded = {
            "pan_left": None,
            "pan_right": None,
            "tilt_up": None,
            "tilt_down": None,
        }

        # Start with first step
        self._set_step(CalibrationStep.PAN_LEFT)
        logger.info("=== PTZ CALIBRATION STARTED ===")
        return True

    def confirm_position(self) -> bool:
        """
        Confirm current position for the current calibration step.
        Records the position and advances to next step.

        Returns:
            True if position recorded successfully
        """
        if self.current_step == CalibrationStep.IDLE:
            logger.warning("Calibration not started")
            return False

        if self.current_step == CalibrationStep.COMPLETE:
            logger.warning("Calibration already complete")
            return False

        # Read current position
        pos = self.get_current_position()
        if not pos:
            logger.error("Failed to read position")
            return False

        # Record based on current step
        if self.current_step == CalibrationStep.PAN_LEFT:
            self._recorded["pan_left"] = pos["pan"]
            logger.info(f"Recorded PAN LEFT: {pos['pan']:.2f}")
            self._set_step(CalibrationStep.PAN_RIGHT)

        elif self.current_step == CalibrationStep.PAN_RIGHT:
            self._recorded["pan_right"] = pos["pan"]
            logger.info(f"Recorded PAN RIGHT: {pos['pan']:.2f}")
            self._set_step(CalibrationStep.TILT_UP)

        elif self.current_step == CalibrationStep.TILT_UP:
            self._recorded["tilt_up"] = pos["tilt"]
            logger.info(f"Recorded TILT UP: {pos['tilt']:.2f}")
            self._set_step(CalibrationStep.TILT_DOWN)

        elif self.current_step == CalibrationStep.TILT_DOWN:
            self._recorded["tilt_down"] = pos["tilt"]
            logger.info(f"Recorded TILT DOWN: {pos['tilt']:.2f}")
            self._finalize_calibration()

        return True

    def _finalize_calibration(self):
        """Calculate and save final limits."""
        # Determine actual min/max
        # pan_left is typically positive (left = positive pan)
        # pan_right is typically negative (right = negative pan)
        pan_left = self._recorded["pan_left"]
        pan_right = self._recorded["pan_right"]
        tilt_up = self._recorded["tilt_up"]
        tilt_down = self._recorded["tilt_down"]

        # Assign min/max based on actual values
        self.limits.pan_max = max(pan_left, pan_right)  # Left = max
        self.limits.pan_min = min(pan_left, pan_right)  # Right = min
        self.limits.tilt_max = max(tilt_up, tilt_down)  # Up = max
        self.limits.tilt_min = min(tilt_up, tilt_down)  # Down = min
        self.limits.calibrated = True
        self.limits.calibration_date = datetime.now().isoformat()

        # Save to file
        self.save_limits()

        # Log results
        logger.info("=" * 50)
        logger.info("=== PTZ CALIBRATION COMPLETE ===")
        logger.info(f"Pan range:  {self.limits.pan_min:.2f} to {self.limits.pan_max:.2f} ({self.limits.pan_range:.1f} deg total)")
        logger.info(f"Tilt range: {self.limits.tilt_min:.2f} to {self.limits.tilt_max:.2f} ({self.limits.tilt_range:.1f} deg total)")
        logger.info("=" * 50)

        self._set_step(CalibrationStep.COMPLETE)

        if self.on_complete:
            self.on_complete(self.limits)

    def cancel_calibration(self):
        """Cancel calibration and return to idle."""
        self._recorded = {
            "pan_left": None,
            "pan_right": None,
            "tilt_up": None,
            "tilt_down": None,
        }
        self._set_step(CalibrationStep.IDLE)
        logger.info("Calibration cancelled")

    def _set_step(self, step: CalibrationStep):
        """Set current step and notify callback."""
        self.current_step = step
        instruction = self.STEP_INSTRUCTIONS.get(step, "")

        logger.info(f"Calibration step: {step.value} - {instruction}")

        if self.on_step_change:
            self.on_step_change(step, instruction)

    def get_step_instruction(self) -> str:
        """Get instruction for current step."""
        return self.STEP_INSTRUCTIONS.get(self.current_step, "")

    def get_status(self) -> Dict[str, Any]:
        """Get calibration status."""
        return {
            "current_step": self.current_step.value,
            "instruction": self.get_step_instruction(),
            "recorded": self._recorded.copy(),
            "limits": asdict(self.limits),
            "camera_connected": self.camera.is_connected if self.camera else False
        }


# Singleton instance
_calibration: Optional[PTZCalibration] = None


def get_ptz_calibration() -> PTZCalibration:
    """Get or create singleton PTZCalibration instance."""
    global _calibration
    if _calibration is None:
        _calibration = PTZCalibration()
    return _calibration


def get_ptz_limits() -> PTZLimits:
    """Get current PTZ limits (loads from config if needed)."""
    return get_ptz_calibration().limits
